from __future__ import annotations

import difflib
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from app.artifacts import ArtifactStore
from app.db import Database
from app.models import SemanticTimeline
from app.settings import Settings
from pipeline import local_models
from pipeline.captions import build_cc, finalize_events
from pipeline.errors import ArtifactError, ConfigurationError, PipelineError
from pipeline.llm import StructuredLLM, load_prompt, logged_cost
from pipeline.media import detect_silences, probe, run
from pipeline.qc import run_qc
from pipeline.sarvam import SarvamClient, mark_overlaps
from pipeline.scenes import UNKNOWN_VISUAL, aggregate_visual, color_distance, build_scenes, candidate_boundaries, merge_scenes, scene_digest, score_boundaries
from pipeline.ads import decide, generate_candidates, load_catalogue
from pipeline.scoring import curve_mean
from pipeline.subtitles import format_utterances, write_srt, write_vtt


LOCATIONS = [
    "living room", "bedroom", "kitchen", "dining room", "office", "restaurant", "cafe", "street", "market",
    "car", "hospital", "police station", "classroom", "railway platform", "river ghat", "terrace", "courtyard",
    "temple", "shop", "garden", "corridor", "stairwell",
]
TARGETED_KINDS = {"product", "brand", "food"}
SCENE_MAX_SECONDS = 300.0


@dataclass
class StageContext:
    episode_id: str
    settings: Settings
    store: ArtifactStore
    db: Database

    @property
    def root(self) -> Path:
        return self.store.episode_dir(self.episode_id)

    @property
    def t(self):
        return self.settings.thresholds

    def stage(self, stage_id: str) -> dict:
        path = self.store.stage_path(self.episode_id, stage_id)
        if not path.exists():
            raise ArtifactError(f"Required stage artifact is missing: {path.name}")
        return self.store.read_json(path)

    def llm(self) -> StructuredLLM:
        return StructuredLLM(self.settings, self.store, self.episode_id)

    def config(self, name: str) -> dict:
        return yaml.safe_load((self.settings.config_dir / name).read_text(encoding="utf-8")) or {}


def _dhash(path: Path) -> str | None:
    return _signature(path)[0]


def _signature(path: Path) -> tuple[str | None, list[float] | None]:
    """dHash (structure) plus mean RGB (colour, which dHash ignores)."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None, None
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8))
        flattened = getattr(resized, "get_flattened_data", resized.getdata)
        pixels = list(flattened())
        color = [round(float(c), 1) for c in image.convert("RGB").resize((1, 1)).getpixel((0, 0))]
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}", color


def _hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def edit_ratio(raw: str, cleaned: str) -> float:
    return _levenshtein(raw, cleaned) / max(1, len(raw))


def _conservative_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([।?!,])", r"\1", text)
    if text and not text.endswith(("।", ".", "?", "!")):
        text += "।"
    return text


def _fuzzy(a: str, b: str) -> bool:
    a, b = a.casefold().strip(), b.casefold().strip()
    if not a or not b:
        return False
    return a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= .8


# --------------------------------------------------------------------------- s01–s05: media


def s01_ingest(ctx: StageContext) -> dict:
    record = ctx.db.get_episode_record(ctx.episode_id)
    if record is None:
        raise ArtifactError("Episode database record is missing.")
    source = Path(record.source_path)
    if not source.exists():
        raise ArtifactError("The registered source video no longer exists. If the server restarted, DATA_DIR is probably not on a persistent disk.")
    metadata = probe(ctx.settings.ffprobe_binary, source, ctx.settings.ffmpeg_binary)
    if metadata["duration"] <= 0:
        raise PipelineError("Could not read a valid video duration.")
    proxy = ctx.root / "proxy.mp4"
    browser_ready = source.suffix.lower() == ".mp4" and metadata["video_codec"] == "h264" and (metadata["audio_codec"] in {"aac", None})
    if browser_ready and metadata["height"] <= 720:
        proxy_path = source
    else:
        run([
            ctx.settings.ffmpeg_binary, "-y", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", "scale=-2:min(720\\,ih)", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(proxy),
        ])
        proxy_path = proxy
    ctx.db.update_episode(
        ctx.episode_id, duration=metadata["duration"], fps=metadata["fps"], width=metadata["width"],
        height=metadata["height"], video_available=True,
    )
    return {**metadata, "source": str(source.relative_to(ctx.root)), "proxy": str(proxy_path.relative_to(ctx.root))}


def _ffmpeg_cuts(ctx: StageContext, source: Path) -> list[float]:
    command = [
        ctx.settings.ffmpeg_binary, "-hide_banner", "-i", str(source), "-vf",
        f"select='gt(scene,{ctx.t.shot_scene_threshold})',showinfo", "-an", "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
    if result.returncode:
        raise PipelineError("ffmpeg shot detection failed.")
    return [float(value) for value in re.findall(r"pts_time:([0-9.]+)", result.stderr)]


def merge_short_shots(points: list[float], min_seconds: float) -> list[float]:
    """Drop cut points that would create shots shorter than min_seconds (merging into the neighbour)."""
    points = sorted(set(points))
    changed = True
    while changed and len(points) > 2:
        changed = False
        for index in range(len(points) - 1):
            if points[index + 1] - points[index] < min_seconds:
                # Remove the interior edge of the short shot; keep the episode start/end.
                remove = index + 1 if index + 1 < len(points) - 1 else index
                if remove == 0:
                    continue
                points.pop(remove)
                changed = True
                break
    return points


def s02_shots(ctx: StageContext) -> dict:
    ingest = ctx.stage("s01_ingest")
    source = ctx.root / ingest["proxy"]
    duration = ingest["duration"]
    cuts = local_models.pyscenedetect_cuts(source)
    detector = "pyscenedetect_adaptive"
    if cuts is None:
        cuts = _ffmpeg_cuts(ctx, source)
        detector = "ffmpeg_scene"
    points = [0.0] + [round(value, 3) for value in cuts if 0 < value < duration] + [round(duration, 3)]
    points = merge_short_shots(points, ctx.t.min_shot_seconds)
    shots = [
        {"shot_id": f"shot_{index + 1:05d}", "start": start, "end": end, "keyframe": None, "dup_of": None, "embedding_ref": None}
        for index, (start, end) in enumerate(zip(points, points[1:]))
    ]
    if not shots:
        shots = [{"shot_id": "shot_00001", "start": 0.0, "end": duration, "keyframe": None, "dup_of": None, "embedding_ref": None}]
    return {"shots": shots, "detector": detector, "threshold": ctx.t.shot_scene_threshold}


def _extract_frame(ctx: StageContext, source: Path, time_value: float, destination: Path) -> None:
    run([ctx.settings.ffmpeg_binary, "-y", "-ss", f"{time_value:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "3", str(destination)], timeout=120)


def s03_keyframes(ctx: StageContext) -> dict:
    ingest, shot_data = ctx.stage("s01_ingest"), ctx.stage("s02_shots")
    source = ctx.root / ingest["proxy"]
    shots = []
    for shot in shot_data["shots"]:
        relative = f"frames/kf_{shot['shot_id']}.jpg"
        _extract_frame(ctx, source, (shot["start"] + shot["end"]) / 2, ctx.root / relative)
        extra = []
        if shot["end"] - shot["start"] > ctx.t.keyframe_second_frame_after:
            second = f"frames/kf_{shot['shot_id']}_b.jpg"
            _extract_frame(ctx, source, shot["start"] + (shot["end"] - shot["start"]) * .75, ctx.root / second)
            extra.append(second)
        signature, color = _signature(ctx.root / relative)
        shots.append({**shot, "frame": relative, "extra_frames": extra, "dhash": signature, "color": color})

    method = "dhash" if any(shot["dhash"] for shot in shots) else "disabled"
    vectors: list[list[float]] | None = None
    if ctx.settings.frame_embedder == "siglip":
        embedder = local_models.siglip_embedder(ctx.settings.device)
        if embedder is None:
            raise ConfigurationError("FRAME_EMBEDDER=siglip needs torch, transformers and Pillow installed.")
        vectors = embedder.embed([ctx.root / shot["frame"] for shot in shots])
        import numpy as np

        np.save(ctx.root / "stages" / "embeddings.npy", np.asarray(vectors, dtype="float32"))
        method = "siglip"

    window = ctx.t.keyframe_dedup_window
    for index, shot in enumerate(shots):
        duplicate = None
        for previous in reversed(shots[max(0, index - window):index]):
            if vectors is not None:
                a, b = vectors[index], vectors[shots.index(previous)]
                similar = sum(x * y for x, y in zip(a, b)) >= ctx.t.keyframe_dedup_similarity
            elif shot["dhash"] and previous["dhash"]:
                similar = (
                    _hamming(shot["dhash"], previous["dhash"]) <= ctx.t.keyframe_dedup_hamming
                    and (color_distance(shot["color"], previous["color"]) or 0) <= .1
                )
            else:
                similar = False
            if similar:
                duplicate = previous["dup_of"] or previous["shot_id"]
                break
        shot["dup_of"] = duplicate
        shot["keyframe"] = None if duplicate else shot["frame"]
        shot["embedding_ref"] = str(index) if vectors is not None else None
    unique = sum(shot["dup_of"] is None for shot in shots)
    return {"shots": shots, "dedup_method": method, "unique_frames": unique, "dedup_ratio": round(1 - unique / max(1, len(shots)), 3)}


def _demucs(ctx: StageContext, full: Path, vocals: Path, duration: float) -> bool:
    if not shutil.which("demucs"):
        return False
    work = ctx.root / "audio" / "demucs"
    segment, overlap = 600.0, 5.0
    pieces: list[tuple[Path, float]] = []
    try:
        if duration <= 1800:
            run(["demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(work), str(full)])
            pieces.append((next(work.rglob("vocals.wav")), 0.0))
        else:
            # Long episodes: 10-minute segments with 5 s overlap to bound memory.
            for index in range(math.ceil(duration / segment)):
                start = max(0.0, index * segment - overlap)
                part = ctx.root / "audio" / f"seg_{index:03d}.wav"
                run([ctx.settings.ffmpeg_binary, "-y", "-ss", str(start), "-t", str(segment + (overlap if index else 0)), "-i", str(full), str(part)])
                out = work / f"seg_{index:03d}"
                run(["demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(out), str(part)])
                pieces.append((next(out.rglob("vocals.wav")), overlap if index else 0.0))
        trimmed = []
        for index, (path, skip) in enumerate(pieces):
            target = ctx.root / "audio" / f"voc_{index:03d}.wav"
            run([ctx.settings.ffmpeg_binary, "-y", "-ss", str(skip), "-i", str(path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)])
            trimmed.append(target)
        if len(trimmed) == 1:
            shutil.move(trimmed[0], vocals)
        else:
            listing = ctx.root / "audio" / "concat.txt"
            listing.write_text("".join(f"file '{path.as_posix()}'\n" for path in trimmed), encoding="utf-8")
            run([ctx.settings.ffmpeg_binary, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(vocals)])
        return True
    except (PipelineError, StopIteration):
        return False


def s04_audio_prep(ctx: StageContext) -> dict:
    ingest = ctx.stage("s01_ingest")
    if not ingest["has_audio"]:
        raise PipelineError("The video has no audio stream to transcribe.")
    source = ctx.root / ingest["source"]
    full = ctx.root / "audio" / "full_16k.wav"
    vocals = ctx.root / "audio" / "vocals_16k.wav"
    run([ctx.settings.ffmpeg_binary, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(full)])
    separated = _demucs(ctx, full, vocals, ingest["duration"])
    if not separated:
        shutil.copy2(full, vocals)
    return {"full_audio": "audio/full_16k.wav", "vocals_audio": "audio/vocals_16k.wav", "vocal_separation": separated, "sample_rate": 16000}


def build_chunks(speech: list[dict], max_seconds: float) -> list[dict]:
    """Merge speech segments into STT chunks no longer than max_seconds, splitting only at silences."""
    chunks: list[dict] = []
    for segment in speech:
        if chunks and segment["end"] - chunks[-1]["start"] <= max_seconds:
            chunks[-1]["end"] = segment["end"]
        else:
            chunks.append({"start": segment["start"], "end": segment["end"]})
    return chunks


def s05_vad(ctx: StageContext) -> dict:
    ingest, audio = ctx.stage("s01_ingest"), ctx.stage("s04_audio_prep")
    source = ctx.root / audio["vocals_audio"]
    duration = ingest["duration"]
    min_silence = ctx.t.vad_min_silence_seconds
    speech = local_models.silero_speech(source, int(min_silence * 1000))
    if speech is not None:
        method = "silero_vad"
        silences, cursor = [], 0.0
        for segment in speech + [{"start": duration, "end": duration}]:
            if segment["start"] - cursor >= min_silence:
                silences.append({"start": round(cursor, 3), "end": round(segment["start"], 3), "duration": round(segment["start"] - cursor, 3)})
            cursor = max(cursor, segment["end"])
    else:
        method = "ffmpeg_silencedetect"
        silences = detect_silences(ctx.settings.ffmpeg_binary, source, min_seconds=min_silence)
        speech, cursor = [], 0.0
        for silence in silences:
            if silence["start"] > cursor + .15:
                speech.append({"start": round(cursor, 3), "end": round(silence["start"], 3)})
            cursor = silence["end"]
        if cursor < duration - .15:
            speech.append({"start": round(cursor, 3), "end": round(duration, 3)})
    return {"speech_segments": speech, "silences": silences, "chunks": build_chunks(speech, 30.0), "method": method}


# --------------------------------------------------------------------------- s06–s08: audio understanding


def s06_stt(ctx: StageContext) -> dict:
    audio = ctx.stage("s04_audio_prep")
    vocals = ctx.root / audio["vocals_audio"]
    raw_path = ctx.root / "stages" / "s06_raw.json"
    audio_hash = ctx.store.file_hash(vocals)
    if raw_path.exists():
        # Re-parse the stored response instead of paying for another transcription.
        stored = ctx.store.read_json(raw_path)
        if stored.get("_audio_hash") == audio_hash:
            utterances = SarvamClient.normalize(stored["response"])
            if utterances:
                return {"utterances": utterances, "provider": "sarvam", "model": "saaras:v4", "language": "bn-IN", "diarization": True, "reparsed": True}
    client = SarvamClient(ctx.settings.sarvam_api_key)
    utterances, raw = client.transcribe(vocals, ctx.root / "stages" / "sarvam_output")
    if not utterances:
        raise PipelineError("Sarvam returned an empty transcript.")
    ctx.store.write_json(raw_path, {"_audio_hash": audio_hash, "response": raw})
    return {"utterances": mark_overlaps(utterances), "provider": "sarvam", "model": "saaras:v4", "language": "bn-IN", "diarization": True, "reparsed": False}


class CleanItem(BaseModel):
    utt_id: str
    text: str


class CleanBatch(BaseModel):
    items: list[CleanItem]


def s07_transcript_clean(ctx: StageContext) -> dict:
    utterances = ctx.stage("s06_stt")["utterances"]
    cleaned = {u["utt_id"]: _conservative_clean(u["text_raw"]) for u in utterances}
    rejected: list[str] = []
    llm = ctx.llm()
    method = "deterministic_conservative"
    if llm.available() and utterances:
        method = "llm_punctuation_restore"
        system = load_prompt("s07_cleanup_v1")
        size, context_size = ctx.t.cleanup_batch_size, ctx.t.cleanup_context
        for offset in range(0, len(utterances), size):
            batch = utterances[offset:offset + size]
            context = utterances[max(0, offset - context_size):offset]
            user = ""
            if context:
                user += "Context (do not return):\n" + "\n".join(f"{u['utt_id']}: {u['text_raw']}" for u in context) + "\n\n"
            user += "Clean these:\n" + "\n".join(f"{u['utt_id']}: {u['text_raw']}" for u in batch)
            response = llm.call(model=ctx.settings.llm_model_default, system=system, user=user, schema=CleanBatch, stage="s07_transcript_clean")
            by_id = {item.utt_id: item.text.strip() for item in response.items}
            for utterance in batch:
                candidate = by_id.get(utterance["utt_id"])
                if not candidate:
                    continue
                if edit_ratio(utterance["text_raw"], candidate) <= ctx.t.cleanup_max_edit_ratio:
                    cleaned[utterance["utt_id"]] = candidate
                else:
                    rejected.append(utterance["utt_id"])
                    cleaned[utterance["utt_id"]] = utterance["text_raw"].strip()
    output = [{**u, "text": cleaned[u["utt_id"]]} for u in utterances]
    return {"utterances": output, "cleanup": method, "rejected": rejected}


def s08_audio_events(ctx: StageContext) -> dict:
    import numpy as np

    audio = ctx.stage("s04_audio_prep")
    path = ctx.root / audio["full_audio"]
    samples, rate = local_models.read_wav_mono(path)
    seconds = max(1, math.ceil(len(samples) / rate))
    padded = np.pad(samples, (0, seconds * rate - len(samples)))
    rms = np.sqrt((padded.reshape(seconds, rate) ** 2).mean(axis=1))
    low, high = float(rms.min()), float(rms.max())
    loudness = [[float(i), round((float(v) - low) / max(1e-5, high - low), 4)] for i, v in enumerate(rms)]

    labels_config: dict[str, dict] = ctx.config("sound_labels_bn.yaml")
    result = local_models.panns_windows(path, ctx.root / "audio", ctx.settings.ffmpeg_binary, ctx.settings.device)
    events: list[dict] = []
    music: list[list[float]] = []
    detector = "not_installed"
    if result is not None:
        detector = "panns_cnn14"
        labels, windows = result
        index_of = {label: i for i, label in enumerate(labels)}
        for label, config in labels_config.items():
            if label not in index_of:
                continue
            threshold = float(config.get("threshold", ctx.t.audio_event_default_threshold))
            current: dict | None = None
            for start, scores in windows:
                score = scores[index_of[label]]
                if score >= threshold:
                    if current and start <= current["end"]:
                        current["end"], current["score"] = start + 1.0, max(current["score"], score)
                    else:
                        current = {"label": label, "start": start, "end": start + 1.0, "score": score}
                        events.append(current)
                else:
                    current = None
        for event in events:
            config = labels_config[event["label"]]
            event["label_bn"] = config["bn"]
            event["in_cc"] = event["end"] - event["start"] >= ctx.t.audio_event_min_cc_seconds and event["score"] >= float(config.get("cc_threshold", .35))
            event["score"] = round(float(event["score"]), 4)
            event["start"], event["end"] = round(event["start"], 3), round(event["end"], 3)
        events.sort(key=lambda e: e["start"])
        for i, event in enumerate(events):
            event["event_id"] = f"ae_{i + 1:05d}"
        if "Music" in index_of:
            per_second: dict[int, list[float]] = {}
            for start, scores in windows:
                per_second.setdefault(int(start), []).append(scores[index_of["Music"]])
            music = [[float(second), round(mean(values), 4)] for second, values in sorted(per_second.items())]
    return {
        "events": events, "loudness": loudness, "music": music, "event_detector": detector,
        "loudness_method": "pcm_rms_1hz_minmax",
        "note": None if result is not None else "Install panns-inference (pip install -e '.[gpu]') for sound events and music ratio.",
    }


# --------------------------------------------------------------------------- s09–s13: visual + semantic


class VisualItem(BaseModel):
    shot_id: str
    location: str
    indoor: bool | None
    objects: list[str]
    visible_brands: list[str]
    activity: str
    visual_mood: str
    people_count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)


class VisualBatch(BaseModel):
    items: list[VisualItem]


MOODS = ["tense", "joyful", "calm", "sad", "romantic", "suspenseful", "action-packed"]
_OBJECT_CONF = 0.35
_BRAND_PROBABILITY = 0.6
_NO_BRAND_PROMPT = "a photo with no logos or brand names visible"


def _siglip_probability(positive_embed: list[float], negative_embed: list[float], image_embed: list[float]) -> float:
    """Softmax of (positive, negative) prompt similarity. SigLIP's raw cosine similarities sit on a shared
    baseline (near 0.8 even for unrelated prompts), so a flat similarity threshold misfires; this pairwise
    contrast is the calibration SigLIP/CLIP zero-shot classification is built around."""
    pos_score, neg_score = sum(a * b for a, b in zip(image_embed, positive_embed)) * 10, sum(a * b for a, b in zip(image_embed, negative_embed)) * 10
    return math.exp(pos_score) / (math.exp(pos_score) + math.exp(neg_score))


def _local_visual_tags(ctx: StageContext, unique: list[dict]) -> tuple[dict[str, dict], str] | None:
    """Free CPU-only replacement for the s09 OpenAI vision call. Each model degrades independently and
    records what ran, the same way PANNs/Silero do for audio (pipeline/stages/core.py:s08_audio_events):
    YOLOv8n for objects/people, SigLIP zero-shot for location/mood/brands, EasyOCR as a second brand
    signal. Returns None only if neither YOLO nor SigLIP is installed — nothing local left to tag with."""
    if not unique:
        return None
    paths = [ctx.root / shot["keyframe"] for shot in unique]
    yolo_results = local_models.yolo_detect(paths, ctx.settings.device, conf=_OBJECT_CONF)
    embedder = local_models.siglip_embedder(ctx.settings.device)
    ocr_results = local_models.ocr_texts(paths)
    if yolo_results is None and embedder is None:
        return None
    detections = yolo_results if yolo_results is not None else [[] for _ in paths]
    ocr_hits = ocr_results if ocr_results is not None else [[] for _ in paths]
    method = "+".join(name for name, present in
                       [("yolo", yolo_results is not None), ("siglip", embedder is not None), ("ocr", ocr_results is not None)] if present)

    brand_names = [brand["name"] for brand in load_catalogue(ctx.settings.brand_catalogue)]
    location_embeds = embedder.embed_text([f"a photo of a {location}" for location in LOCATIONS]) if embedder else []
    mood_embeds = embedder.embed_text([f"a {mood} scene" for mood in MOODS]) if embedder else []
    brand_embeds = embedder.embed_text([f"a photo showing the {name} brand" for name in brand_names]) if embedder and brand_names else []
    no_brand_embed = embedder.embed_text([_NO_BRAND_PROMPT])[0] if brand_embeds else None
    image_embeds = embedder.embed(paths) if embedder else [None] * len(paths)

    def best(labels: list[str], embeds: list[list[float]], image_embed: list[float] | None, default: str) -> str:
        if not embeds or image_embed is None:
            return default
        scores = [sum(a * b for a, b in zip(image_embed, text_embed)) for text_embed in embeds]
        return labels[max(range(len(scores)), key=lambda i: scores[i])]

    tags: dict[str, dict] = {}
    for shot, boxes, image_embed, texts in zip(unique, detections, image_embeds, ocr_hits):
        visible_brands = set()
        if image_embed is not None and brand_embeds:
            visible_brands = {
                name for name, text_embed in zip(brand_names, brand_embeds)
                if _siglip_probability(text_embed, no_brand_embed, image_embed) >= _BRAND_PROBABILITY
            }
        visible_brands |= {name for name in brand_names if any(_fuzzy(name, text) for text in texts)}
        tags[shot["shot_id"]] = {
            "location": best(LOCATIONS, location_embeds, image_embed, "unknown"),
            "indoor": None,
            "objects": sorted({label for label, score in boxes if label != "person" and score >= _OBJECT_CONF}),
            "visible_brands": sorted(visible_brands),
            "activity": "unknown",
            "visual_mood": best(MOODS, mood_embeds, image_embed, "neutral"),
            "people_count": sum(1 for label, score in boxes if label == "person" and score >= _OBJECT_CONF),
            "confidence": 0.6,
        }
    return tags, method


def s09_vision_baseline(ctx: StageContext) -> dict:
    shots = ctx.stage("s03_keyframes")["shots"]
    unique = [shot for shot in shots if shot["keyframe"]]
    local = _local_visual_tags(ctx, unique)
    tags, provider = (local[0], f"local_cv:{local[1]}") if local else ({}, None)
    llm = ctx.llm()
    if not tags and llm.available():
        system = load_prompt("s09_vision_v1", locations=", ".join(LOCATIONS))
        for offset in range(0, len(unique), 4):
            batch = unique[offset:offset + 4]
            response = llm.call(
                model=ctx.settings.llm_model_default, system=system,
                user="Images in order, one per shot ID: " + ", ".join(shot["shot_id"] for shot in batch),
                schema=VisualBatch, stage="s09_vision_baseline",
                images=[ctx.root / shot["keyframe"] for shot in batch], image_detail="low",
            )
            by_id = {item.shot_id: item for item in response.items}
            for position, shot in enumerate(batch):
                item = by_id.get(shot["shot_id"]) or (response.items[position] if position < len(response.items) else None)
                if item:
                    tags[shot["shot_id"]] = {**item.model_dump(exclude={"shot_id"}), "objects": [o.lower() for o in item.objects]}
        provider = "openai"
    for shot in shots:
        if shot["shot_id"] in tags:
            continue
        # Deduplicated shots inherit the tags of the shot that represents them.
        tags[shot["shot_id"]] = dict(tags.get(shot["dup_of"] or "", UNKNOWN_VISUAL))
    return {
        "visual_tags": tags, "provider": provider or "conservative_fallback",
        "note": None if provider else "Install the 'vision' extra (pip install -e '.[vision]') or configure OPENAI_API_KEY for vision labels.",
    }


class BoundaryDecision(BaseModel):
    after_scene: int
    action: Literal["keep", "merge"]


class SceneTitle(BaseModel):
    scene: int
    title: str


class SceneMergeWindow(BaseModel):
    decisions: list[BoundaryDecision]
    titles: list[SceneTitle]


def _split_long(scenes_bounds: list[dict], scored: list[dict], duration: float) -> list[dict]:
    """Safety net: a scene longer than SCENE_MAX_SECONDS is split at its strongest internal cut."""
    bounds = sorted(scenes_bounds, key=lambda b: b["time"])
    while True:
        edges = [0.0] + [b["time"] for b in bounds] + [duration]
        long_index = next((i for i in range(len(edges) - 1) if edges[i + 1] - edges[i] > SCENE_MAX_SECONDS), None)
        if long_index is None:
            return bounds
        start, end = edges[long_index], edges[long_index + 1]
        inside = [b for b in scored if start + 20 <= b["time"] <= end - 20]
        if not inside:
            return bounds
        bounds = sorted(bounds + [max(inside, key=lambda b: b["score"])], key=lambda b: b["time"])


def s10_scenes(ctx: StageContext) -> dict:
    duration = ctx.stage("s01_ingest")["duration"]
    shots = ctx.stage("s03_keyframes")["shots"]
    utterances = ctx.stage("s07_transcript_clean")["utterances"]
    silences = ctx.stage("s05_vad")["silences"]
    tags = ctx.stage("s09_vision_baseline")["visual_tags"]
    embeddings = None
    embedding_path = ctx.root / "stages" / "embeddings.npy"
    if embedding_path.exists() and any(shot.get("embedding_ref") for shot in shots):
        import numpy as np

        matrix = np.load(embedding_path)
        embeddings = {shot["shot_id"]: matrix[int(shot["embedding_ref"])].tolist() for shot in shots if shot.get("embedding_ref")}

    scored = score_boundaries(shots, tags, utterances, silences, ctx.t, embeddings)
    chosen = _split_long(candidate_boundaries(scored, duration, ctx.t), scored, duration)
    scenes = build_scenes(shots, chosen, duration)

    llm = ctx.llm()
    titles: dict[int, str] = {}
    if llm.available() and len(scenes) > 1:
        system = load_prompt("s10_scene_merge_v1")
        merge_after: set[int] = set()
        window = ctx.t.scene_llm_window
        for offset in range(0, len(scenes) - 1, window - 1):
            group = scenes[offset:offset + window]
            if len(group) < 2:
                break
            user = "\n\n".join(f"Scene {offset + i}:\n{scene_digest(scene, shots, tags, utterances)}" for i, scene in enumerate(group))
            user += f"\n\nDecide every boundary after scenes {offset}..{offset + len(group) - 2}."
            response = llm.call(model=ctx.settings.llm_model_scenes, system=system, user=user, schema=SceneMergeWindow, stage="s10_scenes")
            for decision in response.decisions:
                if offset <= decision.after_scene < offset + len(group) - 1 and decision.action == "merge":
                    merge_after.add(decision.after_scene)
            for item in response.titles:
                if offset <= item.scene < offset + len(group):
                    titles.setdefault(item.scene, item.title)
        # Carry the title of the first scene of each merged run.
        kept_titles = [titles.get(i) for i in range(len(scenes)) if (i - 1) not in merge_after]
        scenes = merge_scenes(scenes, merge_after)
        titles = {i: title for i, title in enumerate(kept_titles) if title}

    by_id = {shot["shot_id"]: shot for shot in shots}
    output = []
    for index, scene in enumerate(scenes):
        members = [by_id[shot_id] for shot_id in scene["shot_ids"] if shot_id in by_id]
        scene_utterances = [u for u in utterances if scene["start"] <= (u["start"] + u["end"]) / 2 < scene["end"]]
        output.append({
            **scene, "visual": aggregate_visual(members, tags),
            "speakers": sorted({u["speaker"] for u in scene_utterances}), "utt_ids": [u["utt_id"] for u in scene_utterances],
            "audio_events": [], "music_ratio": 0.0, "entity_ids": [], "draft_title": titles.get(index),
        })
    return {"scenes": output, "boundaries": scored, "method": "heuristic_boundaries", "llm_validated": llm.available()}


class DraftMention(BaseModel):
    utt_id: str
    surface: str


class EntityDraft(BaseModel):
    name: str
    name_bn: str | None
    kind: Literal["product", "brand", "place", "food", "activity", "topic", "other"]
    brand: str | None
    ad_categories: list[str]
    sentiment: Literal["positive", "neutral", "negative"]
    mentions: list[DraftMention]
    confidence: float = Field(ge=0, le=1)


class EntityDrafts(BaseModel):
    entities: list[EntityDraft]


def s11_entities_dialogue(ctx: StageContext) -> dict:
    utterances, scenes = ctx.stage("s07_transcript_clean")["utterances"], ctx.stage("s10_scenes")["scenes"]
    categories: dict[str, list[str]] = ctx.config("ad_categories.yaml")
    by_utterance = {u["utt_id"]: u for u in utterances}
    entities: list[dict] = []
    llm = ctx.llm()

    def add(draft: dict, mentions: list[dict]) -> None:
        key = (draft["name"].casefold(), (draft.get("brand") or "").casefold())
        existing = next((e for e in entities if (e["name"].casefold(), (e.get("brand") or "").casefold()) == key), None)
        if existing:
            known = {(m["utt_id"], m["surface"]) for m in existing["mentions"]}
            existing["mentions"].extend(m for m in mentions if (m["utt_id"], m["surface"]) not in known)
            existing["ad_categories"] = sorted(set(existing["ad_categories"]) | set(draft["ad_categories"]))
            existing["confidence"] = max(existing["confidence"], draft["confidence"])
        else:
            entities.append({**draft, "entity_id": f"entity_{len(entities) + 1:04d}", "mentions": mentions, "presence": "unverified", "visual_check": None})

    if llm.available():
        system = load_prompt(
            "s11_entities_v1", categories=", ".join(sorted(categories)),
            hints="; ".join(f"{name}: {', '.join(terms[:8])}" for name, terms in categories.items()),
        )
        for scene in scenes:
            scene_utterances = [by_utterance[i] for i in scene["utt_ids"] if i in by_utterance]
            if not scene_utterances:
                continue
            visual = scene["visual"]
            user = (
                f"Scene visuals: location={visual['location']}, objects={', '.join(visual['objects']) or 'none'}, "
                f"brands={', '.join(visual['visible_brands']) or 'none'}\n\nUtterances:\n"
                + "\n".join(f"{u['utt_id']} [{u['speaker']} {u['start']:.1f}s]: {u['text']}" for u in scene_utterances)
            )
            drafts = llm.call(model=ctx.settings.llm_model_default, system=system, user=user, schema=EntityDrafts, stage="s11_entities_dialogue")
            for draft in drafts.entities:
                # Grounding: every mention must cite an utterance of this scene.
                grounded = [m for m in draft.mentions if m.utt_id in scene["utt_ids"] and m.utt_id in by_utterance]
                if not grounded:
                    continue
                add(
                    {"name": draft.name.strip().lower(), "name_bn": draft.name_bn, "kind": draft.kind, "brand": draft.brand,
                     "ad_categories": sorted(set(draft.ad_categories) & set(categories)), "sentiment": draft.sentiment, "confidence": draft.confidence},
                    [{"source": "dialogue", "time": by_utterance[m.utt_id]["start"], "utt_id": m.utt_id, "shot_id": None, "surface": m.surface} for m in grounded],
                )
        extractor = "openai_grounded"
    else:
        # Keyword baseline: category hint terms found verbatim in dialogue.
        for category, terms in categories.items():
            for term in sorted(terms, key=len, reverse=True):
                mentions = []
                for u in utterances:
                    match = re.search(re.escape(term), u["text"], re.IGNORECASE)
                    if match:
                        mentions.append({"source": "dialogue", "time": u["start"], "utt_id": u["utt_id"], "shot_id": None, "surface": match.group(0)})
                if mentions:
                    add({"name": term.casefold(), "name_bn": term if re.search(r"[ঀ-৿]", term) else None,
                         "kind": "food" if category == "food_delivery" else "product", "brand": None,
                         "ad_categories": [category], "sentiment": "neutral", "confidence": .6}, mentions)
        extractor = "keyword_baseline"

    for scene in scenes:
        scene["entity_ids"] = [e["entity_id"] for e in entities if any(scene["start"] <= m["time"] < scene["end"] for m in e["mentions"])]
    return {"entities": entities, "scenes": scenes, "extractor": extractor}


class PresenceCheck(BaseModel):
    visible: bool | None
    visible_timestamps: list[float]
    confidence: float = Field(ge=0, le=1)
    note: str


def _sample_window(ctx: StageContext, source: Path, entity_id: str, start: float, end: float) -> list[tuple[Path, float]]:
    """1 fps frames over [start, end], deduplicated (dHash ≈ 0.95 similarity) and thinned to entity_max_frames."""
    pattern = ctx.root / "frames" / f"tgt_{entity_id}_%03d.jpg"
    run([ctx.settings.ffmpeg_binary, "-y", "-ss", f"{start:.3f}", "-t", f"{max(1.0, end - start):.3f}", "-i", str(source), "-vf", "fps=1,scale=480:-2", "-q:v", "4", str(pattern)], timeout=600)
    frames = sorted((ctx.root / "frames").glob(f"tgt_{entity_id}_[0-9][0-9][0-9].jpg"))
    kept: list[tuple[Path, float, str | None]] = []
    for index, path in enumerate(frames):
        signature = _dhash(path)
        if signature and kept and kept[-1][2] and _hamming(signature, kept[-1][2]) <= 3:
            path.unlink(missing_ok=True)
            continue
        kept.append((path, round(start + index, 1), signature))
    limit = ctx.t.entity_max_frames
    if len(kept) > limit:
        step = (len(kept) - 1) / (limit - 1)
        chosen = {round(i * step) for i in range(limit)}
        for index, item in enumerate(kept):
            if index not in chosen:
                item[0].unlink(missing_ok=True)
        kept = [item for index, item in enumerate(kept) if index in chosen]
    return [(path, time_value) for path, time_value, _ in kept]


def _grids(ctx: StageContext, entity_id: str, frames: list[tuple[Path, float]]) -> list[Path]:
    """Stitch frames into strips of five with timestamps burned in. Without Pillow, send frames individually."""
    try:
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]
    except ImportError:
        return [path for path, _ in frames]
    grids = []
    for number, offset in enumerate(range(0, len(frames), 5)):
        group = frames[offset:offset + 5]
        images = [Image.open(path).convert("RGB") for path, _ in group]
        width, height = images[0].size
        canvas = Image.new("RGB", (width * len(images), height), "black")
        draw = ImageDraw.Draw(canvas)
        for position, (image, (_, time_value)) in enumerate(zip(images, group)):
            canvas.paste(image.resize((width, height)), (position * width, 0))
            draw.rectangle([position * width, 0, position * width + 70, 20], fill="black")
            draw.text((position * width + 4, 4), f"{time_value:.1f}s", fill="yellow")
        target = ctx.root / "frames" / f"tgt_{entity_id}_grid{number + 1}.jpg"
        canvas.save(target, quality=85)
        grids.append(target)
    return grids


def _product_vocabulary(categories: dict[str, list[str]]) -> dict[str, str]:
    return {term.casefold(): category for category, terms in categories.items() for term in terms if re.fullmatch(r"[a-z ]+", term.casefold())}


def _local_presence_probabilities(embedder: "local_models.SiglipEmbedder", label: str, frames: list[tuple[Path, float]]) -> list[float]:
    """SigLIP zero-shot presence: pairwise (positive, negative) prompt probability, per sampled frame."""
    positive, negative = embedder.embed_text([f"a photo of a {label}", f"a photo without any {label}"])
    return [_siglip_probability(positive, negative, embedding) for embedding in embedder.embed([path for path, _ in frames])]


def s12_vision_targeted(ctx: StageContext) -> dict:
    stage = ctx.stage("s11_entities_dialogue")
    scenes = stage["scenes"]
    source = ctx.root / ctx.stage("s01_ingest")["proxy"]
    shots = {shot["shot_id"]: shot for shot in ctx.stage("s03_keyframes")["shots"]}
    tags = ctx.stage("s09_vision_baseline")["visual_tags"]
    categories: dict[str, list[str]] = ctx.config("ad_categories.yaml")
    llm = ctx.llm()
    embedder = local_models.siglip_embedder(ctx.settings.device)
    system = load_prompt("s12_presence_v1")
    t = ctx.t
    entities: list[dict] = []

    for entity in stage["entities"]:
        mention_times = [m["time"] for m in entity["mentions"] if m["source"] == "dialogue"]
        scene = next((s for s in scenes if any(s["start"] <= value < s["end"] for value in mention_times)), None)
        if entity["kind"] not in TARGETED_KINDS or scene is None:
            entities.append(entity)
            continue

        # Cheap pre-check against the baseline tags of the scene's shots.
        matching_shots = [
            shot_id for shot_id in scene["shot_ids"]
            if any(_fuzzy(entity["name"], obj) for obj in tags.get(shot_id, {}).get("objects", []))
            or (entity.get("brand") and any(_fuzzy(entity["brand"], b) for b in tags.get(shot_id, {}).get("visible_brands", [])))
        ]
        if matching_shots:
            frames = [shots[s].get("frame") or shots[s].get("keyframe") for s in matching_shots if s in shots]
            entities.append({**entity, "presence": "mentioned_and_shown", "visual_check": {
                "frames_checked": [f for f in frames if f], "visible": True, "visible_frames": [f for f in frames if f],
                "confidence": round(max(tags[s].get("confidence", .7) for s in matching_shots), 3),
                "note": "Matched the baseline visual tags of the scene; no targeted call needed.",
            }})
            continue

        start, end = scene["start"], scene["end"]
        if end - start > t.entity_window_long_scene and mention_times:
            start, end = max(start, min(mention_times) - t.entity_window_padding), min(end, max(mention_times) + t.entity_window_padding)
        frames = _sample_window(ctx, source, entity["entity_id"], start, end)
        checked = [str(path.relative_to(ctx.root).as_posix()) for path, _ in frames]
        if embedder and frames:
            label = entity["name"] + (f" ({entity['brand']})" if entity.get("brand") else "")
            probabilities = _local_presence_probabilities(embedder, label, frames)
            best_prob, worst_prob = max(probabilities), min(probabilities)
            if best_prob >= t.entity_visible_confidence:
                presence, confidence = "mentioned_and_shown", best_prob
                visible_frames = sorted({checked[i] for i, p in enumerate(probabilities) if p >= t.entity_visible_confidence})
                visible, note = True, "SigLIP zero-shot matched the sampled frames."
            elif worst_prob <= 1 - t.entity_absent_confidence and len(frames) >= t.entity_absent_min_frames:
                presence, confidence, visible_frames = "mentioned_only", 1 - worst_prob, []
                visible, note = False, "SigLIP zero-shot found no match across the sampled frames."
            else:
                presence, confidence, visible_frames = "unverified", best_prob, []
                visible, note = None, "SigLIP zero-shot was inconclusive."
            visual_check = {"frames_checked": checked, "visible": visible, "visible_frames": visible_frames, "confidence": round(confidence, 3), "note": note}
        elif llm.available() and frames:
            check = llm.call(
                model=ctx.settings.llm_model_default, system=system,
                user=f"Is a {entity['name']}{' (' + entity['brand'] + ')' if entity.get('brand') else ''} visible? Return the timestamps of the frames where it is visible, and a confidence.",
                schema=PresenceCheck, stage="s12_vision_targeted",
                images=_grids(ctx, entity["entity_id"], frames), image_detail="high",
            )
            visible_frames = [
                checked[min(range(len(frames)), key=lambda i: abs(frames[i][1] - stamp))] for stamp in check.visible_timestamps
            ] if frames else []
            if check.visible is True and check.confidence >= t.entity_visible_confidence:
                presence = "mentioned_and_shown"
            elif check.visible is False and check.confidence >= t.entity_absent_confidence and len(frames) >= t.entity_absent_min_frames:
                presence = "mentioned_only"
            else:
                presence = "unverified"
            visual_check = {"frames_checked": checked, "visible": check.visible, "visible_frames": sorted(set(visible_frames)), "confidence": check.confidence, "note": check.note}
        else:
            presence = "unverified"
            visual_check = {"frames_checked": checked, "visible": None, "visible_frames": [], "confidence": 0.0,
                            "note": "Targeted vision provider is not configured; no absence claim was made."}
        entities.append({**entity, "presence": presence, "visual_check": visual_check})

    # Visual-only entities: legible brands and product-like objects nobody talks about.
    vocabulary = _product_vocabulary(categories)
    for scene in scenes:
        spoken = [e for e in entities if e["entity_id"] in scene["entity_ids"]]
        candidates = [(brand, "brand", None) for brand in scene["visual"].get("visible_brands", [])]
        candidates += [(obj, "product", vocabulary[obj.casefold()]) for obj in scene["visual"].get("objects", []) if obj.casefold() in vocabulary]
        for name, kind, category in candidates:
            if any(_fuzzy(name, e["name"]) or (e.get("brand") and _fuzzy(name, e["brand"])) for e in spoken):
                continue
            shot_id = next((s for s in scene["shot_ids"] if name in tags.get(s, {}).get("objects", []) + tags.get(s, {}).get("visible_brands", [])), scene["shot_ids"][0] if scene["shot_ids"] else None)
            mention = {"source": "visual", "time": shots[shot_id]["start"] if shot_id in shots else scene["start"], "utt_id": None, "shot_id": shot_id, "surface": name}
            existing = next((e for e in entities if e["presence"] == "shown_only" and e["name"].casefold() == name.casefold()), None)
            if existing:
                existing["mentions"].append(mention)
                entity_id = existing["entity_id"]
            else:
                entity_id = f"entity_{len(entities) + 1:04d}"
                entities.append({
                    "entity_id": entity_id, "name": name.lower() if kind == "product" else name, "name_bn": None, "kind": kind,
                    "brand": name if kind == "brand" else None, "ad_categories": [category] if category else [],
                    "mentions": [mention], "sentiment": "neutral", "presence": "shown_only", "visual_check": None,
                    "confidence": round(float(scene["visual"].get("confidence", .5)), 3),
                })
            if entity_id not in scene["entity_ids"]:
                scene["entity_ids"].append(entity_id)
    return {"entities": entities, "scenes": scenes, "provider": "openai" if llm.available() else "conservative_fallback"}


class SemanticDraft(BaseModel):
    title: str
    summary: str
    topics: list[str]
    mood: str
    llm_intensity: float = Field(ge=0, le=1)
    is_cliffhanger: bool


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def smooth_curve(values: list[float], window: int) -> list[float]:
    half = window // 2
    return [mean(values[max(0, i - half): i + half + 1]) for i in range(len(values))]


def s13_scene_semantics(ctx: StageContext) -> dict:
    duration = ctx.stage("s01_ingest")["duration"]
    scenes = ctx.stage("s12_vision_targeted")["scenes"]
    draft_titles = {scene["scene_id"]: scene.get("draft_title") for scene in ctx.stage("s10_scenes")["scenes"]}
    utterances = ctx.stage("s07_transcript_clean")["utterances"]
    audio = ctx.stage("s08_audio_events")
    t = ctx.t
    loudness_values = [value for _, value in audio["loudness"]]
    p10, p90 = _percentile(loudness_values, .1), _percentile(loudness_values, .9)
    llm = ctx.llm()
    system = load_prompt("s13_semantics_v1")
    output = []
    for index, scene in enumerate(scenes):
        scene_utterances = [u for u in utterances if u["utt_id"] in scene["utt_ids"]]
        length = max(.01, scene["end"] - scene["start"])
        speech_seconds = sum(max(0, u["end"] - u["start"]) for u in scene_utterances)
        overlap_seconds = sum(max(0, u["end"] - u["start"]) for u in scene_utterances if u.get("overlap"))
        dialogue_density = min(1.0, speech_seconds / length + .2 * (overlap_seconds / speech_seconds if speech_seconds else 0))
        raw_energy = curve_mean(audio["loudness"], scene["start"], scene["end"]) or 0.0
        energy = min(1.0, max(0.0, (raw_energy - p10) / (p90 - p10))) if p90 > p10 else 0.0
        events = [e for e in audio["events"] if e["start"] < scene["end"] and e["end"] > scene["start"]]
        music_ratio = curve_mean(audio.get("music", []), scene["start"], scene["end"]) or 0.0
        if llm.available():
            draft = llm.call(
                model=ctx.settings.llm_model_scenes, system=system, stage="s13_scene_semantics", schema=SemanticDraft,
                user=(
                    f"Time: {scene['start']:.1f}-{scene['end']:.1f}s"
                    + (f"\nWorking title: {draft_titles.get(scene['scene_id'])}" if draft_titles.get(scene["scene_id"]) else "")
                    + f"\nVisual: {json.dumps(scene['visual'], ensure_ascii=False)}"
                    + f"\nSounds: {', '.join(e['label'] for e in events) or 'none'}; music ratio {music_ratio:.2f}"
                    + "\nDialogue:\n" + ("\n".join(f"{u['speaker']}: {u['text']}" for u in scene_utterances) or "(none)")
                ),
            )
        else:
            draft = SemanticDraft(
                title=draft_titles.get(scene["scene_id"]) or f"Scene {index + 1}",
                summary=(scene_utterances[0]["text"][:180] if scene_utterances else "No dialogue in this scene."),
                topics=[], mood="neutral", llm_intensity=min(1.0, .18 + dialogue_density * .62), is_cliffhanger=False,
            )
        intensity = t.intensity_weight_llm * draft.llm_intensity + t.intensity_weight_audio * energy + t.intensity_weight_dialogue * dialogue_density
        output.append({
            **{k: v for k, v in scene.items() if k != "draft_title"},
            "audio_events": [e["event_id"] for e in events], "music_ratio": round(music_ratio, 4),
            "semantic": {
                "title": draft.title, "summary": draft.summary, "topics": draft.topics, "mood": draft.mood,
                "narrative_intensity": round(min(1.0, intensity), 4),
                "intensity_components": {"llm": round(draft.llm_intensity, 4), "audio_energy": round(energy, 4), "dialogue_density": round(dialogue_density, 4)},
                "is_cliffhanger": draft.is_cliffhanger,
            },
        })
    per_second = []
    for second in range(int(math.ceil(duration)) + 1):
        scene = next((s for s in output if s["start"] <= second < s["end"]), output[-1] if output else None)
        per_second.append(scene["semantic"]["narrative_intensity"] if scene else 0.0)
    smoothed = smooth_curve(per_second, t.intensity_smoothing_seconds) if per_second else []
    protected = sorted({s["scene_id"] for s in output if s["semantic"]["is_cliffhanger"]} | ({output[-1]["scene_id"]} if output else set()))
    return {
        "scenes": output, "intensity": [[float(i), round(v, 4)] for i, v in enumerate(smoothed)],
        "protected_scenes": protected, "provider": "openai_fused" if llm.available() else "fused_deterministic_baseline",
    }


# --------------------------------------------------------------------------- s14–s18: products


def s14_ad_scoring(ctx: StageContext) -> dict:
    duration = ctx.stage("s01_ingest")["duration"]
    semantics = ctx.stage("s13_scene_semantics")
    t = ctx.t
    candidates = generate_candidates(
        duration=duration, scenes=semantics["scenes"], silences=ctx.stage("s05_vad")["silences"],
        utterances=ctx.stage("s07_transcript_clean")["utterances"], entities=ctx.stage("s12_vision_targeted")["entities"],
        intensity=semantics["intensity"], protected=set(semantics["protected_scenes"]), t=t,
    )
    decided, summary = decide(
        candidates, load_catalogue(ctx.settings.brand_catalogue), duration=duration, min_gap=t.ad_default_min_gap,
        n_breaks=max(1, math.floor(duration / 600)), blocked=(t.ad_blocked_head_seconds, t.ad_blocked_tail_seconds),
        min_safety=t.ad_min_safety,
    )
    return {"candidates": decided, "decision": summary}


def s15_subtitles(ctx: StageContext) -> dict:
    cues = format_utterances(ctx.stage("s07_transcript_clean")["utterances"], ctx.t)
    outputs = ctx.root / "outputs"
    ctx.store.write_text(outputs / "episode_bn.srt", write_srt(cues))
    ctx.store.write_text(outputs / "episode_bn.vtt", write_vtt(cues))
    return {"cues": cues, "srt": "outputs/episode_bn.srt", "vtt": "outputs/episode_bn.vtt"}


def s16_captions(ctx: StageContext) -> dict:
    utterances = ctx.stage("s07_transcript_clean")["utterances"]
    events = finalize_events(ctx.stage("s08_audio_events")["events"], utterances, ctx.t)
    cues = build_cc(ctx.stage("s15_subtitles")["cues"], events, ctx.stage("s13_scene_semantics")["scenes"], ctx.t)
    outputs = ctx.root / "outputs"
    ctx.store.write_text(outputs / "episode_bn_cc.srt", write_srt(cues))
    ctx.store.write_text(outputs / "episode_bn_cc.vtt", write_vtt(cues))
    return {"cues": cues, "events": events, "srt": "outputs/episode_bn_cc.srt", "vtt": "outputs/episode_bn_cc.vtt"}


def s17_qc(ctx: StageContext) -> dict:
    transcript = ctx.stage("s07_transcript_clean")
    issues, summary = run_qc(
        ctx.stage("s15_subtitles")["cues"], transcript["utterances"], ctx.stage("s05_vad")["speech_segments"],
        ctx.stage("s08_audio_events")["events"], ctx.t, set(transcript.get("rejected", [])),
    )
    ctx.store.write_json(ctx.root / "outputs" / "qc_report.json", {"summary": summary, "issues": issues})
    return {"issues": issues, "summary": summary}


def _cached_stage_count(ctx: StageContext) -> int:
    path = ctx.root / "logs" / "pipeline.jsonl"
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line).get("event")
        except json.JSONDecodeError:
            continue
        if event == "pipeline_start":
            count = 0
        elif event == "stage_cached":
            count += 1
    return count


def s18_assemble(ctx: StageContext) -> dict:
    ingest = ctx.stage("s01_ingest")
    shots = ctx.stage("s03_keyframes")
    transcript = ctx.stage("s07_transcript_clean")
    audio = ctx.stage("s08_audio_events")
    semantics = ctx.stage("s13_scene_semantics")
    entities = ctx.stage("s12_vision_targeted")
    ads = ctx.stage("s14_ad_scoring")
    subs = ctx.stage("s15_subtitles")
    cc = ctx.stage("s16_captions")
    qc = ctx.stage("s17_qc")
    record = ctx.db.get_episode_record(ctx.episode_id)
    stage_records = ctx.db.get_stages(ctx.episode_id)
    llm_used = (ctx.root / "logs" / "llm_calls.jsonl").exists()
    models = ["saaras:v4 (Sarvam)", f"{shots['dedup_method']} keyframe dedup", ctx.stage("s02_shots")["detector"], ctx.stage("s05_vad")["method"], audio["event_detector"]]
    if llm_used:
        models += sorted({ctx.settings.llm_model_default, ctx.settings.llm_model_scenes})
    timeline = SemanticTimeline.model_validate({
        "schema_version": "1.0",
        "episode": {
            "id": ctx.episode_id, "title": record.title if record else ctx.episode_id, "duration": ingest["duration"],
            "fps": ingest["fps"], "resolution": f"{ingest['width']}×{ingest['height']}", "video_available": True,
        },
        "scenes": semantics["scenes"], "shots": shots["shots"], "utterances": transcript["utterances"],
        "audio_events": cc["events"], "entities": entities["entities"], "ad_candidates": ads["candidates"],
        "subtitles": {
            "sub_srt": subs["srt"], "sub_vtt": subs["vtt"], "cc_srt": cc["srt"], "cc_vtt": cc["vtt"],
            "cue_count": len(subs["cues"]), "cc_cue_count": len(cc["cues"]), "qc_summary": qc["summary"],
        },
        "subtitle_cues": subs["cues"], "cc_cues": cc["cues"], "qc": qc["issues"], "ad_decision": ads["decision"],
        "curves": {"intensity": semantics["intensity"], "loudness": audio["loudness"]},
        "processing": {
            "total_seconds": round(sum(s.elapsed or 0 for s in stage_records), 3),
            "stage_seconds": {s.stage_id: s.elapsed for s in stage_records},
            "llm_cost_usd": round(logged_cost(ctx.root / "logs" / "llm_calls.jsonl"), 4),
            "models": models, "cached_stages": _cached_stage_count(ctx),
            "thresholds": ctx.t.model_dump(),
        },
    })
    ctx.store.write_json(ctx.root / "outputs" / "semantic_timeline.json", timeline)
    return {"timeline": "outputs/semantic_timeline.json", "schema_version": "1.0"}
