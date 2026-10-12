"""Optional local-model adapters. Each returns None when its package is not installed,
so stages can fall back to ffmpeg/heuristics and record which method produced the artifact."""
from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

from .media import run


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def read_wav_mono(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM audio.")
        rate, channels = handle.getframerate(), handle.getnchannels()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype("float32") / 32768
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def pyscenedetect_cuts(video: Path) -> list[float] | None:
    """PySceneDetect AdaptiveDetector (the documented CPU fallback for TransNetV2)."""
    if not installed("scenedetect"):
        return None
    from scenedetect import AdaptiveDetector, detect  # type: ignore[import-not-found]

    scenes = detect(str(video), AdaptiveDetector())
    return [start.get_seconds() for start, _ in scenes[1:]]


def silero_speech(audio: Path, min_silence_ms: int) -> list[dict] | None:
    if not installed("silero_vad"):
        return None
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio  # type: ignore[import-not-found]

    model = load_silero_vad()
    stamps = get_speech_timestamps(read_audio(str(audio), sampling_rate=16000), model, sampling_rate=16000, min_silence_duration_ms=min_silence_ms, return_seconds=True)
    return [{"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3)} for s in stamps]


def panns_windows(audio_16k: Path, work_dir: Path, ffmpeg: str, device: str, window: float = 1.0, hop: float = .5, batch: int = 64):
    """PANNs Cnn14 clip-wise tagging over sliding windows. Returns (labels, [(start, scores)]) or None."""
    if not installed("panns_inference"):
        return None
    import numpy as np
    from panns_inference import AudioTagging, labels  # type: ignore[import-not-found]

    resampled = work_dir / "full_32k.wav"
    if not resampled.exists():
        run([ffmpeg, "-y", "-i", str(audio_16k), "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le", str(resampled)])
    samples, rate = read_wav_mono(resampled)
    tagger = AudioTagging(checkpoint_path=None, device="cuda" if device.startswith("cuda") else "cpu")
    size, step = int(window * rate), int(hop * rate)
    starts = list(range(0, max(1, len(samples) - size + 1), step))
    results: list[tuple[float, list[float]]] = []
    for offset in range(0, len(starts), batch):
        chunk = starts[offset:offset + batch]
        frames = np.stack([np.pad(samples[s:s + size], (0, max(0, size - len(samples[s:s + size])))) for s in chunk])
        clipwise, _ = tagger.inference(frames)
        for start, scores in zip(chunk, clipwise):
            results.append((start / rate, scores.tolist()))
    return list(labels), results


class SiglipEmbedder:
    MODEL = "google/siglip-base-patch16-224"

    def __init__(self, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor  # type: ignore[import-not-found]

        self.torch = torch
        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.MODEL).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL)

    @staticmethod
    def _pooled(output):
        # transformers >=4.49 wraps get_image_features/get_text_features in BaseModelOutputWithPooling;
        # older versions return the pooled tensor directly.
        return output.pooler_output if hasattr(output, "pooler_output") else output

    def embed(self, paths: list[Path], batch: int = 32) -> list[list[float]]:
        from PIL import Image  # type: ignore[import-not-found]

        output: list[list[float]] = []
        for offset in range(0, len(paths), batch):
            images = [Image.open(path).convert("RGB") for path in paths[offset:offset + batch]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                features = self._pooled(self.model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            output.extend(features.cpu().tolist())
        return output

    def embed_text(self, prompts: list[str], batch: int = 32) -> list[list[float]]:
        """Text-tower embeddings, comparable to embed()'s image-tower output via dot product (both L2-normalized)."""
        output: list[list[float]] = []
        for offset in range(0, len(prompts), batch):
            inputs = self.processor(text=prompts[offset:offset + batch], padding="max_length", return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                features = self._pooled(self.model.get_text_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            output.extend(features.cpu().tolist())
        return output


def siglip_embedder(device: str) -> SiglipEmbedder | None:
    if not (installed("transformers") and installed("torch") and installed("PIL")):
        return None
    return SiglipEmbedder(device)


_yolo_model = None


def yolo_detect(paths: list[Path], device: str, conf: float = 0.35) -> list[list[tuple[str, float]]] | None:
    """YOLOv8n COCO detection. Returns, per image, [(label, confidence), ...] or None if ultralytics isn't installed."""
    if not installed("ultralytics"):
        return None
    from ultralytics import YOLO  # type: ignore[import-not-found]

    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    results = _yolo_model.predict(
        [str(path) for path in paths], device="cuda" if device.startswith("cuda") else "cpu", conf=conf, verbose=False,
    )
    return [[(result.names[int(box.cls)], float(box.conf)) for box in result.boxes] for result in results]


_ocr_reader = None


def ocr_texts(paths: list[Path]) -> list[list[str]] | None:
    """EasyOCR text recognition (English + Bengali). Returns, per image, the recognised strings, or None if easyocr isn't installed."""
    if not installed("easyocr"):
        return None
    import easyocr  # type: ignore[import-not-found]

    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en", "bn"], gpu=False)
    return [[str(text) for text in _ocr_reader.readtext(str(path), detail=0)] for path in paths]
