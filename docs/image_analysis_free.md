Mapping this onto what s09\_vision\_baseline and s12\_vision\_targeted currently ask GPT-4.1-mini for — objects, location (closed vocabulary), visual mood, people count, visible brands, and "is entity X visible in this window" — each has a free, CPU-runnable HF counterpart. You already have the right pattern in local\_[models.py](http://models.py) (optional import, None fallback); these slot into the same shape.

Need (current OpenAI call): Presence check ("is a Pepsi can visible?") — s12

Free model: CLIP / SigLIP zero-shot — you already load SigLIP for embeddings in local\_models.py:73. Add a text tower call:

embed "a photo of a {entity}" vs a negative prompt, cosine-compare against the frame embedding.

Why it fits: This is the textbook zero-shot presence task CLIP-family models were built for. No extra model needed — just use

the text encoder half of the SigLIP model you already load. Runs in tens of ms/frame on CPU.

────────────────────────────────────────

Need (current OpenAI call): Objects list + people count — s09

Free model: YOLOv8n (Ultralytics, [yolov8n.pt](http://yolov8n.pt), \~6MB)

Why it fits: Tiny, real-time on CPU even per-keyframe. Gives you labeled boxes directly — person count is just

len(boxes[cls==person]). On HF: ultralytics/yolov8 or just pip install ultralytics.

────────────────────────────────────────

Need (current OpenAI call): Open-vocab objects/brands/products (things not in COCO's 80 classes) — s09/s12

Free model: YOLO-World (yolo-world/yolov8s-world, on HF)

Why it fits: Same YOLO speed, but you pass a text vocabulary at inference time instead of retraining — perfect for your

ad\_categories.yaml product list and brand names. Slightly heavier than YOLOv8n but still CPU-feasible since you only run it

on keyframes, not every frame.

────────────────────────────────────────

Need (current OpenAI call): Visible brand text / on-screen captions — s09

Free model: EasyOCR or PaddleOCR (lightweight mode)

Why it fits: Most "visible brand" hits in real footage are legible text (logos, signage, product labels), not shape

recognition. OCR + fuzzy string match against your brand catalogue is cheap and often more reliable than a vision model

guessing a brand name.

────────────────────────────────────────

Need (current OpenAI call): Location (your closed LOCATIONS vocabulary) — s09

Free model: CLIP/SigLIP zero-shot classification against that fixed label list, same mechanism as presence check

Why it fits: You already have a closed vocabulary — this is exactly what zero-shot CLIP classification is for, cheaper than

Places365 since you don't need a 365-way general classifier for a domain-specific list.

────────────────────────────────────────

Need (current OpenAI call): Visual mood — s09

Free model: CLIP/SigLIP zero-shot against a handful of mood prompts ("a tense scene", "a joyful scene", "a calm scene")

Why it fits: Coarse but free; same call as location, just a different prompt set. Don't expect fine-grained accuracy — treat it

as a soft signal like the LLM fallback already is.

Practical wiring: extend SiglipEmbedder in local\_[models.py](http://models.py) with an embed\_text(prompts) method (it already loads AutoModel/AutoProcessor for google/siglip-base-patch16-224, which has both towers) — that single addition covers presence, location, and mood. Add YOLOv8n as a second optional adapter (yolo\_detect(), same installed()/None pattern) for objects/people/boxes. OCR is a third, independent adapter for brand text. All three are CPU-only-friendly at your scale (sparse keyframes, not full video), matching how PANNs/Silero are wired for audio — gate them behind the gpu extra (or a new lighter extra, since none of these actually need a GPU) and have s09/s12 prefer local results over the OpenAI call the same way provider fields already report "conservative\_fallback" today.

What I didn't check: actual accuracy/latency numbers on your specific footage, and whether ultralytics/easyocr licensing (AGPL for Ultralytics) is acceptable for your deployment — worth a quick look before adding the dependency.