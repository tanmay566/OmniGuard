# OmniGuard Perception — YOLO26 + ByteTrack + Fire/Smoke + Pose

CPU-first perception module for OmniGuard.

## What it runs

- YOLO26n + ByteTrack for person/backpack/handbag/suitcase detection and tracking.
- YOLO26n-pose for pose keypoints and a lightweight fall-event heuristic.
- A separate YOLO26 fine-tuned fire/smoke detector, downloaded automatically from Hugging Face on first run.
- Existing zone, occupancy, baggage, crowd, and intrusion rules.
- Temporal event persistence/deduplication.
- Existing backend incident JSON contract is preserved.

## Run

```bash
cd ref_perception
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Put the input video beside `main.py` as `video.mp4`, or pass a path:

```bash
python main.py /path/to/your/video.mp4
```

The first run downloads the YOLO26 base detector, YOLO26 pose checkpoint, and the fire/smoke checkpoint into their normal model caches. Internet access is required for this first download.

## Output

`annotated_output.mp4` is written by default. The preview window can be disabled with `SHOW_VIDEO = False` in `config.py`.

## Important

The default fire/smoke checkpoint is a community YOLO26-S model hosted on Hugging Face. Its model card reports fire, smoke, and other classes and self-reported metrics; it also warns that performance depends on similarity to the training data. Validate it on your campus footage before treating alerts as reliable safety decisions.

The pose model is COCO-trained YOLO26n-pose and is used here as a heuristic input for a fall event; it is not a dedicated fall detector.

The zone polygons in `zones.py` remain example polygons for a 640x480 working frame and must be set for the camera view you are using.
