# OmniGuard Perception — YOLO26 + ByteTrack

## 1. Install dependencies

From this directory:

```bash
pip install -r requirements.txt
```

## 2. Put your video here

Copy your video into this folder and name it:

```text
video.mp4
```

You can also use another filename and pass its path explicitly.

## 3. Start perception

With `video.mp4`:

```bash
python main.py
```

With another video:

```bash
python main.py /path/to/your/video.mp4
```

## 4. What it does

- Loads `yolo26n.pt` automatically.
- Uses Ultralytics ByteTrack.
- Runs on CPU by default.
- Resizes frames to 640x480 for the current zone coordinates.
- Detects person, backpack, handbag, and suitcase classes.
- Tracks persistent IDs.
- Calculates temporal dwell/stationary state using the video's timeline.
- Checks zones, overcrowding, restricted-zone presence, and unattended baggage.
- Deduplicates persistent events before sending them.
- Sends the existing incident JSON contract to `POST /incidents`.
- Sends occupancy data to `POST /zones/occupancy`.
- Keeps running when the backend is unavailable.
- Shows an annotated preview.
- Writes `annotated_output.mp4` when enabled.

## 5. Backend

By default:

```text
http://localhost:8000
```

Change `BACKEND_URL` in `config.py` if your backend is elsewhere.

## 6. Zones

The current zone polygons are for a 640x480 working frame and are still the original example zones. Update `zones.py` after you inspect the uploaded video.

## 7. CPU mode

The default configuration is CPU-only and uses `yolo26n.pt`.

If processing is too slow, change:

```text
FRAME_SKIP = 2
```

in `config.py`.

Do not change the zone coordinates unless you know the video geometry is different.
