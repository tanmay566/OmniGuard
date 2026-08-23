# Perception Files Reference

Complete listing of every file in the perception module — code, purpose, and how they connect. Companion to `docs/BACKEND_FILES_REFERENCE.md`.

---

## File map

| File | Status | Purpose |
|---|---|---|
| `config.py` | Done | Thresholds, zone capacities, backend URLs |
| `zones.py` | Done | Zone polygon definitions + point-in-zone check |
| `detector.py` | Done, tested on real image | YOLO wrapper |
| `tracker.py` | Done | Centroid tracker — persistent IDs + dwell time |
| `rules.py` | Done | Crowd/baggage/intrusion/fire rule checks |
| `occupancy.py` | Done | Live per-zone person counting + trend detection |
| `incident_sender.py` | Done | Builds incident JSON, POSTs to backend |
| `main.py` | Done | The loop tying everything together |
| `requirements.txt` | Done | ultralytics, opencv-python-headless, numpy, requests |
| `test_detector.py` | Done (proof-of-concept only) | Not part of the real pipeline |

---

## Data flow through these files

```
video frame
  -> detector.detect(frame)                     [detector.py]
       -> raw detections (class, confidence, box)
  -> tracker.update(detections)                 [tracker.py]
       -> adds track_id, dwell_time_seconds
  -> occupancy_tracker.update(tracked_objects)   [occupancy.py]
       -> per-zone counts, sent to backend periodically
  -> rules.check_*(tracked_objects, zones)       [rules.py, zones.py]
       -> incident dict, or None
  -> incident_sender.send_incident(incident)     [incident_sender.py]
       -> POST to backend if a rule fired
```

All orchestrated by `main.py`, which stays deliberately thin — logic belongs in the
modules above, not in the loop.

---

## Complete source code


### `perception/config.py`

```python
"""
config.py
Perception module configuration - thresholds, zone definitions, backend connection.
Tune these numbers when testing against real videos.
"""

# Detection thresholds
CROWD_THRESHOLD = 5  # people in a zone = alert if count >= this
BAGGAGE_DWELL_SECONDS = 30  # unattended object stationary this long = alert
DETECTION_CONFIDENCE = 0.5  # YOLO confidence cutoff (0-1)
TRACKER_MAX_DISAPPEARED = 50  # frames before forgetting a tracked object

# Processing
FRAME_SKIP = 3  # process every Nth frame (1=every frame, 3=every 3rd for speed)
OCCUPANCY_SEND_INTERVAL = 5  # send occupancy update to backend every N frames

# Backend connection
BACKEND_URL = "http://localhost:8000"
INCIDENT_ENDPOINT = f"{BACKEND_URL}/incidents"
OCCUPANCY_ENDPOINT = f"{BACKEND_URL}/zones/occupancy"

# Zone capacities (for occupancy tracking)
# Maps zone name to max safe occupancy
ZONE_CAPACITIES = {
    "Gate_3": 10,           # main gate
    "Gate_1": 12,           # side gate
    "Gate_2": 10,           # side gate
    "Canteen": 50,          # large open area
    "Restricted_Lab": 5,    # small lab, max 5 people
    "Auditorium": 100,      # big hall
}

```

### `perception/zones.py`

```python
"""
zones.py
Zone polygon definitions - specific to one camera angle/resolution (640x480).
Redefine these coordinates when switching to a new video source.
"""
import cv2
import numpy as np

# Define zones as polygons: list of (x,y) points
# These are for a 640x480 video frame (standard YOLO input size)
ZONES = {
    "Gate_3": [(100, 200), (300, 200), (300, 400), (100, 400)],
    "Restricted_Lab": [(400, 50), (600, 50), (600, 200), (400, 200)],
}


def point_in_zone(x, y, zone_name):
    """Check if point (x,y) is inside the named zone polygon."""
    if zone_name not in ZONES:
        return False
    points = np.array(ZONES[zone_name], np.int32)
    result = cv2.pointPolygonTest(points, (x, y), False)
    return result >= 0  # 0 or positive = inside or on the boundary

```

### `perception/detector.py`

```python
"""
detector.py
Wraps YOLO model loading + inference. Keeps the perception module's other files
(tracker.py, rules.py) independent of the raw ultralytics API details.
"""
from ultralytics import YOLO

# Classes we care about from the COCO pretrained model.
# (person=0, backpack=24, suitcase=28, handbag=26 - useful for baggage rule later)
RELEVANT_CLASSES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}


class Detector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.4):
        """
        model_path: 'yolov8n.pt' is the smallest/fastest pretrained YOLOv8 model -
                    good for real-time-ish demo speed. Swap to yolov8s/m for more
                    accuracy if your machine can handle the slower speed.
        confidence: minimum confidence score to keep a detection.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Runs YOLO on a single frame (a numpy array from OpenCV).
        Returns a list of dicts: [{"class": "person", "confidence": 0.91,
                                    "box": (x1, y1, x2, y2)}, ...]
        Only returns classes we care about (RELEVANT_CLASSES), filtered by confidence.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in RELEVANT_CLASSES or conf < self.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class": RELEVANT_CLASSES[cls_id],
                "confidence": round(conf, 2),
                "box": (x1, y1, x2, y2),
            })

        return detections

```

### `perception/tracker.py`

```python
"""
tracker.py
Assigns persistent IDs to detected objects across frames by matching centroids
(centers of bounding boxes) between consecutive frames. This is what makes
dwell-time detection possible - without it, every frame's object looks new.
"""
import numpy as np
from collections import defaultdict


class CentroidTracker:
    """
    Matches new detections to previously tracked objects by nearest center-point
    distance, giving each object a persistent track_id and accumulated dwell time.
    """

    def __init__(self, max_disappeared=50):
        """
        max_disappeared: if an object's centroid isn't matched for this many
                          frames in a row, forget it (it left the frame).
        """
        self.next_id = 0
        self.objects = {}  # {track_id: {"centroid": (x,y)}}
        self.disappeared = defaultdict(int)  # frames since last match
        self.max_disappeared = max_disappeared

        self.dwell_timers = defaultdict(float)  # {track_id: seconds_stationary}
        self.prev_positions = {}  # {track_id: prev_centroid}

    def update(self, detections):
        """
        Input: list of dicts from Detector.detect()
               [{"class": "person", "confidence": 0.89, "box": (x1,y1,x2,y2)}, ...]

        Output: same detections, each now with "track_id" and "dwell_time_seconds"
                added.
        """
        if len(detections) == 0:
            return []

        new_centers = [self._get_center(d["box"]) for d in detections]
        matched, unmatched_new = self._match_to_existing(new_centers)

        # Update matched objects' positions and dwell times
        for new_idx, track_id in matched.items():
            self.disappeared[track_id] = 0
            self.objects[track_id] = {"centroid": new_centers[new_idx]}

            if self._is_stationary(track_id, new_centers[new_idx]):
                self.dwell_timers[track_id] += 1 / 30.0  # assumes ~30fps
            else:
                self.dwell_timers[track_id] = 0

            detections[new_idx]["track_id"] = track_id
            detections[new_idx]["dwell_time_seconds"] = round(self.dwell_timers[track_id], 1)

        # Create new tracks for unmatched detections
        for new_idx in unmatched_new:
            self.next_id += 1
            self.objects[self.next_id] = {"centroid": new_centers[new_idx]}
            self.disappeared[self.next_id] = 0
            self.dwell_timers[self.next_id] = 0

            detections[new_idx]["track_id"] = self.next_id
            detections[new_idx]["dwell_time_seconds"] = 0

        # Age out tracks that weren't matched this frame
        matched_ids = set(matched.values())
        for track_id in list(self.disappeared.keys()):
            if track_id not in matched_ids:
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.objects.pop(track_id, None)
                    self.disappeared.pop(track_id, None)
                    self.dwell_timers.pop(track_id, None)
                    self.prev_positions.pop(track_id, None)

        return detections

    def _get_center(self, box):
        """Compute center (x,y) of a bounding box (x1,y1,x2,y2)."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _match_to_existing(self, new_centers):
        """
        Match new detection centers to existing tracked objects by nearest distance.
        Returns: (matched_dict, unmatched_new_indices)
                 matched_dict = {new_idx: track_id, ...}
        """
        if len(self.objects) == 0:
            return {}, list(range(len(new_centers)))

        existing_ids = list(self.objects.keys())
        existing_centers = [self.objects[tid]["centroid"] for tid in existing_ids]

        matched = {}
        unmatched_new = list(range(len(new_centers)))

        for new_idx, new_center in enumerate(new_centers):
            distances = [
                np.sqrt((new_center[0] - ec[0]) ** 2 + (new_center[1] - ec[1]) ** 2)
                for ec in existing_centers
            ]
            closest_idx = int(np.argmin(distances))
            closest_dist = distances[closest_idx]

            # threshold = max pixel distance to consider a match
            if closest_dist < 50:
                track_id = existing_ids[closest_idx]
                matched[new_idx] = track_id
                if new_idx in unmatched_new:
                    unmatched_new.remove(new_idx)

        return matched, unmatched_new

    def _is_stationary(self, track_id, new_center, threshold=5):
        """Check if object has barely moved since last frame (in pixels)."""
        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = new_center
            return True

        prev = self.prev_positions[track_id]
        dist = np.sqrt((new_center[0] - prev[0]) ** 2 + (new_center[1] - prev[1]) ** 2)
        self.prev_positions[track_id] = new_center

        return dist < threshold

```

### `perception/rules.py`

```python
"""
rules.py
Deterministic rule checks run against tracked objects each frame.
One function per rule - isolated because thresholds get tuned constantly.
Each function returns an incident dict, or None if the rule didn't fire.
"""
import numpy as np
from zones import point_in_zone


def check_crowd(tracked_objects, zones_config, threshold):
    """
    Count persons per zone, return incident if any zone's count >= threshold.
    """
    person_counts = {zone: 0 for zone in zones_config}

    for obj in tracked_objects:
        if obj["class"] != "person":
            continue

        x1, y1, x2, y2 = obj["box"]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        for zone_name in zones_config:
            if point_in_zone(center_x, center_y, zone_name):
                person_counts[zone_name] += 1

    for zone_name, count in person_counts.items():
        if count >= threshold:
            return {
                "type": "overcrowding",
                "zone": zone_name,
                "count": count,
            }

    return None


def check_baggage(tracked_objects, dwell_threshold_sec):
    """
    Find unattended bags: stationary for dwell_threshold_sec+ with no person nearby.
    """
    bags = [o for o in tracked_objects if o["class"] in ["backpack", "handbag", "suitcase"]]
    persons = [o for o in tracked_objects if o["class"] == "person"]

    for bag in bags:
        if bag["dwell_time_seconds"] < dwell_threshold_sec:
            continue

        bag_x1, bag_y1, bag_x2, bag_y2 = bag["box"]
        bag_cx, bag_cy = (bag_x1 + bag_x2) // 2, (bag_y1 + bag_y2) // 2

        owner_nearby = False
        for person in persons:
            px1, py1, px2, py2 = person["box"]
            person_cx, person_cy = (px1 + px2) // 2, (py1 + py2) // 2
            distance = np.sqrt((bag_cx - person_cx) ** 2 + (bag_cy - person_cy) ** 2)

            if distance < 100:  # within 100 pixels = nearby
                owner_nearby = True
                break

        if not owner_nearby:
            return {
                "type": "unattended_baggage",
                "tracked_object_id": bag["track_id"],
                "dwell_time_seconds": bag["dwell_time_seconds"],
                "detection_confidence": bag["confidence"],
            }

    return None


def check_intrusion(tracked_objects, zones_config, restricted_zones=None):
    """
    Alert if a person enters a restricted zone.
    """
    if restricted_zones is None:
        restricted_zones = ["Restricted_Lab"]

    persons = [o for o in tracked_objects if o["class"] == "person"]

    for person in persons:
        x1, y1, x2, y2 = person["box"]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        for zone_name in restricted_zones:
            if point_in_zone(center_x, center_y, zone_name):
                return {
                    "type": "intrusion",
                    "zone": zone_name,
                    "tracked_object_id": person["track_id"],
                    "detection_confidence": person["confidence"],
                }

    return None


def check_fire(fire_confidence, threshold=0.6):
    """
    Placeholder wrapper for a fire/smoke classifier (stretch goal).
    Wire in an actual model's output here when built; returns None until then.
    """
    if fire_confidence >= threshold:
        return {
            "type": "fire",
            "detection_confidence": fire_confidence,
        }
    return None

```

### `perception/occupancy.py`

```python
"""
occupancy.py
Tracks current occupancy per zone and sends updates to the backend.
This runs alongside the incident detection rules.
"""
import requests
from collections import defaultdict
from datetime import datetime
from config import ZONE_CAPACITIES, OCCUPANCY_ENDPOINT
from zones import point_in_zone


class OccupancyTracker:
    """
    Tracks person count per zone and detects trends.
    Sends occupancy updates to backend for live dashboard updates.
    """
    
    def __init__(self, zone_capacities=ZONE_CAPACITIES):
        self.zone_capacities = zone_capacities
        self.current_counts = defaultdict(int)  # {zone_name: count}
        self.history = defaultdict(list)  # {zone_name: [count1, count2, ...]} - last N frames
        self.history_size = 30  # keep last 30 frames of history for trend detection
    
    def update(self, tracked_objects):
        """
        Update occupancy counts based on current tracked objects.
        
        tracked_objects: list of dicts with "class", "box", etc from tracker
        Returns: dict mapping zone names to their current occupancy dicts (for sending to backend)
        """
        # Reset counts
        self.current_counts = defaultdict(int)
        
        # Count persons per zone
        persons = [o for o in tracked_objects if o["class"] == "person"]
        
        for person in persons:
            x1, y1, x2, y2 = person["box"]
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Check which zones this person is in
            for zone_name in self.zone_capacities.keys():
                if point_in_zone(center_x, center_y, zone_name):
                    self.current_counts[zone_name] += 1
        
        # Update history and detect trends
        occupancies = {}
        for zone_name, capacity in self.zone_capacities.items():
            count = self.current_counts[zone_name]
            
            # Keep history
            self.history[zone_name].append(count)
            if len(self.history[zone_name]) > self.history_size:
                self.history[zone_name].pop(0)
            
            # Calculate trend
            trend = self._detect_trend(zone_name)
            
            # Calculate percentage
            occupancy_percentage = (count / capacity * 100) if capacity > 0 else 0
            
            occupancies[zone_name] = {
                "zone": zone_name,
                "current_count": count,
                "capacity": capacity,
                "occupancy_percentage": round(occupancy_percentage, 1),
                "trend": trend,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        return occupancies
    
    def _detect_trend(self, zone_name):
        """
        Detect if occupancy is increasing, stable, or decreasing.
        Uses simple comparison of recent vs older data.
        """
        history = self.history[zone_name]
        
        if len(history) < 10:
            return "stable"  # not enough data
        
        recent = sum(history[-5:]) / 5  # average of last 5 frames
        older = sum(history[-10:-5]) / 5  # average of 5 frames before that
        
        if recent > older + 0.5:  # more than 0.5 people difference
            return "increasing"
        elif recent < older - 0.5:
            return "decreasing"
        else:
            return "stable"
    
    def get_occupancy_for_zone(self, zone_name):
        """Get current occupancy for a specific zone."""
        if zone_name not in self.zone_capacities:
            return None
        
        count = self.current_counts[zone_name]
        capacity = self.zone_capacities[zone_name]
        occupancy_percentage = (count / capacity * 100) if capacity > 0 else 0
        trend = self._detect_trend(zone_name)
        
        return {
            "zone": zone_name,
            "current_count": count,
            "capacity": capacity,
            "occupancy_percentage": round(occupancy_percentage, 1),
            "trend": trend,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


def send_occupancy_update(occupancy_data, endpoint=OCCUPANCY_ENDPOINT):
    """
    Send an occupancy update to the backend.
    
    occupancy_data: dict with zone, current_count, capacity, occupancy_percentage, trend, timestamp
    endpoint: backend URL for occupancy endpoint
    
    Never crash the main loop if this fails - just log and continue.
    """
    try:
        response = requests.post(
            endpoint,
            json=occupancy_data,
            timeout=2  # don't wait longer than 2 sec
        )
        if response.status_code == 200:
            pass  # silent success
        else:
            print(f"⚠ Backend returned {response.status_code} for occupancy update")
    except requests.exceptions.RequestException as e:
        print(f"⚠ Failed to send occupancy update: {e}")
        # Important: don't re-raise, keep the perception loop running

```

### `perception/incident_sender.py`

```python
"""
incident_sender.py
Builds a structured incident matching the schema the backend expects (IncidentIn),
and sends it via HTTP POST. Never crashes the main perception loop on failure.
"""
import uuid
from datetime import datetime
import requests

from config import INCIDENT_ENDPOINT


def build_incident(incident_type, zone=None, tracked_object_id=None,
                    dwell_time_seconds=None, count=None, detection_confidence=None):
    """
    Build a structured incident matching backend/schemas.py IncidentIn exactly.
    This is the handoff contract with the backend engineer - don't change field
    names without checking with them first.
    """
    return {
        "incident_id": f"inc_{uuid.uuid4().hex[:8]}",
        "type": incident_type,
        "zone": zone,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tracked_object_id": tracked_object_id,
        "dwell_time_seconds": dwell_time_seconds,
        "detection_confidence": detection_confidence,
        "count": count,
    }


def send_incident(incident, endpoint=INCIDENT_ENDPOINT):
    """
    POST the incident to the backend's /incidents endpoint.
    Never crash the main loop on a failed send - just log and continue.
    """
    try:
        response = requests.post(endpoint, json=incident, timeout=2)
        if response.status_code == 200:
            print(f"Sent {incident['type']} incident: {incident['incident_id']}")
        else:
            print(f"Backend returned {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send incident: {e}")

```

### `perception/main.py`

```python
"""
main.py
Main perception loop.
Reads video → detects → tracks → runs rules → sends incidents
Also tracks and sends occupancy updates regularly.
"""
import cv2
import numpy as np
from datetime import datetime

from detector import Detector
from tracker import CentroidTracker
from rules import check_crowd, check_baggage, check_intrusion, check_fire
from incident_sender import build_incident, send_incident
from occupancy import OccupancyTracker, send_occupancy_update

from config import (
    CROWD_THRESHOLD, BAGGAGE_DWELL_SECONDS, DETECTION_CONFIDENCE,
    FRAME_SKIP, OCCUPANCY_SEND_INTERVAL, ZONE_CAPACITIES
)
from zones import ZONES


def run(video_path=0):
    """
    Main perception loop.
    
    video_path: 0 for webcam, or path to a video file
    """
    detector = Detector(confidence=DETECTION_CONFIDENCE)
    tracker = CentroidTracker()
    occupancy_tracker = OccupancyTracker(zone_capacities=ZONE_CAPACITIES)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    occupancy_send_counter = 0
    
    print("=" * 60)
    print("PERCEPTION MODULE STARTED")
    print("=" * 60)
    print(f"Video source: {video_path}")
    print(f"Crowd threshold: {CROWD_THRESHOLD} people")
    print(f"Baggage dwell time: {BAGGAGE_DWELL_SECONDS}s")
    print(f"Occupancy send interval: every {OCCUPANCY_SEND_INTERVAL} frames")
    print(f"Zone capacities: {ZONE_CAPACITIES}")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n[!] End of video or error reading frame")
            break
        
        frame_count += 1
        occupancy_send_counter += 1
        
        # Frame skipping for speed
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Resize to standard YOLO input size
        frame = cv2.resize(frame, (640, 480))
        
        # ========== DETECTION STEP ==========
        raw_detections = detector.detect(frame)
        
        # ========== TRACKING STEP ==========
        tracked_objects = tracker.update(raw_detections)
        
        # ========== OCCUPANCY TRACKING STEP ==========
        occupancies = occupancy_tracker.update(tracked_objects)
        
        # Send occupancy updates at regular intervals (not every frame)
        if occupancy_send_counter >= OCCUPANCY_SEND_INTERVAL:
            for zone_name, occupancy_data in occupancies.items():
                send_occupancy_update(occupancy_data)
            occupancy_send_counter = 0
        
        # ========== INCIDENT RULES STEP ==========
        incident = None
        
        # Try crowd check first
        crowd_incident = check_crowd(tracked_objects, ZONES, CROWD_THRESHOLD)
        if crowd_incident:
            incident = build_incident(**crowd_incident)
        
        # Try baggage check
        if not incident:
            baggage_incident = check_baggage(tracked_objects, BAGGAGE_DWELL_SECONDS)
            if baggage_incident:
                incident = build_incident(**baggage_incident)
        
        # Try intrusion check
        if not incident:
            intrusion_incident = check_intrusion(tracked_objects, ZONES)
            if intrusion_incident:
                incident = build_incident(**intrusion_incident)
        
        # Try fire check
        if not incident:
            fire_incident = check_fire(0.0)  # placeholder, no actual fire model yet
            if fire_incident:
                incident = build_incident(**fire_incident)
        
        # ========== SEND INCIDENT (if any) ==========
        if incident:
            send_incident(incident)
        
        # ========== PERIODIC LOGGING ==========
        if frame_count % (FRAME_SKIP * 30) == 0:
            print(f"[Frame {frame_count:5d}] Tracked: {len(tracked_objects)} objects | " +
                  f"Occupancy: {dict(occupancy_tracker.current_counts)}")
    
    cap.release()
    print("\n[!] Perception loop stopped")


if __name__ == "__main__":
    # Run on webcam: run(0)
    # Run on video file: run("path/to/video.mp4")
    run(0)

```

### `perception/requirements.txt`

```
ultralytics
opencv-python-headless
numpy
requests
```

---

## The one contract that matters

`incident_sender.build_incident()` output must exactly match `IncidentIn` in the
backend's `schemas.py` — same field names, same types. This is the single most likely
integration bug between perception and backend, so it's worth re-checking any time
either file changes.

**Perception sends (8 fields):**
```json
{
  "incident_id": "inc_a1b2c3d4",
  "type": "unattended_baggage",
  "zone": "Gate_3",
  "timestamp": "2026-08-19T14:32:10Z",
  "tracked_object_id": 47,
  "dwell_time_seconds": 32.5,
  "detection_confidence": 0.89,
  "count": null
}
```

See `docs/PROPERTIES_REFERENCE.md` for the full field-by-field breakdown across all
incident types, and `docs/DATA_FLOW_COMPLETE.md` for how this gets enriched once it
reaches the backend and agent.

---

## Running it

```bash
cd perception
pip install -r requirements.txt --break-system-packages

# webcam:
python3 main.py

# or edit the bottom of main.py to point at a sample video file:
#   run("path/to/sample_video.mp4")
```

Requires the backend running at the URL set in `config.py` (`BACKEND_URL`,
default `http://localhost:8000`) — incidents/occupancy updates will fail silently
(logged, not crashed) if the backend isn't reachable.

---

## Before running on a new video

1. **Redefine zones** in `zones.py` — polygon coordinates are specific to one camera
   angle/resolution.
2. **Re-tune thresholds** in `config.py` — `CROWD_THRESHOLD`, `BAGGAGE_DWELL_SECONDS`,
   `DETECTION_CONFIDENCE` were set as reasonable defaults, not measured against your
   actual footage.
3. **Test `detector.py` alone first** (see `test_detector.py`) to confirm detection
   quality on the new footage before wiring in tracking and rules.

See `docs/PERCEPTION_GUIDE.md` for the narrative walkthrough of each file, and
`docs/SETUP.md` for full local run instructions across all three modules.
