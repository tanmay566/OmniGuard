"""Perception configuration for OmniGuard."""

# ----------------------- Core detector / tracker -----------------------
MODEL_PATH = "yolo26n.pt"
TRACKER_CONFIG = "bytetrack.yaml"
DEVICE = "cpu"
IMGSZ = 512
DETECTION_CONFIDENCE = 0.35
IOU_THRESHOLD = 0.7
RELEVANT_CLASS_IDS = [0, 24, 26, 28]  # person, backpack, handbag, suitcase
CLASS_NAMES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}

# ----------------------- Additional safety perception -----------------------
POSE_MODEL_PATH = "yolo26n-pose.pt"
POSE_CONFIDENCE = 0.35
POSE_IMAGE_SIZE = 416
POSE_EVERY_N_PROCESSED_FRAMES = 8

# Public YOLO26 fire/smoke fine-tuned checkpoint. It is downloaded on first use.
FIRE_MODEL_REPO = "SalahALHaismawi/yolov26-fire-detection"
FIRE_MODEL_PATH = "fire_smoke_yolo26.pt"
FIRE_CONFIDENCE = 0.30
FIRE_IMAGE_SIZE = 416
FIRE_EVERY_N_PROCESSED_FRAMES = 8
FIRE_PERSISTENCE_SECONDS = 1.0
FIRE_ALERT_CLASSES = {"fire", "smoke"}

# Pose-derived event thresholds.
FALL_PERSISTENCE_SECONDS = 0.8
FALL_MIN_KEYPOINT_CONFIDENCE = 0.35
FALL_BODY_ANGLE_DEGREES = 35.0

# ----------------------- Processing -----------------------
FRAME_SKIP = 8
SHOW_VIDEO = True
SAVE_ANNOTATED_VIDEO = True
OUTPUT_VIDEO_PATH = "annotated_output.mp4"

# ----------------------- Event rules -----------------------
CROWD_THRESHOLD = 5
CROWD_PERSISTENCE_SECONDS = 2.0
BAGGAGE_DWELL_SECONDS = 30.0
INTRUSION_PERSISTENCE_SECONDS = 1.0
BAG_PERSON_NEARBY_PIXELS = 100.0
BAG_OWNER_MEMORY_SECONDS = 10.0
STATIONARY_DISTANCE_PIXELS = 8.0

# ----------------------- Zones / occupancy -----------------------
ZONE_CAPACITIES = {
    
}
RESTRICTED_ZONES = []
OCCUPANCY_SEND_INTERVAL_SECONDS = 5.0

# ----------------------- Backend -----------------------
BACKEND_URL = "http://localhost:8000"
INCIDENT_ENDPOINT = f"{BACKEND_URL}/incidents"
OCCUPANCY_ENDPOINT = f"{BACKEND_URL}/zones/occupancy"
HTTP_TIMEOUT_SECONDS = 2.0

# ----------------------- Runtime -----------------------
DEFAULT_VIDEO_NAME = "video.mp4"
