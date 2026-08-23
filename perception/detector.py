"""YOLO26 + ByteTrack adapter for OmniGuard."""

from __future__ import annotations

from typing import Any

from ultralytics import YOLO

from config import (
    CLASS_NAMES,
    DETECTION_CONFIDENCE,
    DEVICE,
    IMGSZ,
    IOU_THRESHOLD,
    MODEL_PATH,
    RELEVANT_CLASS_IDS,
    TRACKER_CONFIG,
)


class Detector:
    """Runs YOLO26 detection and persistent ByteTrack tracking."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        confidence: float = DETECTION_CONFIDENCE,
        tracker: str = TRACKER_CONFIG,
        device: str = DEVICE,
        imgsz: int = IMGSZ,
        iou: float = IOU_THRESHOLD,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = tracker
        self.device = device
        self.imgsz = imgsz
        self.iou = iou

    def track(self, frame) -> list[dict[str, Any]]:
        """Track one sequential video frame and return normalized detections.

        Ultralytics maintains ByteTrack state because persist=True is used on
        consecutive frames from the same video stream.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=RELEVANT_CLASS_IDS,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        # Track IDs can be absent briefly when the tracker has no assignment.
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(xyxy)

        detections: list[dict[str, Any]] = []
        for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, classes)):
            if cls_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_ids[i]) if i < len(track_ids) else -1

            detections.append(
                {
                    "class": CLASS_NAMES[cls_id],
                    "class_id": int(cls_id),
                    "confidence": round(float(conf), 3),
                    "box": (x1, y1, x2, y2),
                    "track_id": track_id,
                }
            )

        return detections
