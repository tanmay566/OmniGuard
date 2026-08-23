"""YOLO26 pose inference for lightweight behavior cues."""

from __future__ import annotations

from typing import Any

from ultralytics import YOLO

from config import POSE_CONFIDENCE, POSE_IMAGE_SIZE, POSE_MODEL_PATH


class PoseDetector:
    def __init__(
        self,
        model_path: str = POSE_MODEL_PATH,
        confidence: float = POSE_CONFIDENCE,
        imgsz: int = POSE_IMAGE_SIZE,
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def predict(self, frame) -> list[dict[str, Any]]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return []

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else []
        confs = boxes.conf.cpu().numpy() if boxes is not None else []
        keypoints = result.keypoints.xy.cpu().numpy()
        kp_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None

        detections: list[dict[str, Any]] = []
        for i in range(len(keypoints)):
            box = tuple(map(int, xyxy[i])) if len(xyxy) > i else (0, 0, 0, 0)
            points = keypoints[i]
            confidences = kp_conf[i] if kp_conf is not None else None
            detections.append(
                {
                    "box": box,
                    "confidence": round(float(confs[i]), 3) if len(confs) > i else 0.0,
                    "keypoints": [
                        {
                            "x": float(points[j][0]),
                            "y": float(points[j][1]),
                            "confidence": float(confidences[j]) if confidences is not None else 1.0,
                        }
                        for j in range(len(points))
                    ],
                }
            )
        return detections
