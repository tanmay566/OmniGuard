"""Fire/smoke detector adapter for OmniGuard.

Uses a YOLO26 fine-tuned fire/smoke checkpoint downloaded from Hugging Face
on first use. The model is intentionally separate from the COCO detector so
normal object tracking remains stable and the backend contract stays unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import FIRE_CONFIDENCE, FIRE_IMAGE_SIZE, FIRE_MODEL_PATH, FIRE_MODEL_REPO


class FireDetector:
    def __init__(
        self,
        model_path: str = FIRE_MODEL_PATH,
        repo_id: str = FIRE_MODEL_REPO,
        confidence: float = FIRE_CONFIDENCE,
        imgsz: int = FIRE_IMAGE_SIZE,
        device: str = "cpu",
    ) -> None:
        resolved = self._resolve_model(model_path, repo_id)
        print(f"[fire] loading model: {resolved}")
        self.model = YOLO(resolved)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

        names = self.model.names
        if isinstance(names, dict):
            self.names = {int(k): str(v).lower() for k, v in names.items()}
        else:
            self.names = {i: str(v).lower() for i, v in enumerate(names)}

    @staticmethod
    def _resolve_model(model_path: str, repo_id: str) -> str:
        if os.path.isfile(model_path):
            return model_path

        from huggingface_hub import snapshot_download

        cache_dir = snapshot_download(repo_id=repo_id, allow_patterns=["*.pt"])
        pt_files = sorted(Path(cache_dir).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt fire model found in Hugging Face repo {repo_id}."
            )
        # Prefer the smallest checkpoint when multiple are present on CPU.
        pt_files.sort(key=lambda p: p.stat().st_size)
        return str(pt_files[0])

    def predict(self, frame) -> list[dict[str, Any]]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, classes):
            label = self.names.get(int(cls_id), str(cls_id))
            x1, y1, x2, y2 = map(int, box)
            detections.append(
                {
                    "class": label,
                    "class_id": int(cls_id),
                    "confidence": round(float(conf), 3),
                    "box": (x1, y1, x2, y2),
                }
            )
        return detections
