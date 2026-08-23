"""OmniGuard perception entry point.

Run:
    python main.py path/to/video.mp4

Or put a file named video.mp4 in this directory and simply run:
    python main.py

Pipeline:
    YOLO26n + ByteTrack -> objects/tracks
    YOLO26n-pose       -> pose/fall cues
    YOLO26 fire model  -> fire/smoke cues
    zones/rules        -> incidents
    event state        -> persistence/deduplication
    incident sender    -> existing FastAPI JSON contract
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import cv2
import numpy as np

from config import (
    BAGGAGE_DWELL_SECONDS,
    CROWD_PERSISTENCE_SECONDS,
    CROWD_THRESHOLD,
    DEFAULT_VIDEO_NAME,
    FIRE_EVERY_N_PROCESSED_FRAMES,
    FIRE_PERSISTENCE_SECONDS,
    FALL_PERSISTENCE_SECONDS,
    FRAME_SKIP,
    IMGSZ,
    INTRUSION_PERSISTENCE_SECONDS,
    MODEL_PATH,
    OCCUPANCY_SEND_INTERVAL_SECONDS,
    OUTPUT_VIDEO_PATH,
    POSE_EVERY_N_PROCESSED_FRAMES,
    RESTRICTED_ZONES,
    SAVE_ANNOTATED_VIDEO,
    SHOW_VIDEO,
    TRACKER_CONFIG,
    ZONE_CAPACITIES,
)
from detector import Detector
from event_state import EventStateManager
from fire_detector import FireDetector
from incident_sender import build_incident, send_incident
from occupancy import OccupancyTracker, send_occupancy_update
from pose_detector import PoseDetector
from rules import check_baggage, check_crowd, check_fire, check_fall, check_intrusion
from tracker import TrackingState
from zones import ZONES


def _resolve_video_path(video_path: str | None) -> str:
    if video_path:
        return video_path
    for candidate in (DEFAULT_VIDEO_NAME, "input.mp4", "input.mov", "video.avi"):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "No video supplied. Put your video beside main.py as video.mp4 "
        "or run: python main.py /path/to/video.mp4"
    )


def _resize_for_perception(frame: np.ndarray) -> np.ndarray:
    # Keep the existing 4:3 coordinate convention so the current zone polygons remain usable.
    height = int(IMGSZ * 480 / 640)
    return cv2.resize(frame, (IMGSZ, height), interpolation=cv2.INTER_AREA)


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _associate_pose_with_tracks(pose_detections, tracked_objects):
    persons = [o for o in tracked_objects if o.get("class") == "person" and int(o.get("track_id", -1)) >= 0]
    for pose in pose_detections:
        best_id = None
        best_iou = 0.0
        for person in persons:
            score = _iou(pose["box"], person["box"])
            if score > best_iou:
                best_iou = score
                best_id = int(person["track_id"])
        if best_iou >= 0.25:
            pose["track_id"] = best_id
    return pose_detections


def _draw_pose(frame, pose_detections):
    # COCO-17 skeleton pairs.
    pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    for pose in pose_detections:
        kps = pose.get("keypoints", [])
        for kp in kps:
            if kp.get("confidence", 1.0) >= 0.35:
                cv2.circle(frame, (int(kp["x"]), int(kp["y"])), 2, (255, 0, 255), -1)
        for a, b in pairs:
            if a >= len(kps) or b >= len(kps):
                continue
            if kps[a].get("confidence", 1.0) < 0.35 or kps[b].get("confidence", 1.0) < 0.35:
                continue
            cv2.line(
                frame,
                (int(kps[a]["x"]), int(kps[a]["y"])),
                (int(kps[b]["x"]), int(kps[b]["y"])),
                (255, 0, 255),
                1,
            )
    return frame


def _draw_fire(frame, fire_detections):
    for det in fire_detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _draw_overlay(frame, tracked_objects, occupancies, pose_detections, fire_detections):
    for zone_name, points in ZONES.items():
        polygon = np.asarray(points, dtype=np.int32)
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)
        x, y = points[0]
        cv2.putText(
            frame,
            zone_name,
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["box"]
        color = (0, 255, 0) if obj["class"] == "person" else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj['class']} #{obj['track_id']} {obj['confidence']:.2f}"
        if obj.get("dwell_time_seconds", 0) > 0:
            label += f" d={obj['dwell_time_seconds']:.1f}s"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    _draw_pose(frame, pose_detections)
    _draw_fire(frame, fire_detections)

    y = 22
    for zone_name, data in occupancies.items():
        text = f"{zone_name}: {data['current_count']}/{data['capacity']} ({data['trend']})"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    return frame


def run(video_path: str | None = None) -> None:
    source = _resolve_video_path(video_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 1 or source_fps > 240:
        source_fps = 30.0

    print("Loading YOLO26 + ByteTrack...")
    detector = Detector()
    print("Loading YOLO26 pose model...")
    pose_detector = PoseDetector()
    print("Loading YOLO26 fire/smoke model...")
    fire_detector = FireDetector()

    tracking_state = TrackingState()
    occupancy_tracker = OccupancyTracker(zone_capacities=ZONE_CAPACITIES)
    event_state = EventStateManager()

    writer = None
    frame_number = 0
    processed_number = 0
    processing_started = time.monotonic()
    latest_pose: list[dict[str, Any]] = []
    latest_fire: list[dict[str, Any]] = []

    print("=" * 76)
    print("OMNIGUARD PERCEPTION — YOLO26 + BYTETRACK + FIRE/SMOKE + POSE")
    print("=" * 76)
    print(f"Video: {source}")
    print(f"Main model: {MODEL_PATH}")
    print(f"Tracker: {TRACKER_CONFIG}")
    print(f"Device: CPU")
    print(f"Source: {original_width}x{original_height} @ {source_fps:.2f} FPS")
    print(f"Frame skip: {FRAME_SKIP}")
    print(f"Pose every: {POSE_EVERY_N_PROCESSED_FRAMES} processed frame(s)")
    print(f"Fire every: {FIRE_EVERY_N_PROCESSED_FRAMES} processed frame(s)")
    print(f"Restricted zones: {RESTRICTED_ZONES}")
    print("=" * 76)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1

            if FRAME_SKIP > 1 and (frame_number - 1) % FRAME_SKIP != 0:
                continue

            processed_number += 1
            perception_frame = _resize_for_perception(frame)
            video_time = (frame_number - 1) / source_fps

            tracked_objects = detector.track(perception_frame)
            tracked_objects = tracking_state.update(tracked_objects, timestamp=video_time)

            if processed_number % POSE_EVERY_N_PROCESSED_FRAMES == 0:
                latest_pose = pose_detector.predict(perception_frame)
                latest_pose = _associate_pose_with_tracks(latest_pose, tracked_objects)

            if processed_number % FIRE_EVERY_N_PROCESSED_FRAMES == 0:
                latest_fire = fire_detector.predict(perception_frame)

            occupancies = occupancy_tracker.update(tracked_objects)
            if occupancy_tracker.should_send(interval_seconds=OCCUPANCY_SEND_INTERVAL_SECONDS):
                for occupancy_data in occupancies.values():
                    send_occupancy_update(occupancy_data)

            current_incidents: list[dict[str, Any]] = []
            current_incidents.extend(check_crowd(tracked_objects, ZONES, CROWD_THRESHOLD))
            current_incidents.extend(check_baggage(tracked_objects, BAGGAGE_DWELL_SECONDS, tracking_state, video_time))
            current_incidents.extend(check_intrusion(tracked_objects, RESTRICTED_ZONES))
            current_incidents.extend(check_fire(latest_fire))

            for fall in check_fall(latest_pose):
                # Preserve the actual ByteTrack ID when pose/body boxes can be associated.
                current_incidents.append(fall)
                if fall.get("tracked_object_id") is not None:
                    pass

            persistence_by_type = {
                "overcrowding": CROWD_PERSISTENCE_SECONDS,
                "intrusion": INTRUSION_PERSISTENCE_SECONDS,
                "unattended_baggage": 0.0,
                "fire": FIRE_PERSISTENCE_SECONDS,
                "smoke": FIRE_PERSISTENCE_SECONDS,
                "fall": FALL_PERSISTENCE_SECONDS,
            }

            for candidate in current_incidents:
                event_type = candidate["type"]
                if event_state.update(candidate, persistence_by_type.get(event_type, 0.0), now=video_time):
                    # Perception rules use the key `type`, while build_incident()
                    # accepts `incident_type` before constructing the backend JSON.
                    incident_args = dict(candidate)
                    incident_args["incident_type"] = incident_args.pop("type")
                    send_incident(build_incident(**incident_args))

            event_state.clear_missing(current_incidents, now=video_time)

            annotated = _draw_overlay(
                perception_frame.copy(), tracked_objects, occupancies, latest_pose, latest_fire
            )
            fire_active = any(str(d.get("class", "")).lower() in {"fire", "smoke"} for d in latest_fire)
            if fire_active:
                cv2.putText(
                    annotated,
                    "FIRE / SMOKE DETECTED — VERIFY",
                    (10, annotated.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                annotated,
                f"YOLO26 + ByteTrack | frame {frame_number} | processed {processed_number}",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if SAVE_ANNOTATED_VIDEO and writer is None:
                h, w = annotated.shape[:2]
                fps = source_fps / max(FRAME_SKIP, 1)
                if not 1 <= fps <= 240:
                    fps = 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
                if not writer.isOpened():
                    print(f"[warn] Could not open video writer: {OUTPUT_VIDEO_PATH}")
                    writer = None

            if writer is not None:
                writer.write(annotated)

            if SHOW_VIDEO:
                cv2.imshow("OmniGuard Perception", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("[!] Stopped by user")
                    break

            if processed_number % 20 == 0:
                elapsed = max(time.monotonic() - processing_started, 1e-6)
                rate = processed_number / elapsed
                print(
                    f"[frame={frame_number:6d}] objects={len(tracked_objects):2d} "
                    f"fire={len(latest_fire):2d} pose={len(latest_pose):2d} "
                    f"processing_fps={rate:.2f}"
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("=" * 76)
        print("PERCEPTION LOOP STOPPED")
        if SAVE_ANNOTATED_VIDEO:
            print(f"Annotated output: {OUTPUT_VIDEO_PATH}")
        print("=" * 76)


def parse_args():
    parser = argparse.ArgumentParser(description="Run OmniGuard perception on a video.")
    parser.add_argument("video", nargs="?", default=None, help="Input video; defaults to video.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.video)
