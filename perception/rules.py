"""Deterministic event rules for OmniGuard perception."""

from __future__ import annotations

import math
from typing import Any

from config import (
    BAG_PERSON_NEARBY_PIXELS,
    FALL_BODY_ANGLE_DEGREES,
    FALL_MIN_KEYPOINT_CONFIDENCE,
    RESTRICTED_ZONES,
)
from tracker import TrackingState
from zones import point_for_object, point_in_zone

BAG_CLASSES = {"backpack", "handbag", "suitcase"}


def _distance_between_boxes(box_a, box_b) -> float:
    ax, ay = point_for_object(box_a, ground_point=False)
    bx, by = point_for_object(box_b, ground_point=False)
    return math.hypot(ax - bx, ay - by)


def check_crowd(tracked_objects, zones_config, threshold):
    incidents = []
    person_counts = {zone: 0 for zone in zones_config}
    for obj in tracked_objects:
        if obj.get("class") != "person":
            continue
        x, y = point_for_object(obj["box"], ground_point=True)
        for zone_name in zones_config:
            if point_in_zone(x, y, zone_name):
                person_counts[zone_name] += 1
    for zone_name, count in person_counts.items():
        if count >= threshold:
            incidents.append({"type": "overcrowding", "zone": zone_name, "count": count})
    return incidents


def check_baggage(tracked_objects, dwell_threshold_sec, tracking_state=None, now=None):
    import time
    now = time.monotonic() if now is None else float(now)
    bags = [o for o in tracked_objects if o.get("class") in BAG_CLASSES]
    persons = [o for o in tracked_objects if o.get("class") == "person"]
    incidents = []
    for bag in bags:
        dwell = float(bag.get("dwell_time_seconds", 0.0))
        if dwell < dwell_threshold_sec or not bag.get("stationary", False):
            continue
        bag_box = bag["box"]
        nearby = [p for p in persons if _distance_between_boxes(bag_box, p["box"]) <= BAG_PERSON_NEARBY_PIXELS]
        if nearby:
            nearest = min(nearby, key=lambda p: _distance_between_boxes(bag_box, p["box"]))
            if tracking_state is not None:
                tracking_state.remember_owner(int(bag["track_id"]), int(nearest["track_id"]), now)
            continue
        owner_id, owner_last_nearby = tracking_state.owner_for(int(bag["track_id"])) if tracking_state else (None, None)
        owner_visible_and_near = any(
            int(p["track_id"]) == owner_id and _distance_between_boxes(bag_box, p["box"]) <= BAG_PERSON_NEARBY_PIXELS
            for p in persons
        )
        if owner_visible_and_near:
            continue
        incidents.append({
            "type": "unattended_baggage",
            "tracked_object_id": int(bag["track_id"]),
            "dwell_time_seconds": round(dwell, 1),
            "detection_confidence": bag.get("confidence"),
        })
    return incidents


def check_intrusion(tracked_objects, restricted_zones=None):
    restricted_zones = RESTRICTED_ZONES if restricted_zones is None else restricted_zones
    incidents = []
    for person in tracked_objects:
        if person.get("class") != "person":
            continue
        x, y = point_for_object(person["box"], ground_point=True)
        for zone_name in restricted_zones:
            if point_in_zone(x, y, zone_name):
                incidents.append({
                    "type": "intrusion",
                    "zone": zone_name,
                    "tracked_object_id": int(person["track_id"]),
                    "detection_confidence": person.get("confidence"),
                })
    return incidents


def _body_orientation_is_horizontal(keypoints: list[dict[str, float]]) -> bool:
    # COCO keypoints: shoulders 5,6; hips 11,12; ankles 15,16.
    needed = [5, 6, 11, 12]
    if len(keypoints) <= max(needed):
        return False
    pts = [keypoints[i] for i in needed]
    if any(p.get("confidence", 1.0) < FALL_MIN_KEYPOINT_CONFIDENCE for p in pts):
        return False
    shoulder = ((pts[0]["x"] + pts[1]["x"]) / 2, (pts[0]["y"] + pts[1]["y"]) / 2)
    hip = ((pts[2]["x"] + pts[3]["x"]) / 2, (pts[2]["y"] + pts[3]["y"]) / 2)
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    # Angle from horizontal. A standing body is near vertical; a fallen body is closer to horizontal.
    angle_from_horizontal = abs(math.degrees(math.atan2(dy, dx)))
    angle_from_horizontal = min(angle_from_horizontal, 180 - angle_from_horizontal)
    return angle_from_horizontal < FALL_BODY_ANGLE_DEGREES


def check_fall(pose_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incidents = []
    for pose in pose_detections:
        if not _body_orientation_is_horizontal(pose.get("keypoints", [])):
            continue
        incidents.append({
            "type": "fall",
            "tracked_object_id": pose.get("track_id"),
            "detection_confidence": pose.get("confidence"),
        })
    return incidents


def check_fire(fire_detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incidents = []
    for det in fire_detections:
        label = str(det.get("class", "")).lower()
        if label not in {"fire", "smoke"}:
            continue
        incidents.append({
            "type": "fire" if label == "fire" else "smoke",
            "zone": None,
            "tracked_object_id": None,
            "detection_confidence": det.get("confidence"),
        })
    return incidents
