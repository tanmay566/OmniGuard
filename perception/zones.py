"""Zone polygon definitions and point-in-zone helpers."""

from __future__ import annotations

import cv2
import numpy as np

# These coordinates are for the 640x480 working frame.
# Redefine them for your actual video/camera view.
ZONES = {

}


def point_in_zone(x: float, y: float, zone_name: str) -> bool:
    """Return True when a point is inside/on a configured zone polygon."""
    points = ZONES.get(zone_name)
    if points is None:
        return False
    polygon = np.asarray(points, dtype=np.int32)
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def point_for_object(box: tuple[int, int, int, int], *, ground_point: bool = True) -> tuple[int, int]:
    """Return a useful point for spatial reasoning.

    For people, bottom-center is usually a better proxy for the ground position.
    For bags/objects, callers may still choose the center.
    """
    x1, y1, x2, y2 = box
    if ground_point:
        return ((x1 + x2) // 2, y2)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def zone_names() -> list[str]:
    return list(ZONES.keys())
