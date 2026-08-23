"""Per-track temporal state used on top of Ultralytics ByteTrack IDs."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from config import STATIONARY_DISTANCE_PIXELS


@dataclass
class TrackState:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    stationary_since: float | None = None
    last_position: tuple[float, float] | None = None
    last_box: tuple[int, int, int, int] | None = None
    owner_track_id: int | None = None
    owner_last_nearby: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dwell_time_seconds(self) -> float:
        if self.stationary_since is None:
            return 0.0
        return max(0.0, self.last_seen - self.stationary_since)


class TrackingState:
    """Stores temporal information; ByteTrack remains responsible for IDs."""

    def __init__(self, stationary_distance_pixels: float = STATIONARY_DISTANCE_PIXELS) -> None:
        self.states: dict[int, TrackState] = {}
        self.stationary_distance_pixels = stationary_distance_pixels

    @staticmethod
    def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: list[dict[str, Any]], timestamp: float | None = None) -> list[dict[str, Any]]:
        now = time.monotonic() if timestamp is None else float(timestamp)
        active_ids: set[int] = set()

        for det in detections:
            track_id = int(det.get("track_id", -1))
            if track_id < 0:
                # No tracker ID yet; do not invent one because the backend should
                # only receive actual persistent IDs.
                continue

            active_ids.add(track_id)
            box = det["box"]
            position = self._center(box)
            state = self.states.get(track_id)

            if state is None or state.class_name != det["class"]:
                state = TrackState(
                    track_id=track_id,
                    class_name=det["class"],
                    first_seen=now,
                    last_seen=now,
                    last_position=position,
                    last_box=box,
                )
                # A newly detected object is not considered stationary yet.
                self.states[track_id] = state
            else:
                movement = math.dist(state.last_position, position) if state.last_position else 0.0
                if movement <= self.stationary_distance_pixels:
                    if state.stationary_since is None:
                        state.stationary_since = state.last_seen
                else:
                    state.stationary_since = None

                state.last_seen = now
                state.last_position = position
                state.last_box = box

            det["dwell_time_seconds"] = round(state.dwell_time_seconds, 1)
            det["stationary"] = state.stationary_since is not None

        # ByteTrack handles actual track expiration. We retain recent local state
        # long enough to bridge short tracker gaps, but never synthesize detections.
        stale_after = 30.0
        for track_id in list(self.states):
            if track_id not in active_ids and now - self.states[track_id].last_seen > stale_after:
                self.states.pop(track_id, None)

        return [d for d in detections if int(d.get("track_id", -1)) >= 0]

    def get(self, track_id: int) -> TrackState | None:
        return self.states.get(track_id)

    def remember_owner(self, bag_track_id: int, person_track_id: int, timestamp: float) -> None:
        state = self.states.get(bag_track_id)
        if state is None:
            return
        state.owner_track_id = person_track_id
        state.owner_last_nearby = timestamp

    def owner_for(self, bag_track_id: int) -> tuple[int | None, float | None]:
        state = self.states.get(bag_track_id)
        if state is None:
            return None, None
        return state.owner_track_id, state.owner_last_nearby
