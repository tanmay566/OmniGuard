"""Live per-zone occupancy tracking and backend delivery."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

from config import HTTP_TIMEOUT_SECONDS, OCCUPANCY_ENDPOINT, ZONE_CAPACITIES
from zones import point_for_object, point_in_zone


class OccupancyTracker:
    def __init__(self, zone_capacities: dict[str, int] | None = None, history_size: int = 30) -> None:
        self.zone_capacities = zone_capacities or ZONE_CAPACITIES
        self.current_counts = defaultdict(int)
        self.history = defaultdict(list)
        self.history_size = history_size
        self.last_send_time = 0.0

    def update(self, tracked_objects: list[dict]) -> dict[str, dict]:
        self.current_counts = defaultdict(int)
        persons = [o for o in tracked_objects if o.get("class") == "person"]

        for person in persons:
            x, y = point_for_object(person["box"], ground_point=True)
            for zone_name in self.zone_capacities:
                if point_in_zone(x, y, zone_name):
                    self.current_counts[zone_name] += 1

        occupancies: dict[str, dict] = {}
        for zone_name, capacity in self.zone_capacities.items():
            count = int(self.current_counts[zone_name])
            history = self.history[zone_name]
            history.append(count)
            if len(history) > self.history_size:
                history.pop(0)

            percentage = (count / capacity * 100.0) if capacity > 0 else 0.0
            occupancies[zone_name] = {
                "zone": zone_name,
                "current_count": count,
                "capacity": capacity,
                "occupancy_percentage": round(percentage, 1),
                "trend": self._detect_trend(zone_name),
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

        return occupancies

    def should_send(self, now: float | None = None, interval_seconds: float = 5.0) -> bool:
        now = time.monotonic() if now is None else float(now)
        if now - self.last_send_time >= interval_seconds:
            self.last_send_time = now
            return True
        return False

    def _detect_trend(self, zone_name: str) -> str:
        history = self.history[zone_name]
        if len(history) < 10:
            return "stable"
        recent = sum(history[-5:]) / 5.0
        older = sum(history[-10:-5]) / 5.0
        if recent > older + 0.5:
            return "increasing"
        if recent < older - 0.5:
            return "decreasing"
        return "stable"

    def get_occupancy_for_zone(self, zone_name: str):
        if zone_name not in self.zone_capacities:
            return None
        count = int(self.current_counts[zone_name])
        capacity = self.zone_capacities[zone_name]
        return {
            "zone": zone_name,
            "current_count": count,
            "capacity": capacity,
            "occupancy_percentage": round((count / capacity * 100.0) if capacity > 0 else 0.0, 1),
            "trend": self._detect_trend(zone_name),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }


_last_occupancy_failure_log = 0.0
_OCCUPANCY_FAILURE_LOG_INTERVAL = 5.0


def send_occupancy_update(occupancy_data: dict, endpoint: str = OCCUPANCY_ENDPOINT) -> bool:
    global _last_occupancy_failure_log
    try:
        response = requests.post(endpoint, json=occupancy_data, timeout=HTTP_TIMEOUT_SECONDS)
        if response.status_code in (200, 201):
            return True
        print(f"[occupancy] backend returned {response.status_code}: {response.text[:200]}")
    except requests.RequestException as exc:
        now = time.monotonic()
        if now - _last_occupancy_failure_log >= _OCCUPANCY_FAILURE_LOG_INTERVAL:
            print(f"[occupancy] backend unavailable: {exc}")
            _last_occupancy_failure_log = now
    return False
