"""Temporal event persistence and deduplication for perception incidents."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class EventState:
    first_seen: float
    last_seen: float
    active: bool = False


class EventStateManager:
    """Turns repeated frame-level conditions into discrete incident events."""

    def __init__(self) -> None:
        self.states: dict[tuple[Any, ...], EventState] = {}

    @staticmethod
    def key(incident: dict[str, Any]) -> tuple[Any, ...]:
        event_type = incident.get("type")
        zone = incident.get("zone")

        # Crowd is a zone-level event; object events are track-level events.
        if event_type == "overcrowding":
            return (event_type, zone)
        return (event_type, zone, incident.get("tracked_object_id"))

    def update(
        self,
        incident: dict[str, Any],
        persistence_seconds: float,
        now: float | None = None,
    ) -> bool:
        """Return True exactly when this condition becomes a new active event."""
        now = time.monotonic() if now is None else float(now)
        key = self.key(incident)
        state = self.states.get(key)

        if state is None:
            self.states[key] = EventState(first_seen=now, last_seen=now, active=False)
            return persistence_seconds <= 0

        state.last_seen = now
        if not state.active and now - state.first_seen >= persistence_seconds:
            state.active = True
            return True
        return False

    def clear_missing(
        self,
        current_incidents: list[dict[str, Any]],
        now: float | None = None,
        clear_after_seconds: float = 0.0,
    ) -> None:
        """Clear conditions that are no longer present in the current frame."""
        now = time.monotonic() if now is None else float(now)
        current_keys = {self.key(i) for i in current_incidents}

        for key in list(self.states):
            state = self.states[key]
            if key in current_keys:
                continue
            if now - state.last_seen >= clear_after_seconds:
                self.states.pop(key, None)
