"""Backend incident transport. Keep the IncidentIn JSON contract unchanged."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import requests

from config import HTTP_TIMEOUT_SECONDS, INCIDENT_ENDPOINT


def build_incident(
    incident_type,
    zone=None,
    tracked_object_id=None,
    dwell_time_seconds=None,
    count=None,
    detection_confidence=None,
):
    """Build the existing backend-compatible IncidentIn payload."""
    return {
        "incident_id": f"inc_{uuid.uuid4().hex[:8]}",
        "type": incident_type,
        "zone": zone,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tracked_object_id": tracked_object_id,
        "dwell_time_seconds": dwell_time_seconds,
        "detection_confidence": detection_confidence,
        "count": count,
    }


_last_backend_failure_log = 0.0
_BACKEND_FAILURE_LOG_INTERVAL = 5.0


def send_incident(incident: dict, endpoint: str = INCIDENT_ENDPOINT) -> bool:
    global _last_backend_failure_log
    try:
        response = requests.post(endpoint, json=incident, timeout=HTTP_TIMEOUT_SECONDS)
        if response.status_code in (200, 201):
            print(f"[incident] sent {incident['type']} {incident['incident_id']}")
            return True
        print(f"[incident] backend returned {response.status_code}: {response.text[:300]}")
    except requests.RequestException as exc:
        now = time.monotonic()
        if now - _last_backend_failure_log >= _BACKEND_FAILURE_LOG_INTERVAL:
            print(f"[incident] backend unavailable: {exc}")
            _last_backend_failure_log = now
    return False
