import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from backend.database import Incident, SessionLocal, ZoneOccupancyRecord
from backend.schemas import AgentOutput, IncidentIn

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-3-haiku-20240307")

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

ZONE_STATIC_CONTEXT = {
    "Gate_3": {
        "description": "Busy pedestrian entrance gate.",
        "risk_level": "high",
        "notes": "Unattended objects near gates should be treated as high risk.",
    },
    "Gate_1": {
        "description": "Main entrance gate with moderate traffic.",
        "risk_level": "medium",
        "notes": "High foot traffic during class change times.",
    },
    "Restricted_Lab": {
        "description": "Restricted access laboratory area.",
        "risk_level": "critical",
        "notes": "Only authorized personnel allowed. Any intrusion should be treated as critical.",
    },
    "Library": {
        "description": "Study area with high occupancy.",
        "risk_level": "medium",
        "notes": "Overcrowding can occur during exam periods.",
    },
    "Cafeteria": {
        "description": "Highly crowded area during lunch hours.",
        "risk_level": "medium",
        "notes": "Monitor for overcrowding during peak hours.",
    },
}

TOOLS = [
    {
        "name": "get_recent_incidents",
        "description": "Get recent incidents in a zone. Use this to correlate the current incident with nearby recent incidents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "description": "Zone name, e.g. Gate_3",
                },
                "minutes": {
                    "type": "integer",
                    "description": "Lookback window in minutes",
                    "default": 30,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of incidents to return",
                    "default": 10,
                },
                "exclude_incident_id": {
                    "type": "string",
                    "description": "Exclude current incident id from results",
                },
            },
            "required": ["zone"],
        },
    },
    {
        "name": "get_object_track_history",
        "description": "Get recent incidents involving the same tracked object ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tracked_object_id": {
                    "type": "integer",
                    "description": "Tracked object ID from perception module",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of events to return",
                    "default": 20,
                },
            },
            "required": ["tracked_object_id"],
        },
    },
    {
        "name": "get_zone_context",
        "description": "Get static zone context, latest occupancy, and recent incidents for a zone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "description": "Zone name, e.g. Gate_3",
                }
            },
            "required": ["zone"],
        },
    },
]

SYSTEM_PROMPT = """
You are a bounded campus safety triage agent.
You receive structured incident facts from a deterministic computer vision perception module.
You must never invent facts that are not present.
Your job:
1. Assign severity: critical, high, medium, low
2. Correlate with recent related incidents if useful
3. Recommend an action: dispatch_security, verify, monitor, or none
4. Write a short explanation citing concrete data points
5. Draft a short control room notification

Rules:
- If uncertain, prefer lower severity rather than guessing high
- Use tools only when needed
- Do not output markdown
- Output exactly one JSON object
- The JSON object must have these keys:
  severity, reasoning_summary, correlated_incident_ids, recommended_action, notification_draft
"""

def _incident_to_tool_dict(incident: Incident) -> Dict[str, Any]:
    return {
        "incident_id": incident.incident_id,
        "type": incident.type,
        "zone": incident.zone,
        "timestamp": incident.timestamp,
        "tracked_object_id": incident.tracked_object_id,
        "dwell_time_seconds": incident.dwell_time_seconds,
        "detection_confidence": incident.detection_confidence,
        "count": incident.count,
        "severity": incident.severity,
        "status": incident.status,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
    }

def get_recent_incidents(
    zone: Optional[str] = None,
    minutes: int = 30,
    limit: int = 10,
    exclude_incident_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        query = db.query(Incident)
        if zone:
            query = query.filter(Incident.zone == zone)
        if exclude_incident_id:
            query = query.filter(Incident.incident_id != exclude_incident_id)
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        query = query.filter(Incident.created_at >= cutoff)
        query = query.order_by(Incident.created_at.desc()).limit(limit)
        return [_incident_to_tool_dict(item) for item in query.all()]
    finally:
        db.close()

def get_object_track_history(
    tracked_object_id: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        query = db.query(Incident).filter(Incident.tracked_object_id == tracked_object_id)
        query = query.order_by(Incident.created_at.desc()).limit(limit)
        return [_incident_to_tool_dict(item) for item in query.all()]
    finally:
        db.close()

def get_zone_context(zone: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        zone_info = ZONE_STATIC_CONTEXT.get(zone, {})
        occupancy = (
            db.query(ZoneOccupancyRecord)
            .filter(ZoneOccupancyRecord.zone == zone)
            .order_by(ZoneOccupancyRecord.updated_at.desc()) # ✅ Changed from created_at
            .first()
        )
        # ...
        return {
            "zone": zone,
            "static_context": zone_info,
            "latest_occupancy": occupancy if occupancy else None, # ✅ Removed .to_dict()
            "recent_incidents": incidents,
        }
    finally:
        db.close()

def _execute_tool(name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if name == "get_recent_incidents":
            return {"result": get_recent_incidents(**tool_input)}
        if name == "get_object_track_history":
            return {"result": get_object_track_history(**tool_input)}
        if name == "get_zone_context":
            return {"result": get_zone_context(**tool_input)}
        return {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        return {"error": str(exc)}

def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty LLM output")

    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output")

    return json.loads(text[start:end])

def _call_llm_agent(payload: IncidentIn) -> AgentOutput:
    if not client:
        raise RuntimeError("Anthropic client is not configured")

    user_payload = {
        "task": "triage_incident",
        "incident": payload.model_dump(),
        "instructions": (
            "Analyze this incident. "
            "Use tools only if needed. "
            "Return exactly one JSON object matching the AgentOutput schema."
        ),
    }

    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
    )

    content = response.content[0].text
    parsed = _extract_json(content)
    return AgentOutput(**parsed)

ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}
ALLOWED_ACTIONS = {"dispatch_security", "verify", "monitor", "none"}

def _deterministic_fallback(
    payload: IncidentIn,
    error_message: Optional[str] = None,
) -> AgentOutput:
    return AgentOutput(
        severity="low",
        reasoning_summary=error_message or "Fallback triage used because the model could not produce a valid result.",
        correlated_incident_ids=[],
        recommended_action="verify",
        notification_draft="Incident received; operator review required.",
    )

def run_triage_agent(payload: IncidentIn) -> AgentOutput:
    try:
        return _call_llm_agent(payload)
    except Exception as exc:
        return _deterministic_fallback(payload, str(exc))