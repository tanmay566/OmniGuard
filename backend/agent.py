import os
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from dotenv import load_dotenv
from anthropic import Anthropic

from schemas import IncidentIn, AgentOutput
from database import SessionLocal, Incident, ZoneOccupancyRecord

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-3-haiku-20240307")


client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

ZONE_STATIC_CONTEXT = {
    "Gate_3": {
        "description": "Busy pedestrian entrance gate." ,
        "risk_level":"high",
        "notes": "Unattended objects near gates should be treated as high risk " ,

    },
    "Gate_1":{
        "desciption": "Main entrance gate with moderate traffic.",
        "risk_level":"medium",
        "notes": "High foot traffic druring class change times.",

    },
    "Restricted_Lab":{
        "description": "Restricted access laboratory area.",
        "risk_level":"critical",
        "notes": "Only authorized personnel allowed. Any intrusion should be treated as critical.",
    },
    "Library": {
        "description": "study area with high occupancy.",
        "risk_level":"medium",
        "notes": "Overcrowding can occur during exam periods.",
    },
    "Cafeteria": {
        "description": "highly crowded area during lunch hours.",
        "risk_level":"medium",
        "notes": "Monitor for overcrowding during peak hours.",
    },
}

# agent tools

TOOLS = [
    {
        "name": "get_recent_incidents",
        "description": (
            "Get recent incidents in a zone."
            "Use this to correlate the current incident with nearby recent incidents."
        ),
        "input_schema": {
            "type": "object",
            "properties":{
                "zone":{
                    "type": "string",
                    "description": "Zone name , e.g. Gate_3",
                },
                "minutes": {
                    "type": "integer",
                    "description": "Maximum number of incidents to return",
                    "default":10,
                },
                "exclude_incident_id": {
                    "type": "string",
                    "description": "Exclude current incident id from results",
                },
            },
            "required":["zone"],
        },
    },
    {
        "name": "get_object_track_history",
        "description": (
            "Get recent incident involving the same tracked object ID. "
            "Useful for unattended baggage or intrusion events."
        ),
        "input_schema":{
            "type":"object",
            "properties":{
                "tracked_object_id": {
                    "type":"integer",
                    "description": "Tracked object ID from perception module",
                },
                "limit":{
                    "type": "integer",
                    "description":"Maximum number of events to return",
                    "default":20,
                },
            },
            "required":["tracked_object_id"],
        },
    },
    {
        "name": "get_zone_context",
        "description": (
            "Get static zone context, latest occupancy, and recent incidents for a zone"
        ),
        "input_schema: {"
        "type": "object",
        "properties": {
            "zone":{
                "type": "string",
                "description": "Zone name, e.g Gate_3",
            },

        },
        "required":["zone"],
    },
]
 # system prompt for the agent
SYSTEM_PROMPT =  """
you are a bounded campus safety triage agent 
you recieve strcuctrred incident facts from a deterministic computer visoon perception module.
you don't see raw video 
you must never invent facts that are not present (that is NO room for HALLUCINATION) in the tool results

Your job:
1. Assign severity : critical , high , medium , low 
2.Corrrelate with recent related incidents if useful 
3. Recommend an action : dispatch_security , verify , monitor , or none
4. Write a short explanation citing concrete data points
5. Draft a short control room notification 

rules:
- If uncertain , prefer lowere severity rather than guessing high 
-Use tools only when needed 
-Do not output markdown
-Output exactly one Json object 
-the JSON object must have these keys:
severity , reasoning_summary , correlated_incident_ids , recommended_action , notification_draft"""


# database read helpers for tools ( tools give claude options what action or info it needs... databse read helpers actually get the info from database and convert it into a  meaniful format for claude i.e dict format)
def _incident_to_tool_dict(incident: Incident) -> Dict[str,Any]:
    """
    convert and Incident DB row into a small dict that tools can return
    """
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
        limit: int =10,
        exclude_incident_id: Optional[str]= None,
) ->List[Dict[str, Any]]:
    """ Tool: get recent incidents in a zone ."""
    db =SessionLocal()
    try:
        cutoff =datetime.utcnow() -timedelta(minutes=int(minutes or 30))

        query = db.query(Incident).filter(Incident.created_at >=cutoff)

        if zone :
            query =  query.filter(Incident.zone == zone)

        if exclude_incident_id:
            query = query.filter(Incident.incident_id != exclude_incident_id)

        incidents= (
            query.order_by(Incident.created_at.desc())
            .limit(int(limit or 10))
            .all()
        )
        return [ _incident_to_tool_dict(i) for i in incidents]
    finally:
        db.close()


def get_object_track_history(
        tracked_object_id: int ,
        limit: int  =20,
) -> List[Dict[str, Any]]:
    """Tool: get recent incidents involving the same tracked object ID."""
    db = SessionLocal()
    try:
        incidents = (
            db.query(Incident)
            .filter(Incident.tracked_object_id == int(tracked_object_id))
            .order_by(Incident.created.at.desc())
            .limit(int(limit or 20))
            .all()
        )
        return {
            "tracked_object_id": tracked_object_id,
            "event_count": len(incidents),
            "events": [_incident_to_tool_dict(i) for i in incidents],
        }
    finally:
        db.close()

def get_zone_context(zone:str) -> Dict[str, Any]:
    """ Tool: get zone context.
    Returns:
    -static zone information
    -latest occupancy record
    -recent incidents in the zone
    """
    db = SessionLocal()
    try:
        latest_occupancy = {
            db.query(ZoneOccupancyRecord)
            .filter(ZoneOccupancyRecord.zone ==zone)
            .order_by(ZoneOccupancyRecord.updated_at.desc())
            .first()
        }
        occupancy_payload = None
        if latest_occupancy :
            occupancy_payload = {
                 "zone": latest_occupancy.zone,
                "current_count": latest_occupancy.current_count,
                "capacity": latest_occupancy.capacity,
                "occupancy_percentage": latest_occupancy.occupancy_percentage,
                "timestamp": latest_occupancy.timestamp,
                "trend": latest_occupancy.trend,
                "updated_at": latest_occupancy.updated_at.isoformat(),
            }
            recent_incidents = get_recent_incidents(
                zone=zone,
                minutes=30,
                limit =5,
            )
            return {
                "zone" : zone ,
                "static_context": ZONE_STATIC_CONTEXT.get(
                    zone,
                    {
                         "description": "General campus zone.",
                         "risk_level": "medium",
                         "notes": "No special static context available.",

                    },
                ),
                "latest_occupancy": occupancy_payload,
                "recent_incidents":recent_incidents,
            }
    finally:
        db.close()
def _execute_tool(name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool requested by the model.
    """
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

#json parsing (jisko samajh nhi aa rha ye sab CLAUDE SE PUCHO BC, mujhse nai ..itna basic bhi nhi aata!!!!)
def _extract_json(text: str) -> Dict[str, Any]:
    """
    extract json from LLM output 
    handles cases where model returns Pure Json..
    JSON embedded in extra text """

    text = text.strip()

    if not text :
        raise ValueError("LLM returned empty text.")

    #remove markdown code block if present 
    code_block = re.search(r" '''(?:json)?\s*(.*?)\s*'''", text , re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    #find first and last 
    start = text.find("{")
    end = text.rfind("}") +1 
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in LLM OUTPUT.")

    json_text = text[start:end]
    return json.loads(json_text)

# LLM agent
def _call_llm_agent(payload: IncidentIn) -> AgentOutput:
    """
    Call Anthropic Claude with bounded tool use.

    Max tool-call turns:
        3

    If the model cannot produce valid output, raise exception.
    The outer wrapper will catch it and use deterministic fallback.
    """
    if not client:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    user_payload = {
        "task": "triage_incident",
        "incident": payload.model_dump(),
        "instructions": (
            "Analyze this incident. "
            "Use tools only if needed. "
            "Return exactly one JSON object matching the AgentOutput schema."
        ),
    }

    messages = [
        {
            "role": "user",
            "content": json.dumps(user_payload, default=str),
        }
    ]

    # bounded loop :maximum 3 tool-call turns.
    for _ in range(3):
        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=1200,
            temperature=0,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # If model wants to use tools, execute them and continue.
        if response.stop_reason == "tool_use":
            assistant_content = response.content
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )

            tool_results = []

            for block in assistant_content:
                if getattr(block, "type", None) == "tool_use":
                    tool_result = _execute_tool(block.name, block.input)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(tool_result, default=str),
                        }
                    )

            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

            continue
        # otherwise model should have final answer
        text_blocks = [
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]

        final_text = "\n".join([t for t in text_blocks if t])
        parsed_json = _extract_json(final_text)

        return AgentOutput.model_validate(parsed_json)

    raise RuntimeError("Agent exceeded maximum tool-call turns.")

#deterministic fallback


ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}
ALLOWED_ACTIONS = {"dispatch_security", "verify", "monitor", "none"}


def _escalate_severity(severity: str) -> str:
    """
    Escalate severity by one level.
    """
    mapping = {
        "low": "medium",
        "medium": "high",
        "high": "critical",
        "critical": "critical",
    }
    return mapping.get(severity, "medium")


def _deterministic_fallback(
    payload: IncidentIn,
    error_message: Optional[str] = None,
) -> AgentOutput:
    """
    Deterministic fallback if LLM fails.

    This ensures the system still works during:
    - API outage
    - missing API key
    - invalid LLM output
    - network failure
    """
    notes = []
    correlated_incident_ids = []

    incident_type = payload.type.lower().strip()

    # ------------------------------------------------------------------
    # Base severity rules
    # ------------------------------------------------------------------

    if incident_type == "fire":
        severity = "critical"
        action = "dispatch_security"
        notes.append("Fire/smoke incident confirmed by perception module.")

    elif incident_type == "intrusion":
        severity = "high"
        action = "dispatch_security"
        notes.append("Restricted-zone intrusion confirmed by perception module.")

    elif incident_type == "unattended_baggage":
        dwell = payload.dwell_time_seconds or 0.0
        confidence = payload.detection_confidence or 0.0

        if dwell >= 60:
            severity = "high"
            action = "dispatch_security"
        elif dwell >= 30:
            severity = "medium"
            action = "verify"
        else:
            severity = "low"
            action = "monitor"

        if confidence and confidence < 0.55:
            severity = "medium"
            action = "verify"
            notes.append("Detection confidence is below 0.55.")

        notes.append(f"Object stationary for {dwell:.1f} seconds.")

    elif incident_type == "overcrowding":
        count = payload.count or 0

        if count >= 25:
            severity = "critical"
            action = "dispatch_security"
        elif count >= 15:
            severity = "high"
            action = "dispatch_security"
        elif count >= 8:
            severity = "medium"
            action = "verify"
        else:
            severity = "low"
            action = "monitor"

        notes.append(f"Detected person count: {count}.")

    else:
        severity = "medium"
        action = "verify"
        notes.append(f"Unknown incident type: {payload.type}.")

    #simple correlation escalation
    if payload.zone:
        try:
            recent = get_recent_incidents(
                zone=payload.zone,
                minutes=20,
                limit=5,
                exclude_incident_id=payload.incident_id,
            )

            correlated_incident_ids = [
                item["incident_id"]
                for item in recent
                if item.get("incident_id")
            ][:3]

            if len(recent) >= 2 and severity != "critical":
                severity = _escalate_severity(severity)
                notes.append(
                    "Multiple recent incidents detected in the same zone."
                )
        except Exception:
            pass
    # ensuring high or critical incidents don't remain as passive monitoring
    if severity in {"high", "critical"} and action == "monitor":
        action = "dispatch_security"

    reasoning = (
        f"Deterministic fallback: {payload.type} in "
        f"{payload.zone or 'unknown zone'}. "
        + " ".join(notes)
    )

    if error_message:
        reasoning += f" Agent API error: {error_message}"

    notification = (
        f"{severity.upper()} alert: {payload.type} at "
        f"{payload.zone or 'unknown zone'}. "
        f"Recommended action: {action.replace('_', ' ')}."
    )

    return AgentOutput(
        severity=severity,
        reasoning_summary=reasoning,
        correlated_incident_ids=correlated_incident_ids,
        recommended_action=action,
        notification_draft=notification,
    )

#public agent entrypoint
def run_triage_agent(payload: IncidentIn) -> AgentOutput:
    """
    Main function used by backend.

    Input:
        IncidentIn

    Output:
        AgentOutput

    This function never crashes due to LLM failure.
    It falls back to deterministic rules.
    """
    try:
        agent_output = _call_llm_agent(payload)

        # Normalize values defensively.
        severity = agent_output.severity.lower().strip()
        if severity not in ALLOWED_SEVERITIES:
            severity = "medium"

        action = agent_output.recommended_action.lower().strip().replace(" ", "_")
        if action not in ALLOWED_ACTIONS:
            action = "verify"

        return AgentOutput(
            severity=severity,
            reasoning_summary=agent_output.reasoning_summary,
            correlated_incident_ids=agent_output.correlated_incident_ids or [],
            recommended_action=action,
            notification_draft=agent_output.notification_draft,
        )

    except Exception as exc:
        return _deterministic_fallback(payload, str(exc))




        


            
        