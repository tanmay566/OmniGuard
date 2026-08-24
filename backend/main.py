"""
main.py

FastAPI backend for Agentic Campus Safety Intelligence Platform.

Responsibilities:
- Receive incidents from perception module
- Run bounded triage agent
- Store incident + agent output in database
- Broadcast live updates to dashboard over WebSocket
- Support human-in-the-loop acknowledge/override/false-positive actions
- Receive and serve zone occupancy telemetry
"""

import os
import uuid
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from collections import deque

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from dotenv import load_dotenv

from database import (
    engine,
    get_db,
    Incident,
    ZoneOccupancyRecord,
    Base,
)
from schemas import (
    IncidentIn,
    IncidentOut,
    AckRequest,
    OverrideRequest,
    FalsePositiveRequest,
    ZoneOccupancy,
    ZoneOccupancyOut,
)
from agent import run_triage_agent
load_dotenv()
Base.metadata.create_all(bind=engine)
# app setup

app = FastAPI(
    title="Agentic Campus Safety Intelligence Platform",
    description= (
        "Deterministic perception + bounded LLM triage agent + human-in-the-loop."
    ),
    version="1.0.0",
)
#allow origin for running on localhost
app.add_middleware (
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
),


#frontend serving
FRONTEND_CANDIDATES = [
    Path(__file__).parent /" index.html",
    Path(__file__).parent.parent / "frontend" / "index.html",
    Path(__file__).parent.parent /"index.html",
    Path.cwd() / "index.html",
]

@app.get("/" , response_class=HTMLResponse)
def serve_frontend():

    """
    serve the Omniguard frontend if available.
    If the Html file is not found , return a simple backend status page.
    """
    for path in FRONTEND_CANDIDATES:
        if path.exists():
            return path.read_text(encoding="utf-8")

    return """
    <!doctype html>
    <html>
      <head>
        <title>OmniGuard API</title>
      </head>
      <body style="font-family: Arial, sans-serif; padding: 24px;">
        <h1>OmniGuard API is running</h1>
        <p>Frontend HTML file not found.</p>
        <p>Useful links:</p>
        <ul>
          <li><a href="/docs">API docs</a></li>
          <li><a href="/api/health">Health</a></li>
          <li><a href="/api/incidents">Incidents</a></li>
          <li><a href="/api/zones">Zones</a></li>
          <li><a href="/zones/occupancy">Zone occupancy</a></li>
          <li><a href="/api/stats">Stats</a></li>
        </ul>
      </body>
    </html>"""
#constants

ALLOWED_SEVERITIES ={"critical","high", "medium","low",}
ALLOWED_ACTIONS={"dispatch_security", "verify", "monitor" , "none"}
ALLOWED_STATUSES={"new", "acknowledged", "resolved", "false_positive"}
ACTIVE_STATUSES={"new" , "acknowledged"}

SEVERITY_PRIORITY = {
    "critical":0,
    "high":1,
    "medium":2,
    "low":3,
}
STATUS_PRIORITY={
    "new":0,
    "acknowledges":1,
    "resolved":2,
    "false_positive":3,
}

SYSTEM_START_TIME = datetime.now(timezone.utc)

#frontend-depending map configuration(hardcoded for now)
#coordinates (eg x=18, y=72 means place pin at left : 18%, top:72%)
ZONE_CONFIG = {
    "Gate_1": {
        "display_name": "Gate 1",
        "type": "entrance",
        "x": 10,
        "y": 80,
        "capacity": 15,
    },
    "Gate_3": {
        "display_name": "Gate 3",
        "type": "entrance",
        "x": 22,
        "y": 72,
        "capacity": 12,
    },
    "Restricted_Lab": {
        "display_name": "Restricted Lab",
        "type": "restricted",
        "x": 64,
        "y": 28,
        "capacity": 8,
    },
    "Library": {
        "display_name": "Library",
        "type": "academic",
        "x": 48,
        "y": 46,
        "capacity": 60,
    },
    "Parking": {
        "display_name": "Parking",
        "type": "parking",
        "x": 82,
        "y": 70,
        "capacity": 25,
    },
    "Quad": {
        "display_name": "Quad",
        "type": "open_area",
        "x": 50,
        "y": 62,
        "capacity": 80,
    },
    "Hostel_A": {
        "display_name": "Hostel A",
        "type": "residence",
        "x": 28,
        "y": 22,
        "capacity": 40,
    },
    "Academic_Block": {
        "display_name": "Academic Block",
        "type": "academic",
        "x": 58,
        "y": 40,
        "capacity": 70,
    },
}

DEFAULT_ZONE_CONFIG = {
    "display_name": "Unknown Zone",
    "type": "general",
    "x": 50,
    "y": 50,
    "capacity": 20,
}
#IN-MEMORY NOTIFICATION LOG 
NOTIFICATION_LOG =deque(maxlen=200)

#websocket manager 

class ConnectionManager:

    ""
    """ manages live dashboard websocket connections.
    
    Event broadcast:
    connected 
    ping 
    incident_new
    incident_update
    zone_occupancy
    demo_reset"""
    def __init__(self):
        self.active_connections: List[WebSocket]= []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self,websocket:WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def broadcast(self, message:dict):
        dead_connections= []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for connection in dead_connections:
            self.disconnect(connection)
manager = ConnectionManager()

#time helpers
def utcnow_iso() -> str:
    """
    ISO 8601 timestamp with Z suffix.
    Matches frontend/perception timestamp style.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def db_now() -> datetime:
    """
    Naive UTC datetime for SQLite/SQLAlchemy.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

#generic helpers

def incident_to_out(incident: Incident) -> dict:
    """
    Convert SQLAlchemy Incident object into JSON-compatible IncidentOut dict.

    This is the combined object used by the dashboard:
    - perception facts
    - agent reasoning
    - operator status
    - backend timestamps
    """
    # Defensive handling for JSON column.
    if incident.correlated_incident_ids is None:
        incident.correlated_incident_ids = []

    return IncidentOut.model_validate(incident).model_dump(mode="json")


def sort_incidents(incidents: List[Incident]) -> List[Incident]:
    """
    Sort incidents for the dashboard alert list.

    Priority:
    1. Active incidents first: new, acknowledged
    2. Status priority
    3. Severity priority: critical > high > medium > low
    4. Newest first
    """

    def sort_key(incident: Incident):
        severity = (incident.severity or "medium").lower()
        status = (incident.status or "new").lower()

        created_timestamp = 0
        if incident.created_at:
            try:
                created_timestamp = incident.created_at.timestamp()
            except Exception:
                created_timestamp = 0

        return (
            0 if status in ACTIVE_STATUSES else 1,
            STATUS_PRIORITY.get(status, 9),
            SEVERITY_PRIORITY.get(severity, 9),
            -created_timestamp,
        )

    return sorted(incidents, key=sort_key)


def get_incident_or_404(db: Session, incident_id: str) -> Incident:
    """
    Fetch incident by perception incident_id.

    Also supports numeric DB id as fallback:
        /api/incidents/12
    """
    incident = (
        db.query(Incident)
        .filter(Incident.incident_id == incident_id)
        .first()
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    return incident


def occupancy_status(occupancy_percentage: float) -> str:
    """
    Convert occupancy percentage into UI status.
    """
    if occupancy_percentage >= 100:
        return "critical"
    if occupancy_percentage >= 85:
        return "warning"
    return "normal"


def zone_display_name(zone: str) -> str:
    """
    Convert zone id to human-readable name.
    Example:
        Gate_3 -> Gate 3
    """
    cfg = ZONE_CONFIG.get(zone)
    if cfg and cfg.get("display_name"):
        return cfg["display_name"]

    return zone.replace("_", " ").strip()


def log_notification(
    event: str,
    incident_payload: Optional[dict] = None,
    message: Optional[str] = None,
):
    """
    Simulated notification routing.

    This is where a real system could send:
    - webhook
    - SMS
    - email
    - control-room popup

    For demo, we store it in memory and expose /api/notifications.
    """
    notification = {
        "id": f"notif_{uuid.uuid4().hex[:10]}",
        "event": event,
        "message": message,
        "incident_id": None,
        "severity": None,
        "zone": None,
        "time": utcnow_iso(),
    }

    if incident_payload:
        notification["incident_id"] = incident_payload.get("incident_id")
        notification["severity"] = incident_payload.get("severity")
        notification["zone"] = incident_payload.get("zone")

        if not message:
            notification["message"] = (
                incident_payload.get("notification_draft")
                or incident_payload.get("reasoning_summary")
                or f"{event} for incident {incident_payload.get('incident_id')}"
            )

    NOTIFICATION_LOG.appendleft(notification)

#deduplication /cooldown
def find_recent_duplicate(
    db: Session,
    payload: IncidentIn,
    cooldown_seconds: int = 90,
) -> Optional[Incident]:
    """
    Simple deduplication/cooldown logic.

    If perception repeatedly sends similar incidents in a short window,
    we avoid spamming the dashboard and agent.

    Rules:
    - unattended_baggage / intrusion:
        same type + zone + tracked_object_id
    - fire / overcrowding:
        same type + zone
    """
    cutoff = db_now() - timedelta(seconds=cooldown_seconds)

    query = db.query(Incident).filter(
        Incident.type == payload.type,
        Incident.created_at >= cutoff,
    )

    if payload.zone:
        query = query.filter(Incident.zone == payload.zone)
    else:
        query = query.filter(Incident.zone.is_(None))

    if (
        payload.type in {"unattended_baggage", "intrusion"}
        and payload.tracked_object_id is not None
    ):
        query = query.filter(
            Incident.tracked_object_id == payload.tracked_object_id
        )

    elif payload.type in {"fire", "overcrowding"} and payload.zone:
        # Same type and zone is enough for short cooldown.
        pass

    else:
        return None

    return query.order_by(Incident.created_at.desc()).first()
#core incident processing pipeleine 

async def save_and_broadcast_incident(
        db: Session ,
        incident : Incident,
        event_type : str = "incident_update",
        notify : bool = False,
) ->dict:
    """
    Commit incident changes , broadcast to dashboard , optionally log notification"""
    db.commit()
    db.refresh(incident)

    payload = incident_to_out(incident)

    await manager.broadcast(
        {
            "event": event_type,
            "data": payload ,
        }
    )
    if notify and payload.get("notification_draft"):
        log_notification(
            event=event_type,
            incident_payload=payload,
            message=payload.get("notification_draft"),
        )

    return payload


async def process_incident_payload(
    payload: IncidentIn,
    db: Session,
) -> dict:
    """
    Main incident pipeline.

    Pipeline:
    1. If exact incident_id exists, return it.
    2. If recent duplicate exists, reuse/update it.
    3. Save raw perception incident.
    4. Run bounded triage agent.
    5. Save agent output.
    6. Broadcast to dashboard.
    """

    #exact duplicate incident_id
    existing = (
        db.query(Incident)
        .filter(Incident.incident_id == payload.incident_id)
        .first()
    )
    if existing:
        return incident_to_out(existing)

    #short term duplicate/cooldown 

    duplicate = find_recent_duplicate(db, payload)

    if duplicate:
        incident = duplicate
        #update mutable perception facts

        incident.timestamp = payload.timestamp
        if payload.dwell_time_seconds is not None:
            incident.dwell_time_seconds = payload.dwell_time_seconds

        if payload.detection_confidence is not None:
            incident.detection_confidence = payload.detection_confidence
        if payload.count is not None:
            incident.count = payload.count

        incident.updated_at = db_now()

        if incident.severity:
            return await save_and_broadcast_incident(
                db,
                incident,
                event_type="incident_update",
                notify=False,
            )
    else:

        #save raw incidents first

        incident = Incident(
            incident_id=payload.incident_id,
            type =payload.type,
            zone=payload.zone,
            timestamp=payload.timestamp,
            tracked_object_id=payload.tracked_object_id,
            dwell_time_seconds=payload.dwell_time_seconds,
            detection_confidence=payload.detection_confidence,
            count=payload.count,
            status="new",
            correlated_incident_ids=[],
            created_at=db_now(),
            updated_at=db_now(),
        )

        db.add(incident)
        db.commit()
        db.refresh(incident)

        #running agent
    agent_output = await run_in_threadpool(
        run_triage_agent,
        payload,
    )

    #save agent output 
    incident.severity = agent_output.severity
    incident.reasoning_summary = agent_output.reasoning_summary
    incident.correlated_incident_ids = agent_output.correlated_incident_ids or []
    incident.recommended_action = agent_output.recommended_action
    incident.notification_draft = agent_output.notification_draft
    incident.updated_at = db_now()

    #broadcast and return 

    event_type = "incident_update"if duplicate else "incident_new"

    return await save_and_broadcast_incident(
        db, 
        incident,
        event_type=event_type,
        notify=not duplicate,
    )


#occupancy helpers
def get_latest_occupancy_by_zone(db: Session) -> Dict[str, ZoneOccupancyRecord]:
    """
    Get latest occupancy record for each zone.
    """
    subquery = (
        db.query(
            ZoneOccupancyRecord.zone,
            func.max(ZoneOccupancyRecord.id).label("max_id"),
        )
        .group_by(ZoneOccupancyRecord.zone)
        .subquery()
    )

    latest_records = (
        db.query(ZoneOccupancyRecord)
        .join(subquery, ZoneOccupancyRecord.id == subquery.c.max_id)
        .all()
    )

    return {record.zone: record for record in latest_records}


def latest_zone_occupancy_payloads(db: Session) -> List[dict]:
    """
    Convert latest occupancy records into JSON payloads.
    """
    latest_by_zone = get_latest_occupancy_by_zone(db)

    payloads = []

    for record in latest_by_zone.values():
        payloads.append(
            ZoneOccupancyOut.model_validate(record).model_dump(mode="json")
        )

    return payloads

#zone card helper

def get_zone_cards(db: Session) -> List[dict]:
    """
    Build zone cards for frontend map + occupancy gauges.

    Each zone card contains:
    - static map coordinates
    - latest occupancy
    - occupancy status
    - active incident count
    - latest incident summary
    """
    latest_by_zone = get_latest_occupancy_by_zone(db)

    # Include both configured zones and zones seen in occupancy data.
    all_zones = set(ZONE_CONFIG.keys()) | set(latest_by_zone.keys())

    zone_cards = []

    for zone in sorted(all_zones):
        cfg = ZONE_CONFIG.get(zone, DEFAULT_ZONE_CONFIG)

        occ = latest_by_zone.get(zone)

        if occ:
            current_count = occ.current_count
            capacity = occ.capacity
            occupancy_percentage = occ.occupancy_percentage
            trend = occ.trend or "stable"
            timestamp = occ.timestamp
            updated_at = occ.updated_at.isoformat() if occ.updated_at else None
        else:
            current_count = 0
            capacity = cfg.get("capacity", 20)
            occupancy_percentage = 0.0
            trend = "stable"
            timestamp = utcnow_iso()
            updated_at = None

        active_incident_count = (
            db.query(Incident)
            .filter(
                Incident.zone == zone,
                Incident.status.in_(["new", "acknowledged"]),
            )
            .count()
        )

        latest_incident = (
            db.query(Incident)
            .filter(Incident.zone == zone)
            .order_by(Incident.created_at.desc())
            .first()
        )

        latest_incident_summary = None
        if latest_incident:
            latest_incident_summary = {
                "incident_id": latest_incident.incident_id,
                "type": latest_incident.type,
                "severity": latest_incident.severity,
                "status": latest_incident.status,
                "created_at": latest_incident.created_at.isoformat()
                if latest_incident.created_at
                else None,
            }

        occ_status = occupancy_status(occupancy_percentage)

        # Map pin color logic for frontend.
        if active_incident_count > 0 or occ_status == "critical":
            pin_color = "red"
        elif occ_status == "warning":
            pin_color = "amber"
        else:
            pin_color = "green"

        zone_cards.append(
            {
                "zone": zone,
                "display_name": cfg.get("display_name", zone_display_name(zone)),
                "location_type": cfg.get("type", "general"),
                "x": cfg.get("x", 50),
                "y": cfg.get("y", 50),
                "current_count": current_count,
                "capacity": capacity,
                "occupancy_percentage": round(occupancy_percentage, 1),
                "trend": trend,
                "timestamp": timestamp,
                "updated_at": updated_at,
                "occupancy_status": occ_status,
                "active_incidents": active_incident_count,
                "latest_incident": latest_incident_summary,
                "map_pin": {
                    "x": cfg.get("x", 50),
                    "y": cfg.get("y", 50),
                    "color": pin_color,
                },
            }
        )

    return zone_cards

#stats helper
def get_stats(db: Session) -> dict:
    """
    Stats for dashboard header cards / status chips.
    """
    active_filter = Incident.status.in_(["new", "acknowledged"])

    active_alerts = (
        db.query(Incident)
        .filter(active_filter)
        .count()
    )

    critical_active = (
        db.query(Incident)
        .filter(active_filter, Incident.severity == "critical")
        .count()
    )

    high_active = (
        db.query(Incident)
        .filter(active_filter, Incident.severity == "high")
        .count()
    )

    medium_active = (
        db.query(Incident)
        .filter(active_filter, Incident.severity == "medium")
        .count()
    )

    low_active = (
        db.query(Incident)
        .filter(active_filter, Incident.severity == "low")
        .count()
    )

    status_rows = (
        db.query(Incident.status, func.count(Incident.incident_id))
        .group_by(Incident.status)
        .all()
    )

    status_counts = {
        "new": 0,
        "acknowledged": 0,
        "resolved": 0,
        "false_positive": 0,
    }

    for status_value, count_value in status_rows:
        if status_value in status_counts:
            status_counts[status_value] = count_value

    latest_occupancy = get_latest_occupancy_by_zone(db)

    occupancy_critical = sum(
        1
        for occ in latest_occupancy.values()
        if occ.occupancy_percentage >= 100
    )

    occupancy_warning = sum(
        1
        for occ in latest_occupancy.values()
        if 85 <= occ.occupancy_percentage < 100
    )

    agent_status = "online" if os.getenv("ANTHROPIC_API_KEY") else "fallback"

    return {
        "active_alerts": active_alerts,
        "critical": critical_active,
        "high": high_active,
        "medium": medium_active,
        "low": low_active,
        "status_counts": status_counts,
        "zones_tracked": len(get_latest_occupancy_by_zone(db)),
        "occupancy_critical": occupancy_critical,
        "occupancy_warning": occupancy_warning,
        "agent_status": agent_status,
        "websocket_clients": len(manager.active_connections),
        "time": utcnow_iso(),
    }

#activity feed helper
def get_activity_feed(db: Session, limit: int = 30) -> List[dict]:
    """
    Unified activity feed for frontend live feed panel.

    For MVP, this is derived from incidents.
    Later, it can also include occupancy alerts, operator actions, etc.
    """
    incidents = (
        db.query(Incident)
        .order_by(Incident.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )

    items = []

    for incident in incidents:
        title = (
            f"{zone_display_name(incident.zone or 'Unknown Zone')}: "
            f"{incident.type.replace('_', ' ').title()}"
        )

        items.append(
            {
                "id": incident.incident_id,
                "incident_id": incident.incident_id,
                "type": "incident",
                "event": incident.type,
                "severity": incident.severity,
                "status": incident.status,
                "zone": incident.zone,
                "title": title,
                "message": incident.notification_draft
                or incident.reasoning_summary
                or "",
                "time": incident.created_at.isoformat()
                if incident.created_at
                else None,
            }
        )

    return items
# demo/ stimulation helper
class SimulateRequest(BaseModel):
    """
    Request body for demo incident simulation.
    """
    scenario: str ="unattended_baggage"
    zone: Optional[str] = None


def make_simulation_payload(req: SimulateRequest) -> IncidentIn:
    """Create a simulated IncidentIn payload for demo/testing."""

    scenario = req.scenario.lower().strip()

    """ Create a simulated IncidentIn payload for demo/testing.
    """
    scenario = req.scenario.lower().strip()
    now = utcnow_iso()
    incident_id = f"inc_demo_{uuid.uuid4().hex[:10]}"

    if scenario in {"fire", "smoke", "fire_smoke"}:
        return IncidentIn(
            incident_id=incident_id,
            type="fire",
            zone=req.zone or "Library",
            timestamp=now,
            detection_confidence=round(random.uniform(0.90, 0.99), 2),
        )

    if scenario in {"intrusion", "restricted_entry", "restricted_zone"}:
        return IncidentIn(
            incident_id=incident_id,
            type="intrusion",
            zone=req.zone or "Restricted_Lab",
            timestamp=now,
            tracked_object_id=random.randint(10, 99),
            detection_confidence=round(random.uniform(0.75, 0.96), 2),
        )

    if scenario in {"overcrowding", "crowd", "crowd_count"}:
        return IncidentIn(
            incident_id=incident_id,
            type="overcrowding",
            zone=req.zone or "Quad",
            timestamp=now,
            count=random.randint(22, 48),
            detection_confidence=round(random.uniform(0.72, 0.94), 2),
        )

    # Default: unattended baggage
    return IncidentIn(
        incident_id=incident_id,
        type="unattended_baggage",
        zone=req.zone or "Gate_3",
        timestamp=now,
        tracked_object_id=random.randint(10, 99),
        dwell_time_seconds=float(random.choice([25, 35, 45, 70, 90])),
        detection_confidence=round(random.uniform(0.72, 0.97), 2),
    )
#websocket routes

async def handle_websocket(websocket: WebSocket):
    """ shared websocket handler
    
    frontend can connect to:
    /ws
    /ws/events"""

    await manager.connect(websocket)

    try : 
        await websocket.send_json(
            {
                "event": "conncted",
                "data": {
                    "message": "OmniGuard live event stream connected.",
                    "time": utcnow_iso(),
                },
            }
        )
        while True:
            data = await websocket.receive_text()

            if data.lower() == "ping":
                await websocket.send_json(
                    {
                        "event": "pong",
                        "data": {
                            "time": utcnow_iso(),
                        },
                    }
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws")
async def websocket_main(websocket: WebSocket):
    await handle_websocket(websocket)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await handle_websocket(websocket)
#health

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "OmniGuard-backend",
        "time":utcnow_iso(),
    }

#incident ingestion
@app.post("/api/incidents", response_model=IncidentOut)
@app.post("/incidents", response_model=IncidentOut, include_in_schema=False)
async def ingest_incident(
    payload: IncidentIn,
    db: Session = Depends(get_db),
):
    """
    Main perception-to-backend endpoint.

    Perception module sends IncidentIn here.
    """
    return await process_incident_payload(payload, db)
#incident read endpoints
@app.get("/api/incidents", response_model=List[IncidentOut])
@app.get("/incidents", response_model=List[IncidentOut], include_in_schema=False)
def list_incidents(
    limit: int = Query(default=50, ge=1, le=200),
    status: Optional[str] = None,
    zone: Optional[str] = None,
    severity: Optional[str] = None,
    active: bool = False,
    db: Session = Depends(get_db),
):
    """
    Get incidents for dashboard alert list.

    Useful query params:
        /api/incidents?active=true
        /api/incidents?status=new
        /api/incidents?zone=Gate_3
        /api/incidents?severity=critical
    """
    query = db.query(Incident)

    if active:
        query = query.filter(Incident.status.in_(["new", "acknowledged"]))

    if status:
        query = query.filter(Incident.status == status)

    if zone:
        query = query.filter(Incident.zone == zone)

    if severity:
        query = query.filter(Incident.severity == severity)

    incidents = (
        query.order_by(Incident.created_at.desc())
        .limit(limit)
        .all()
    )

    incidents = sort_incidents(incidents)

    return [incident_to_out(incident) for incident in incidents]


@app.get("/api/incidents/{incident_id}", response_model=IncidentOut)
def get_incident(
    incident_id: str,
    db: Session = Depends(get_db),
):
    """
    Get one incident.
    """
    incident = get_incident_or_404(db, incident_id)
    return incident_to_out(incident)


@app.get("/api/incidents/{incident_id}/explain")
def explain_incident(
    incident_id: str,
    db: Session = Depends(get_db),
):
    """
    Explainability endpoint for the frontend "why flagged?" panel.

    This does not call the LLM live.
    It returns stored reasoning from the agent.
    """
    incident = get_incident_or_404(db, incident_id)

    return {
        "incident_id": incident.incident_id,
        "type": incident.type,
        "zone": incident.zone,
        "severity": incident.severity,
        "status": incident.status,
        "reasoning_summary": incident.reasoning_summary,
        "recommended_action": incident.recommended_action,
        "notification_draft": incident.notification_draft,
        "correlated_incident_ids": incident.correlated_incident_ids or [],
        "evidence": {
            "timestamp": incident.timestamp,
            "tracked_object_id": incident.tracked_object_id,
            "dwell_time_seconds": incident.dwell_time_seconds,
            "detection_confidence": incident.detection_confidence,
            "count": incident.count,
        },
        "generated_at": incident.updated_at.isoformat()
        if incident.updated_at
        else None,
    }
#operator actions 
@app.post("/api/incidents/{incident_id}/acknowledge", response_model=IncidentOut)
async def acknowledge_incident(
    incident_id: str,
    body: AckRequest,
    db: Session = Depends(get_db),
):
    """
    Operator acknowledges an incident.
    """
    incident = get_incident_or_404(db, incident_id)

    incident.status = "acknowledged"
    incident.acknowledged_at = db_now()
    incident.acknowledged_by = body.operator_name
    incident.updated_at = db_now()

    return await save_and_broadcast_incident(
        db,
        incident,
        event_type="incident_update",
    )


@app.post("/api/incidents/{incident_id}/resolve", response_model=IncidentOut)
async def resolve_incident(
    incident_id: str,
    db: Session = Depends(get_db),
):
    """
    Operator resolves an incident.
    """
    incident = get_incident_or_404(db, incident_id)

    incident.status = "resolved"
    incident.updated_at = db_now()

    if not incident.acknowledged_at:
        incident.acknowledged_at = db_now()
        incident.acknowledged_by = "operator_resolve"

    return await save_and_broadcast_incident(
        db,
        incident,
        event_type="incident_update",
    )


@app.post("/api/incidents/{incident_id}/override", response_model=IncidentOut)
async def override_incident(
    incident_id: str,
    body: OverrideRequest,
    db: Session = Depends(get_db),
):
    """
    Operator overrides agent severity.

    This is a core human-in-the-loop feature.
    """
    if body.new_severity not in ALLOWED_SEVERITIES:
        raise HTTPException(
            status_code=422,
            detail="new_severity must be one of: critical, high, medium, low",
        )

    incident = get_incident_or_404(db, incident_id)

    incident.severity = body.new_severity

    if hasattr(incident, "status_reason"):
        incident.status_reason = body.reason

    # If operator overrides a new incident, treat it as acknowledged.
    if incident.status == "new":
        incident.status = "acknowledged"
        incident.acknowledged_at = db_now()
        incident.acknowledged_by = "operator_override"

    incident.updated_at = db_now()

    return await save_and_broadcast_incident(
        db,
        incident,
        event_type="incident_update",
    )


@app.post("/api/incidents/{incident_id}/false-positive", response_model=IncidentOut)
@app.post(
    "/api/incidents/{incident_id}/false_positive",
    response_model=IncidentOut,
    include_in_schema=False,
)
@app.post(
    "/api/incidents/{incident_id}/mark-false-positive",
    response_model=IncidentOut,
    include_in_schema=False,
)
async def mark_false_positive(
    incident_id: str,
    body: FalsePositiveRequest,
    db: Session = Depends(get_db),
):
    """
    Operator marks incident as false positive.
    """
    incident = get_incident_or_404(db, incident_id)

    incident.status = "false_positive"

    if hasattr(incident, "status_reason"):
        incident.status_reason = body.reason

    if not incident.acknowledged_at:
        incident.acknowledged_at = db_now()
        incident.acknowledged_by = "operator_false_positive"

    incident.updated_at = db_now()

    return await save_and_broadcast_incident(
        db,
        incident,
        event_type="incident_update",
    )


class ManualStatusUpdate(BaseModel):
    """
    Generic status update endpoint for frontend compatibility.
    """
    status: str
    operator_name: Optional[str] = None
    reason: Optional[str] = None


@app.post("/api/incidents/{incident_id}/status", response_model=IncidentOut)
async def update_incident_status(
    incident_id: str,
    body: ManualStatusUpdate,
    db: Session = Depends(get_db),
):
    """
    Generic operator status update.

    Accepts:
        new
        acknowledged
        resolved
        false_positive
    """
    status = body.status.lower().strip()

    if status not in ALLOWED_STATUSES:
        raise HTTPException(
            status_code=422,
            detail="status must be one of: new, acknowledged, resolved, false_positive",
        )

    incident = get_incident_or_404(db, incident_id)

    incident.status = status

    if status in {"acknowledged", "resolved", "false_positive"}:
        incident.acknowledged_at = db_now()
        incident.acknowledged_by = body.operator_name or "operator"

    if body.reason and hasattr(incident, "status_reason"):
        incident.status_reason = body.reason

    incident.updated_at = db_now()

    return await save_and_broadcast_incident(
        db,
        incident,
        event_type="incident_update",
    )
#zone occupancy endpoints
@app.post("/api/occupancy", response_model=ZoneOccupancyOut)
@app.post("/occupancy", response_model=ZoneOccupancyOut, include_in_schema=False)
async def ingest_occupancy(
    payload: ZoneOccupancy,
    db: Session = Depends(get_db),
):
    """
    Receive live occupancy telemetry from perception module.

    This is not an incident.
    It powers occupancy gauges and zone map status.
    """
    record = ZoneOccupancyRecord(
        zone=payload.zone,
        current_count=payload.current_count,
        capacity=payload.capacity,
        occupancy_percentage=payload.occupancy_percentage,
        timestamp=payload.timestamp,
        trend=payload.trend,
        updated_at=db_now(),
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    out = ZoneOccupancyOut.model_validate(record).model_dump(mode="json")

    await manager.broadcast(
        {
            "event": "zone_occupancy",
            "data": out,
        }
    )
    #optional stimulated notification for severe overcrowding 
    if record.occupancy_percentage >=100:
        log_notification(
            event="occupancy_alert",
            message=(
                f"Occupancy alert in {zone_display_name(record.zone)}:"
                f"{record.current_count}/{record.capacity}"
                f"({record.occupancy_percentage : .1f}%)."
            ),
        )
    return out 
@app.get("/zones/occupancy", response_model=List[ZoneOccupancyOut])
@app.get(
    "/api/zones/occupancy",
    response_model=List[ZoneOccupancyOut],
    include_in_schema=False,
)
def get_zones_occupancy(db: Session = Depends(get_db)):
    """
    Get latest occupancy for all zones.
    """
    return latest_zone_occupancy_payloads(db)


@app.get("/api/zones")
@app.get("/zones", include_in_schema=False)
def get_zones(db: Session = Depends(get_db)):
    """
    Get zone cards for frontend map + gauges.

    Returns:
    - zone id
    - display name
    - map coordinates as percentages
    - latest occupancy
    - occupancy status
    - active incidents
    - latest incident summary
    """
    return get_zone_cards(db)


# stats/ activity endpoints

@app.get("/api/stats")
@app.get("/stats", include_in_schema=False)
def stats(db: Session = Depends(get_db)):
    """
    Dashboard stats for header cards/status chips.
    """
    return get_stats(db)


@app.get("/api/activity")
@app.get("/activity", include_in_schema=False)
def activity_feed(
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Activity feed for live feed panel.
    """
    return get_activity_feed(db, limit)


@app.get("/api/notifications")
@app.get("/notifications", include_in_schema=False)
def notifications(
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Simulated control-room notification log.
    """
    return list(NOTIFICATION_LOG)[:limit]


@app.get("/api/system/status")
@app.get("/system/status", include_in_schema=False)
def system_status(db: Session = Depends(get_db)):
    """
    System status for frontend status indicators.
    """
    incident_count = db.query(Incident).count()
    occupancy_count = db.query(ZoneOccupancyRecord).count()

    uptime_seconds = int(
        (datetime.now(timezone.utc) - SYSTEM_START_TIME).total_seconds()
    )

    return {
        "backend": "online",
        "agent": "online" if os.getenv("ANTHROPIC_API_KEY") else "fallback",
        "websocket_clients": len(manager.active_connections),
        "incident_count": incident_count,
        "occupancy_record_count": occupancy_count,
        "uptime_seconds": uptime_seconds,
        "time": utcnow_iso(),
    }


#demo / simulation endpoints

@app.post("/api/demo/simulate", response_model=IncidentOut)
async def simulate_demo_incident(
    req: SimulateRequest,
    db: Session = Depends(get_db),
):
    """
    Create a simulated incident for demo/testing.

    Example body:
    {
      "scenario": "unattended_baggage",
      "zone": "Gate_3"
    }

    Supported scenarios:
        unattended_baggage
        intrusion
        fire
        overcrowding
    """
    payload = make_simulation_payload(req)
    return await process_incident_payload(payload, db)


@app.post("/api/demo/reset")
async def reset_demo_data(
    db: Session = Depends(get_db),
):
    """
    Reset demo database.

    WARNING:
        This deletes all incidents and occupancy records.
        Use only for hackathon/demo.
    """
    db.query(ZoneOccupancyRecord).delete()
    db.query(Incident).delete()
    db.commit()

    NOTIFICATION_LOG.clear()

    await manager.broadcast(
        {
            "event": "demo_reset",
            "data": {
                "message": "Demo data cleared.",
                "time": utcnow_iso(),
            },
        }
    )

    return {
        "status": "reset",
        "time": utcnow_iso(),
    }
