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
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from agent import run_triage_agent
from database import Base, Incident, ZoneOccupancyRecord, engine, get_db
from schemas import (
    AckRequest,
    FalsePositiveRequest,
    IncidentIn,
    IncidentOut,
    OverrideRequest,
    ZoneOccupancy,
    ZoneOccupancyOut,
)

load_dotenv()

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Agentic Campus Safety Intelligence Platform",
    description="Deterministic perception + bounded LLM triage agent + human-in-the-loop.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_CANDIDATES = [
    Path(__file__).parent / "index.html",
    Path(__file__).parent.parent / "frontend" / "index.html",
    Path(__file__).parent.parent / "index.html",
    Path.cwd() / "index.html",
]

ALLOWED_SEVERITIES = {"critical", "high", "medium", "low"}
ALLOWED_ACTIONS = {"dispatch_security", "verify", "monitor", "none"}
ALLOWED_STATUSES = {"new", "acknowledged", "resolved", "false_positive"}
ACTIVE_STATUSES = {"new", "acknowledged"}

SEVERITY_PRIORITY = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

STATUS_PRIORITY = {
    "new": 0,
    "acknowledged": 1,
    "resolved": 2,
    "false_positive": 3,
}

SYSTEM_START_TIME = datetime.now(timezone.utc)

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

NOTIFICATION_LOG = deque(maxlen=200)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections = [ws for ws in self.active_connections if ws != websocket]

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    for candidate in FRONTEND_CANDIDATES:
        if candidate.exists():
            return HTMLResponse(candidate.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>OmniGuard Backend</h1><p>Frontend not found.</p>")

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def zone_display_name(zone: str) -> str:
    config = ZONE_CONFIG.get(zone, DEFAULT_ZONE_CONFIG)
    return config.get("display_name", zone.replace("_", " "))

def _incident_to_out(incident: Incident) -> IncidentOut:
    return IncidentOut(
        incident_id=incident.incident_id,
        type=incident.type,
        zone=incident.zone,
        severity=incident.severity,
        status=incident.status,
        tracked_object_id=incident.tracked_object_id,
        detection_confidence=incident.detection_confidence,
        dwell_time_seconds=incident.dwell_time_seconds,
        count=incident.count,
        timestamp=incident.timestamp,
        created_at=incident.created_at,
        updated_at=incident.updated_at,
        
    )

@app.post("/incidents", response_model=IncidentOut)
async def create_incident(payload: IncidentIn, db: Session = Depends(get_db)):
    incident = Incident(
        incident_id=payload.incident_id,
        type=payload.type,
        zone=payload.zone,
        severity=None,
        status="new",
        tracked_object_id=payload.tracked_object_id,
        detection_confidence=payload.detection_confidence,
        dwell_time_seconds=payload.dwell_time_seconds,
        count=payload.count,
        timestamp=payload.timestamp,
        
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db.add(incident)
    db.commit()
    db.refresh(incident)

    # run triage
    try:
        agent_output = run_triage_agent(payload)
    except Exception:
        agent_output = None

    if agent_output is not None:
        incident.severity = agent_output.severity
        
        db.commit()

    await manager.broadcast({
        "event": "incident_created",
        "incident": _incident_to_out(incident).model_dump(),
    })

    return _incident_to_out(incident)

@app.get("/incidents", response_model=List[IncidentOut])
async def list_incidents(
    db: Session = Depends(get_db),
    zone: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    query = db.query(Incident)
    if zone:
        query = query.filter(Incident.zone == zone)
    if status:
        query = query.filter(Incident.status == status)
    incidents = query.order_by(Incident.created_at.desc()).limit(limit).all()
    return [_incident_to_out(item) for item in incidents]

@app.get("/incidents/{incident_id}", response_model=IncidentOut)
async def get_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.incident_id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _incident_to_out(incident)

@app.post("/incidents/{incident_id}/ack")
async def acknowledge_incident(
    incident_id: str,
    request: AckRequest,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.incident_id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = "acknowledged"
    incident.updated_at = datetime.now(timezone.utc)
    db.commit()
    await manager.broadcast({
        "event": "incident_acknowledged",
        "incident_id": incident_id,
        "actor": request.actor,
    })
    return {"status": "ok"}

@app.post("/incidents/{incident_id}/override")
async def override_incident(
    incident_id: str,
    request: OverrideRequest,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.incident_id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.severity = request.severity
    incident.status = "acknowledged"
    incident.updated_at = datetime.now(timezone.utc)
    db.commit()
    await manager.broadcast({
        "event": "incident_overridden",
        "incident_id": incident_id,
        "severity": request.severity,
    })
    return {"status": "ok"}

@app.post("/incidents/{incident_id}/false_positive")
async def false_positive_incident(
    incident_id: str,
    request: FalsePositiveRequest,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.incident_id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.status = "false_positive"
    incident.updated_at = datetime.now(timezone.utc)
    db.commit()
    await manager.broadcast({
        "event": "incident_false_positive",
        "incident_id": incident_id,
        "reason": request.reason,
    })
    return {"status": "ok"}

@app.get("/zones")
async def list_zones() -> Dict[str, Dict[str, Any]]:
    return ZONE_CONFIG

@app.get("/occupancy", response_model=List[ZoneOccupancyOut])
async def get_occupancy(db: Session = Depends(get_db), limit: int = Query(20, ge=1, le=200)):
    rows = (
        db.query(ZoneOccupancyRecord)
        .order_by(ZoneOccupancyRecord.updated_at.desc())  # ✅ Changed from created_at
        .limit(limit)
        .all()
    )
    # ✅ Removed .to_dict(), Pydantic V2 handles it via from_attributes=True
    return [ZoneOccupancyOut.model_validate(row) for row in rows]

@app.post("/occupancy")
async def create_occupancy(payload: ZoneOccupancy, db: Session = Depends(get_db)):
    record = ZoneOccupancyRecord(
        zone=payload.zone,
        current_count=payload.current_count,
        capacity=payload.capacity,
        occupancy_percentage=payload.occupancy_percentage, # ✅ Added
        timestamp=payload.timestamp, # ✅ Added
        trend=payload.trend, # ✅ Added
        updated_at=datetime.now(timezone.utc), # ✅ Changed from created_at
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    
    await manager.broadcast({
        "event": "occupancy_update",
        "zone": payload.zone,
        "record": ZoneOccupancyOut.model_validate(record).model_dump(), # ✅ Fixed to_dict()
    })
    return {"status": "ok"}
if __name__ == "__main__":
    import uvicorn
    # This actually starts the localhost server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)