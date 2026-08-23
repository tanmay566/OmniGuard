# Architecture

## One-line pitch

A campus surveillance system where deterministic computer vision detects incidents, and
a bounded LLM agent reasons over those confirmed facts to prioritize, correlate, and
explain — with a human always able to acknowledge or override.

## Why this design

Perception (the safety-critical part) stays deterministic and explainable — the LLM
never sees raw video, only structured facts the CV layer already confirmed. This means
the agent can't hallucinate a threat that wasn't actually detected. It also means the
system degrades gracefully: if the LLM API is slow or down, the deterministic fallback
keeps incidents flowing.

## Full pipeline

```
Video input (webcam / sample CCTV clips)
   -> Frame preprocessing (OpenCV: resize, sample fps)
   -> AI detection (YOLOv8/11, pretrained: person/backpack/suitcase + fire/smoke classifier)
   -> Object tracking (centroid tracker: persistent IDs + dwell time)
   -> Rule engine (deterministic thresholds, dwell timers, zone checks)
   -> Structured incident JSON -> sent to backend
   -> Agent layer (LLM w/ tool-use, bounded, schema-forced output):
        - assigns severity
        - checks/correlates nearby recent incidents
        - drafts control-room notification
        - produces explainable reasoning
        - FALLBACK: deterministic severity table if LLM call times out/fails
   -> FastAPI backend: logs to DB, pushes via WebSocket, exposes ack/override endpoints
   -> Dashboard: live feed with boxes, prioritized alert list, campus map, occupancy gauges
```

## Components

### Perception module
Runs as a single continuous Python process reading video frames locally. No network
dependency except the final incident POST call. See `docs/PERCEPTION_GUIDE.md`.

### Backend
FastAPI app with SQLAlchemy (Postgres-ready), exposing incident ingestion, operator action
endpoints, and a WebSocket for live dashboard updates. See `docs/API.md` and
`docs/BACKEND_SCHEMA_GUIDE.md`.

### Agent layer
Not a separate service — a Python function inside the backend that calls an LLM API
with tool-use, bounded to 2-3 tool-call turns, forced into a strict output schema, with
a deterministic fallback on any failure.

### Dashboard
Live video panel with bounding boxes, prioritized/color-coded alert list, static campus
map with zone pins, per-zone occupancy gauges, "why flagged" explainability view.

## Deployment view

Everything runs on 1-2 laptops on the same local network for the demo:

- **Perception module** — local Python process, no separate deployment
- **Backend + database** — one FastAPI process, SQLite file alongside it
- **Agent layer** — function inside the backend, makes outbound calls to the LLM
  provider's cloud API
- **Dashboard** — static HTML/JS or React dev server, connects via WebSocket

In production, this is designed to extend to Jetson-class edge devices per camera,
sending only structured incidents (not raw video) across the network — a genuine
privacy-preserving property worth stating in the report, even though we don't build it
for the demo.

## Key design decisions

1. **Agent never sees raw video** — only structured facts already confirmed by CV. This
   is the single most important boundary in the system.
2. **DB write happens before the agent call, always** — an incident is never lost even
   if the LLM call fails or times out.
3. **Deterministic fallback for severity** — if the agent is unavailable, a plain
   lookup table (fire=critical, intrusion=high, baggage=medium, overcrowding=medium)
   keeps the system functional.
4. **Occupancy is separate from incidents** — incidents are discrete alerts, occupancy
   is continuous state. Mixing them makes both confusing (see
   `docs/OCCUPANCY_FEATURE.md`).
5. **Human-in-the-loop, not autonomous action** — the agent recommends, an operator
   acknowledges, overrides, or dismisses. Every action is logged with who and when.

## What we explicitly did not build, and why

| Feature | Why cut |
|---|---|
| Violence/scuffle detection | UCF Crime dataset too unreliable in hackathon time |
| Real face recognition | Out of scope for this PS; avoids privacy/consent complexity |
| Multi-camera re-identification | Too complex for the timeframe |
| Real GIS integration | Static map image + pins gets most of the visual payoff cheaply |
| Real edge deployment (Jetson) | Architecture supports it; not needed to prove the concept |
| ERP/LMS integration | Not relevant to this problem statement |
