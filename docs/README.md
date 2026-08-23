# Agentic Campus Safety Intelligence Platform

An AI-powered campus surveillance system that detects overcrowding, unattended baggage,
restricted-zone intrusion, and fire/smoke in real time — then uses an LLM agent to
prioritize, correlate, and explain incidents, with a human always able to acknowledge
or override.

Built for the GGSIPU East Delhi Campus hackathon (General Administration Branch problem
statement), combining computer vision perception with an agentic reasoning layer.

## One-line pitch

Deterministic computer vision does the perception (the safety-critical part); a bounded
LLM agent reasons over the confirmed facts to prioritize, correlate, and explain — with a
human always in the loop.

## Team

| Role | Owns |
|---|---|
| Perception engineer | YOLO detection, tracking, rule engine |
| Backend + agent engineer | FastAPI, database, WebSocket, LLM agent |
| Frontend engineer | Live feed, alert list, campus map, occupancy gauges |
| Data + integration | Sample footage, zone definitions, docs, demo |

## Tech stack

Python, OpenCV, Ultralytics YOLOv8/11, FastAPI, SQLAlchemy, WebSocket, Anthropic API
(tool-use), HTML/JS or React.

## Quick start

See [`docs/SETUP.md`](docs/SETUP.md) for full local setup instructions.

```bash
# Backend
cd backend && pip install -r requirements.txt && python3 main.py

# Perception (separate terminal)
cd perception && pip install -r requirements.txt && python3 main.py

# Frontend (separate terminal)
cd frontend && open index.html  # or serve with a dev server
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design and data flow
- [`docs/API.md`](docs/API.md) — backend endpoint reference
- [`docs/SETUP.md`](docs/SETUP.md) — how to run everything locally
- [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) — the live walkthrough for judges
- [`docs/PROPERTIES_REFERENCE.md`](docs/PROPERTIES_REFERENCE.md) — every incident field explained
- [`docs/DATA_FLOW_COMPLETE.md`](docs/DATA_FLOW_COMPLETE.md) — full property lifecycle trace
- [`docs/OCCUPANCY_FEATURE.md`](docs/OCCUPANCY_FEATURE.md) — live zone occupancy tracking
- [`docs/BACKEND_SCHEMA_GUIDE.md`](docs/BACKEND_SCHEMA_GUIDE.md) — schemas/models/database explained
- [`docs/PERCEPTION_GUIDE.md`](docs/PERCEPTION_GUIDE.md) — perception module file-by-file guide

## What's in scope

Crowd/overcrowding detection, unattended baggage, restricted-zone intrusion, fire/smoke
detection, severity scoring, alert deduplication, incident correlation, evidence
snapshots, simulated notification routing, human-in-the-loop acknowledge/override.

## What's explicitly out of scope

Violence detection, real face recognition, multi-camera re-identification, real GIS,
real edge (Jetson) deployment — see `docs/ARCHITECTURE.md` for the reasoning.

## License

Built for hackathon purposes — GGSIPU, USAR / General Administration Branch.
