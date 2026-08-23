import os
from dotenv import load_dotenv

load_dotenv() 

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./campus_safety.db")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))

CROWD_THRESHOLD = int(os.getenv("CROWD_THRESHOLD", "15"))
BAGGAGE_DWELL_SECONDS = int(os.getenv("BAGGAGE_DWELL_SECONDS", "30"))
DEDUP_WINDOW_SECONDS = int(os.getenv("DEDUP_WINDOW_SECONDS", "60"))
CORRELATION_WINDOW_MINUTES = int(os.getenv("CORRELATION_WINDOW_MINUTES", "5"))
