''' campus safety agent '''


import os 
import json
import re 
from typing import List, Dict, Any , Optional 
from datetime import datetime , timedelta 

from dotenv import load_dotenv
from anthropic import Anthropic 

from schemas import IncidentIn , AgentOutput 
from database import SessionLocal , Incident , ZoneOccupancyRecord
#env setup
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("")
AGENT_MODEL = os.getenv("", "")

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


#static zone
