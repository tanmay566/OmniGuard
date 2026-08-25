from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

try:
    from backend.database import Base
except ImportError:
    from database import Base


class Incident(Base):
    __tablename__ = "incidents"

    #from cv
    incident_id: Mapped[str] = mapped_column(String, primary_key=True)
    type: Mapped[str] = mapped_column(String)  
    zone: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[str] = mapped_column(String, index=True)
    tracked_object_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    dwell_time_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    person_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    detection_confidence: Mapped[float] = mapped_column(Float)

    #from agent
    severity: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reasoning_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    correlated_incident_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    recommended_action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notification_draft: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    #used by mods
    status: Mapped[str] = mapped_column(String, default="new")  
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True) 

    #metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    @property
    def count(self) -> Optional[int]:
        return self.person_count

    @count.setter
    def count(self, value: Optional[int]) -> None:
        self.person_count = value

    def __repr__(self):
        return f"<Incident {self.incident_id} {self.type} {self.severity}>"


class ZoneOccupancyRecord(Base):
    __tablename__ = "zone_occupancy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    zone: Mapped[str] = mapped_column(String, index=True)
    current_count: Mapped[int] = mapped_column(Integer)
    capacity: Mapped[int] = mapped_column(Integer)
    occupancy_percentage: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[str] = mapped_column(String, index=True)
    trend: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
