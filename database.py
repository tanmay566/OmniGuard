from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool
from .config import DATABASE_URL
from .models import Base


#engine start
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    poolclass = StaticPool
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        poolclass=poolclass,
        echo=False
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

#session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#functions-
def create_all_tables():
    Base.metadata.create_all(bind=engine)
    print(f"✓ Database tables created")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db():
    create_all_tables() #on startup


async def close_db():
    engine.dispose() #on shutdown
    print("✓ Database connection closed")
