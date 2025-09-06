from sqlmodel import SQLModel, create_engine, Session
from app.config import settings
import os

if settings.db_dsn.startswith("sqlite"):
    os.makedirs("data", exist_ok=True)

engine = create_engine(settings.db_dsn, echo=False)

def init_db():
    from app.db import models  # noqa: F401
    SQLModel.metadata.create_all(engine)

# ✅ Dependency đúng chuẩn
def get_session():
    with Session(engine) as session:
        yield session
