# app/db/base.py
from sqlmodel import SQLModel, create_engine, Session
from app.config import settings
import os

# Tạo thư mục data khi dùng SQLite file
if settings.db_dsn.startswith("sqlite"):
    os.makedirs("data", exist_ok=True)

engine = create_engine(settings.db_dsn, echo=False)

def init_db():
    from app.db import models  # đăng ký models
    SQLModel.metadata.create_all(engine)

# ✅ ĐÚNG: hàm generator dùng yield
def get_session():
    with Session(engine) as session:
        yield session
