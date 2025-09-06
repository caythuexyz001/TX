from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    db_dsn: str = os.getenv("DB_DSN", "sqlite:///./data/tai_xiu.db")
    window: int = int(os.getenv("WINDOW", 50))
    alpha_warn: float = float(os.getenv("ALPHA_WARN", 0.10))
    theta: float = float(os.getenv("THETA", 0.10))
    api_key: str | None = os.getenv("API_KEY")

settings = Settings()