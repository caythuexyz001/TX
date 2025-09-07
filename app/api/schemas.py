from pydantic import BaseModel, Field
from typing import Optional


class IngestIn(BaseModel):
d1: int = Field(ge=1, le=6)
d2: int = Field(ge=1, le=6)
d3: int = Field(ge=1, le=6)
md5: Optional[str] = None


class PredictOut(BaseModel):
prediction_id: int
label_suggest: str
p_tai: float
p_xiu: float
method: str


class StatsOut(BaseModel):
transition: list[list[float]]
counts: list[list[int]]
last_label: str | None
p_value_row: dict[str, float]
entropy: float


class PredictionItem(BaseModel):
id: int
md5: str
label_pred: str
p_tai: float
p_xiu: float
actual_label: str | None
correct: bool | None
ts: str
resolved_ts: str | None


class HistoryOut(BaseModel):
items: list[PredictionItem]


class SummaryOut(BaseModel):
wins: int
losses: int
total: int
winrate: float
