from sqlmodel import SQLModel, Field
from datetime import datetime


class Round(SQLModel, table=True):
id: int | None = Field(default=None, primary_key=True)
ts: datetime = Field(default_factory=datetime.utcnow, index=True)
md5: str | None = Field(default=None, index=True)
d1: int
d2: int
d3: int
total: int = 0
label: str = Field(index=True) # 'TAI' | 'XIU'
is_triple: bool = False
source: str = "manual"


def compute(self):
self.total = self.d1 + self.d2 + self.d3
self.is_triple = (self.d1 == self.d2 == self.d3)
self.label = 'TAI' if 11 <= self.total <= 17 else 'XIU'


class Prediction(SQLModel, table=True):
id: int | None = Field(default=None, primary_key=True)
md5: str
label_pred: str
p_tai: float
p_xiu: float
algo: str = "markov"
version: str = "1.0.0"
ts: datetime = Field(default_factory=datetime.utcnow, index=True)
# Resolution fields (link to actual outcome)
round_id: int | None = None
actual_label: str | None = None # 'TAI' | 'XIU'
correct: bool | None = None
resolved_ts: datetime | None = None


class PatternEpisode(SQLModel, table=True):
id: int | None = Field(default=None, primary_key=True)
start_round_id: int
end_round_id: int
pattern_type: str # 'RUN' | 'ALT' | 'BLOCK'
pattern_str: str
length: int
confidence: float = 0.0
ts: datetime = Field(default_factory=datetime.utcnow, index=True)
