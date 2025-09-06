# predictor_baccarat.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any, List

Side = Literal["player", "banker", "tie"]  # tay con / nhà cái / hòa


@dataclass
class PredictResult:
    side: Side
    probs: Dict[Side, float]
    reason: str = "freq-last-50"


class BacPredictor:
    """
    Predictor tối thiểu, an toàn cho Railway.
    - Nếu bạn có thuật toán thật, chỉ cần thay phần predict() & update().
    - GIỮ tên class là BacPredictor để main.py import được.
    """
    def __init__(self) -> None:
        self.history: List[Side] = []

    def reset(self) -> None:
        self.history = []

    def update(self, last: Side | None = None) -> None:
        if last is not None:
            self.history.append(last)
            # tránh phình bộ nhớ
            if len(self.history) > 5000:
                self.history = self.history[-2000:]

    def predict(self) -> PredictResult:
        # Baseline: đếm tần suất 50 ván gần nhất + prior nho nhỏ
        base: Dict[Side, float] = {"player": 1.0, "banker": 1.0, "tie": 0.2}
        for s in self.history[-50:]:
            base[s] = base.get(s, 0.0) + 1.0

        total = sum(base.values())
        probs = {k: (v / total) for k, v in base.items()}
        side: Side = max(probs, key=probs.get)  # type: ignore
        return PredictResult(side=side, probs=probs, reason="frequency-50")
