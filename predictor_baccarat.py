# predictor.py
# Tool soi cầu Tài/Xỉu — ensemble nhiều mô hình + học trọng số online
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple, Callable, Optional

Label = str  # "T" (Tài) | "X" (Xỉu)

EPS = 1e-9

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _normalize(px: Tuple[float, float]) -> Tuple[float, float]:
    t, x = px
    s = (t + x) or 1.0
    return (t / s, x / s)

def _logloss(p: float) -> float:
    # log-loss cho xác suất của nhãn thật
    p = _clip01(p)
    p = max(EPS, min(1 - EPS, p))
    return -math.log(p)

@dataclass
class StatPack:
    total: int = 0
    wins_ensemble: int = 0
    wins_last20: int = 0  # số win trong 20 ván gần nhất (ensemble)
    last_guess: Optional[Label] = None
    last_conf: float = 0.0

class EnsemblePredictor:
    """
    - Mô hình thành phần:
        * Markov k = 2,3,4
        * N-gram (giống Markov nhưng quét nhiều bối cảnh)
        * Dirichlet (đếm tần suất cửa T/X + prior)
        * EWMA (exponential weighted moving average)
        * Streak bias (đảo chiều khi chuỗi quá dài)
    - Trộn bằng trọng số học online (multiplicative weights theo log-loss).
    - Đầu ra: xác suất, dự đoán, độ tự tin, giải thích top-3 mô hình đóng góp.
    """
    def __init__(self):
        self.history: List[Label] = []
        # trọng số khởi tạo cho từng model
        self.models: List[Tuple[str, Callable[[List[Label]], Tuple[float, float]], str]] = []
        self.w: Dict[str, float] = {}
        self.stat = StatPack()
        self._init_models()

    # ---------------- Core API ----------------
    def update(self, result: Label):
        """Thêm kết quả thật (T/X) và học online trọng số."""
        if result not in ("T", "X"):
            return
        # học online: đánh giá các model trên lịch sử *trước* khi thêm result
        if self.history:
            tps = self._all_model_probs(self.history)
            # loss theo nhãn thật
            for name, (pT, pX, _) in tps.items():
                prob_true = pT if result == "T" else pX
                loss = _logloss(prob_true)
                # multiplicative weights: w *= exp(-eta*loss)
                eta = 0.35
                self.w[name] *= math.exp(-eta * loss)

        self.history.append(result)

    def undo(self):
        if self.history:
            self.history.pop()

    def reset(self):
        self.history.clear()
        self.stat = StatPack()
        self._init_models()

    def predict(self) -> Dict:
        """Trả về: probs, guess, confidence, reasons(top-3)."""
        # tổng hợp
        mix_T, mix_X, contrib = self._ensemble_probs(self.history)
        guess = "T" if mix_T >= mix_X else "X"
        conf = abs(mix_T - mix_X)
        self.stat.last_guess = guess
        self.stat.last_conf = conf

        # giải thích top-3 đóng góp
        contrib_sorted = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:3]
        reasons = []
        for name, val in contrib_sorted:
            reasons.append(f"{name}: {val:.2f}")

        return {
            "probs": {"T": mix_T, "X": mix_X},
            "guess": guess,
            "confidence": conf,
            "reasons": reasons,
        }

    def stats(self) -> Dict:
        N = len(self.history)
        t = self.history.count("T")
        x = N - t
        # win-rate ensemble
        wr20 = self._win_rate_last_k(20)
        return {
            "total": N,
            "tai": t,
            "xiu": x,
            "p_tai": (t / N * 100) if N else 0.0,
            "p_xiu": (x / N * 100) if N else 0.0,
            "wr20": wr20 * 100,
            "last_guess": self.stat.last_guess,
            "last_conf": self.stat.last_conf,
        }

    def streaks(self, last_n: int = 6) -> List[Tuple[str, int]]:
        """Trả về last_n chuỗi T/X gần nhất."""
        if not self.history:
            return []
        res: List[Tuple[str, int]] = []
        cur = self.history[0]
        cnt = 1
        for h in self.history[1:]:
            if h == cur:
                cnt += 1
            else:
                res.append((cur, cnt))
                cur, cnt = h, 1
        res.append((cur, cnt))
        return res[-last_n:]

    # ---------------- Models & Ensemble ----------------
    def _init_models(self):
        self.models = [
            ("Markov k=2", lambda H: self._markov(H, 2), "M2"),
            ("Markov k=3", lambda H: self._markov(H, 3), "M3"),
            ("Markov k=4", lambda H: self._markov(H, 4), "M4"),
            ("Ngram mix", self._ngram_mix, "NG"),
            ("Dirichlet", self._dirichlet, "DIR"),
            ("EWMA d=0.94", lambda H: self._ewma(H, 0.94), "EW94"),
            ("EWMA d=0.97", lambda H: self._ewma(H, 0.97), "EW97"),
            ("Streak bias", self._streak_bias, "STK"),
        ]
        self.w = {name: 1.0 for name, _, _ in self.models}

    def _all_model_probs(self, H: List[Label]) -> Dict[str, Tuple[float, float, float]]:
        """Trả pT,pX,weight cho từng model (không chuẩn hoá)."""
        out: Dict[str, Tuple[float, float, float]] = {}
        for name, fn, _abbr in self.models:
            pT, pX = fn(H)
            out[name] = (pT, pX, self.w[name])
        return out

    def _ensemble_probs(self, H: List[Label]) -> Tuple[float, float, Dict[str, float]]:
        # gom xác suất theo trọng số
        sumT = sumX = 0.0
        contrib: Dict[str, float] = {}
        allp = self._all_model_probs(H)
        # chuẩn hoá trọng số
        wsum = sum(self.w.values()) or 1.0
        for name, (pT, pX, w) in allp.items():
            ww = w / wsum
            sumT += ww * pT
            sumX += ww * pX
            contrib[name] = ww * (pT if pT >= pX else pX)  # đóng góp theo xác suất cao hơn
        sumT, sumX = _normalize((sumT, sumX))
        return sumT, sumX, contrib

    # ---------- từng model ----------
    def _markov(self, H: List[Label], k: int) -> Tuple[float, float]:
        if len(H) < k:
            return (0.5, 0.5)
        counts = defaultdict(lambda: {"T": 0, "X": 0})
        for i in range(len(H) - k):
            ctx = tuple(H[i : i + k])
            nxt = H[i + k]
            counts[ctx][nxt] += 1
        ctx = tuple(H[-k:])
        c = counts.get(ctx)
        if not c:
            # không có pattern này, dùng Dirichlet nhẹ
            return self._dirichlet(H, alpha=1.2, window=20)
        pT = (c["T"] + 1.0) / (c["T"] + c["X"] + 2.0)
        pX = 1.0 - pT
        return _normalize((pT, pX))

    def _ngram_mix(self, H: List[Label]) -> Tuple[float, float]:
        # trộn markov k=2..4
        pts = []
        for k in (2, 3, 4):
            pts.append(self._markov(H, k))
        pT = sum(p[0] for p in pts) / len(pts)
        pX = 1.0 - pT
        return _normalize((pT, pX))

    def _dirichlet(self, H: List[Label], alpha: float = 1.5, window: int = 40) -> Tuple[float, float]:
        recent = H[-window:] if window > 0 else H[:]
        t = recent.count("T")
        x = len(recent) - t
        pT = (t + alpha) / (t + x + 2 * alpha)
        pX = 1.0 - pT
        return _normalize((pT, pX))

    def _ewma(self, H: List[Label], decay: float) -> Tuple[float, float]:
        if not H:
            return (0.5, 0.5)
        # encode: T=1, X=0
        y = 0.5
        for lab in H:
            y = decay * y + (1 - decay) * (1.0 if lab == "T" else 0.0)
        pT = _clip01(y)
        pX = 1.0 - pT
        return _normalize((pT, pX))

    def _streak_bias(self, H: List[Label]) -> Tuple[float, float]:
        if not H:
            return (0.5, 0.5)
        # tính chuỗi cuối
        cur = H[-1]
        k = 1
        for i in range(len(H) - 2, -1, -1):
            if H[i] == cur:
                k += 1
            else:
                break
        # nếu chuỗi dài, thiên về đảo chiều; ngắn thì theo chuỗi
        if k >= 4:
            if cur == "T":
                pT = 0.45
            else:
                pT = 0.55
        elif k == 3:
            pT = 0.5 if cur == "T" else 0.5
        else:
            if cur == "T":
                pT = 0.55
            else:
                pT = 0.45
        pX = 1.0 - pT
        return _normalize((pT, pX))

    # ---------- tiện ích ----------
    def _win_rate_last_k(self, k: int) -> float:
        if len(self.history) < 2:
            return 0.0
        wins = 0
        total = 0
        # duyệt từng điểm, dự đoán cho (0..n-2), so với vị trí kế tiếp
        start = max(0, len(self.history) - k - 1)
        for i in range(start, len(self.history) - 1):
            H = self.history[: i + 1]
            pT, pX, _ = self._ensemble_probs(H)
            guess = "T" if pT >= pX else "X"
            if guess == self.history[i + 1]:
                wins += 1
            total += 1
        return wins / total if total else 0.0
