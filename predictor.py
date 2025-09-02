# predictor.py
from __future__ import annotations
import csv
import hashlib
import io
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

Label = str  # "Tài" | "Xỉu" | "Triple"


# --------- utils ----------
def _label_from_sum(s: int) -> Label:
    return "Xỉu" if s <= 10 else "Tài"


def _label_from_dice(d: List[int]) -> Label:
    if len(d) == 3 and d[0] == d[1] == d[2]:
        return "Triple"
    return _label_from_sum(sum(d))


def _safe_probs(p: Dict[Label, float]) -> Dict[Label, float]:
    total = sum(max(0.0, v) for v in p.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in p.items()}


def _hash_feature(md5: str) -> float:
    """Đặc trưng ổn định từ chuỗi MD5 (không phải giải mã)."""
    if not md5:
        return 0.0
    try:
        h = hashlib.md5(md5.encode("utf-8")).hexdigest()
    except Exception:
        h = md5
    try:
        v = int(h[-2:], 16) / 255.0
    except Exception:
        v = 0.0
    return v


# --------- dataclasses ----------
@dataclass
class Record:
    ts: float
    md5: str
    dice: List[int]
    guess: Label
    real: Label
    win: bool
    algo: str


@dataclass
class Stats:
    total: int = 0
    wins: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total) if self.total else 0.0

    def add(self, win: bool):
        self.total += 1
        if win:
            self.wins += 1


# --------- core AI ----------
class TaiXiuAI:
    """
    Lõi dự đoán Tai Xỉu (demo):
    - Nhiều thuật toán xác suất label (dice): freq, md5_bias, momentum, dirichlet, mix
    - Nhiều chiến lược dự đoán combo (bộ 3 xúc xắc): center, edges, random, mix
    - ĐÃ KHOÁ: luôn dùng algo_mix + combo_mix
    - ĐÃ ÉP: combo khớp với label (Tài >=11, Xỉu <=10, Triple = [k,k,k])
    """

    def __init__(self):
        self.records: List[Record] = []
        self.per_algo: Dict[str, Stats] = {}
        self.combo_stats: Dict[str, Stats] = {}
        self.window_default = 20

        # --- thuật toán label ---
        self.algorithms: List[Tuple[str, Callable]] = [
            ("algo_freq", self._algo_freq),
            ("algo_md5_bias", self._algo_md5_bias),
            ("algo_recent_momentum", self._algo_recent_momentum),
            ("algo_dirichlet", self._algo_dirichlet),
            ("algo_mix", self._algo_mix),  # bản gộp (để cuối)
        ]

        # --- chiến lược combo ---
        self.combo_strategies: List[Tuple[str, Callable]] = [
            ("combo_center", self._combo_center),
            ("combo_edges", self._combo_edges),
            ("combo_random", self._combo_random),
            ("combo_mix", self._combo_mix),  # bản gộp (để cuối)
        ]

        # --- khoá về MIX ---
        self.fixed_algo_name = "algo_mix"
        self.fixed_combo_name = "combo_mix"
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name)

        # cache xác suất tổng
        self._sum_cache: Dict[int, float] = {}

    # ---------- public API ----------
    def next_prediction(
        self,
        md5_value: str = "",
        alpha: int = 1,
        window_n: Optional[int] = None,
        temperature: float = 0.85,
    ) -> Dict:
        # ép index về MIX (phòng khi có ai gọi đổi)
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name, self.current_algo_index)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name, self.current_combo_idx)

        # 1) xác suất label
        algo_name, algo_fn = self.algorithms[self.current_algo_index]
        probs = algo_fn(md5_value=md5_value, alpha=alpha, window_n=window_n, temperature=temperature)
        probs = _safe_probs(probs)
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        prob_label = probs[label]

        # 2) combo thô (từ chiến lược combo)
        combo_name, combo_fn = self.combo_strategies[self.current_combo_idx]
        raw_combo, _ = combo_fn(md5_value=md5_value, temperature=temperature)

        # 3) ÉP combo khớp label
        combo, p_combo = self._force_combo_match_label(label, raw_combo)

        return {
            "label": label,
            "prob_label": prob_label,
            "probs": probs,
            "combo": combo,
            "sum": sum(combo),
            "p_combo": p_combo,
            "algo": algo_name,
            "combo_algo": combo_name,
        }

    def update_with_real(self, md5: str, real_label: Label, dice: Optional[List[int]]) -> Dict:
        """Ghi nhận 1 ván đã có kết quả (để học thống kê)."""
        if dice is None:
            dice = [0, 0, 0]
        last_pred = self.next_prediction(md5_value=md5)
        guess = last_pred["label"]
        win = (guess == real_label)

        rec = Record(
            ts=time.time(),
            md5=md5 or "",
            dice=dice,
            guess=guess,
            real=real_label,
            win=win,
            algo=self.algorithms[self.current_algo_index][0],
        )
        self.records.append(rec)

        self._stat_inc(self.per_algo, rec.algo, rec.win)
        self._stat_inc(self.combo_stats, self.combo_strategies[self.current_combo_idx][0], rec.win)

        return {
            "md5": rec.md5,
            "dice": rec.dice,
            "guess": rec.guess,
            "real": rec.real,
            "win": rec.win,
            "algo": rec.algo,
        }

    def summary(self) -> Dict:
        total = len(self.records)
        wins = sum(1 for r in self.records if r.win)
        last = self.records[-1] if self.records else None
        return {
            "total_rounds": total,
            "wins": wins,
            "win_rate": (wins / total) if total else 0.0,
            "current_algo": self.algorithms[self.current_algo_index][0],
            "current_combo_algo": self.combo_strategies[self.current_combo_idx][0],
            "last_record": {
                "md5": last.md5,
                "dice": last.dice,
                "guess": last.guess,
                "real": last.real,
                "win": last.win,
                "algo": last.algo,
            } if last else None,
            "per_algo": {k: {"total": v.total, "wins": v.wins, "win_rate": v.win_rate} for k, v in self.per_algo.items()},
        }

    def export_algo_stats_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["algo", "total", "wins", "win_rate"])
        for k, v in self.per_algo.items():
            w.writerow([k, v.total, v.wins, f"{v.win_rate:.4f}"])
        return buf.getvalue()

    def export_combo_stats_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["combo_algo", "total", "wins", "win_rate"])
        for k, v in self.combo_stats.items():
            w.writerow([k, v.total, v.wins, f"{v.win_rate:.4f}"])
        return buf.getvalue()

    def reset(self):
        self.records.clear()
        self.per_algo.clear()
        self.combo_stats.clear()
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name)

    def cleanup(self):
        return

    # ---------- helpers ----------
    def _index_of(self, items: List[Tuple[str, Callable]], name: str, default_idx: int = 0) -> int:
        for i, (n, _) in enumerate(items):
            if n == name:
                return i
        return default_idx

    def _stat_inc(self, table: Dict[str, Stats], key: str, win: bool):
        if key not in table:
            table[key] = Stats()
        table[key].add(win)

    # ===== thuật toán xác suất label =====
    def _recent_counts(self, window_n: Optional[int]) -> Dict[Label, int]:
        n = window_n or self.window_default
        last = self.records[-n:] if n > 0 else self.records[:]
        c = {"Tài": 0, "Xỉu": 0, "Triple": 0}
        for r in last:
            c[r.real] = c.get(r.real, 0) + 1
        return c

    def _algo_freq(self, **kw) -> Dict[Label, float]:
        alpha = max(1, int(kw.get("alpha", 1)))
        counts = self._recent_counts(kw.get("window_n"))
        base = {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
        p = {k: base[k] * alpha + counts.get(k, 0) for k in base}
        return _safe_probs(p)

    def _algo_md5_bias(self, **kw) -> Dict[Label, float]:
        f = _hash_feature(kw.get("md5_value", ""))
        p_t = 0.45 + 0.15 * f
        p_x = 0.45 + 0.15 * (1.0 - f)
        p_tri = 0.10 * (0.5 - abs(f - 0.5))  # nhỏ
        return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": p_tri})

    def _algo_recent_momentum(self, **kw) -> Dict[Label, float]:
        last = self.records[-8:]
        bias = sum(1 if r.real == "Tài" else -1 if r.real == "Xỉu" else 0 for r in last)
        p_t = 0.48 + 0.02 * bias
        p_x = 0.48 - 0.02 * bias
        p_tri = 0.04
        return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": p_tri})

    def _algo_dirichlet(self, **kw) -> Dict[Label, float]:
        alpha = max(1, int(kw.get("alpha", 1)))
        c = self._recent_counts(kw.get("window_n"))
        prior = {"Tài": alpha, "Xỉu": alpha, "Triple": max(1, alpha // 10)}
        p = {k: prior[k] + c.get(k, 0) for k in prior}
        return _safe_probs(p)

    def _algo_mix(self, **kw) -> Dict[Label, float]:
        others = [fn for name, fn in self.algorithms if name != "algo_mix"]
        if not others:
            return {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
        agg = {"Tài": 0.0, "Xỉu": 0.0, "Triple": 0.0}
        for fn in others:
            p = fn(**kw)
            for k, v in p.items():
                agg[k] += v
        for k in agg:
            agg[k] /= len(others)
        return _safe_probs(agg)

    # ===== combo strategies =====
    def _combo_center(self, **kw) -> Tuple[List[int], float]:
        # thiên về tổng trung tâm (10 hoặc 11)
        s = random.choice([10, 11])
        d = self._sample_sum(s)
        return d, self._sum_prob(s)

    def _combo_edges(self, **kw) -> Tuple[List[int], float]:
        s = random.choice([3, 4, 17, 18])
        d = self._sample_sum(s)
        return d, self._sum_prob(s)

    def _combo_random(self, **kw) -> Tuple[List[int], float]:
        d = [random.randint(1, 6) for _ in range(3)]
        return d, self._sum_prob(sum(d))

    def _combo_mix(self, **kw) -> Tuple[List[int], float]:
        cands: List[Tuple[List[int], float]] = []
        for name, fn in self.combo_strategies:
            if name == "combo_mix":
                continue
            d, p = fn(**kw)
            cands.append((d, p))
        if not cands:
            return self._combo_random(**kw)
        d, p = max(cands, key=lambda x: x[1])
        return d, p

    # ===== ép combo khớp label =====
    def _weighted_choice_by_sum(self, sums: List[int]) -> int:
        weights = [self._sum_prob(s) for s in sums]
        total = sum(weights) or 1.0
        r = random.random() * total
        acc = 0.0
        for s, w in zip(sums, weights):
            acc += w
            if r <= acc:
                return s
        return sums[-1]

    def _force_combo_match_label(self, label: str, combo: List[int]) -> Tuple[List[int], float]:
        if label == "Triple":
            k = random.randint(1, 6)
            return [k, k, k], 6.0 / 216.0  # xác suất nhóm triple (bất kỳ k) ~ 2.777%

        s = sum(combo)
        if label == "Tài" and s >= 11:
            return combo, self._sum_prob(s)
        if label == "Xỉu" and s <= 10:
            return combo, self._sum_prob(s)

        # lệch → tạo lại combo phù hợp
        if label == "Tài":
            target_sums = list(range(11, 19))
        else:
            target_sums = list(range(3, 11))
        s_new = self._weighted_choice_by_sum(target_sums)
        new_combo = self._sample_sum(s_new)
        return new_combo, self._sum_prob(s_new)

    # ===== xác suất tổng & dựng combo theo tổng =====
    def _sum_prob(self, s: int) -> float:
        if not self._sum_cache:
            freq = {k: 0 for k in range(3, 19)}
            for a in range(1, 7):
                for b in range(1, 7):
                    for c in range(1, 7):
                        freq[a + b + c] += 1
            total = 6 ** 3
            for k, v in freq.items():
                self._sum_cache[k] = v / total
        return float(self._sum_cache.get(s, 0.0))

    def _sample_sum(self, s: int) -> List[int]:
        s = max(3, min(18, s))
        for _ in range(30):
            a = random.randint(1, 6)
            b = random.randint(1, 6)
            c = s - a - b
            if 1 <= c <= 6:
                return [a, b, c]
        return [random.randint(1, 6) for _ in range(3)]
