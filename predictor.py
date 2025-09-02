# predictor.py
from __future__ import annotations
import csv
import hashlib
import io
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable


Label = str  # "Tài" | "Xỉu" | "Triple"


def _label_from_sum(s: int) -> Label:
    if s <= 10:
        return "Xỉu"
    if s >= 11:
        return "Tài"
    return "Xỉu"


def _label_from_dice(d: List[int]) -> Label:
    if len(d) == 3 and d[0] == d[1] == d[2]:
        return "Triple"
    return _label_from_sum(sum(d))


def _safe_probs(p: Dict[Label, float]) -> Dict[Label, float]:
    total = sum(max(0.0, v) for v in p.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in p.items()}


def _hash_feature(md5: str) -> float:
    """Một đặc trưng nhỏ từ MD5 (ổn định, không giải mã)."""
    if not md5:
        return 0.0
    try:
        h = hashlib.md5(md5.encode("utf-8")).hexdigest()
    except Exception:
        h = md5
    # Lấy 2 ký tự cuối quy về [0..1)
    try:
        v = int(h[-2:], 16) / 255.0
    except Exception:
        v = 0.0
    return v


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

    def add(self, win: bool):
        self.total += 1
        if win:
            self.wins += 1

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total) if self.total else 0.0


class TaiXiuAI:
    """
    AI đơn giản nhưng đủ: có nhiều thuật toán con và bản 'mix' gộp trung bình.
    ĐÃ KHOÁ cố định dùng algo_mix + combo_mix, không tự đổi nữa.
    """

    def __init__(self):
        self.records: List[Record] = []
        self.per_algo: Dict[str, Stats] = {}
        self.combo_stats: Dict[str, Stats] = {}
        self.window_default = 20

        # ===== Thuật toán =====
        self.algorithms: List[Tuple[str, Callable]] = [
            ("algo_freq", self._algo_freq),
            ("algo_md5_bias", self._algo_md5_bias),
            ("algo_recent_momentum", self._algo_recent_momentum),
            ("algo_dirichlet", self._algo_dirichlet),
            ("algo_mix", self._algo_mix),  # <- bản gộp CUỐI CÙNG
        ]

        # ===== Combo (dự đoán bộ 3 xúc xắc) =====
        self.combo_strategies: List[Tuple[str, Callable]] = [
            ("combo_center", self._combo_center),
            ("combo_edges", self._combo_edges),
            ("combo_random", self._combo_random),
            ("combo_mix", self._combo_mix),  # <- bản gộp CUỐI CÙNG
        ]

        # ===== KHÓA về bản mix (yêu cầu của bạn) =====
        self.fixed_algo_name = "algo_mix"
        self.fixed_combo_name = "combo_mix"
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name)

    # ---------------- public API ----------------

    def next_prediction(
        self,
        md5_value: str = "",
        alpha: int = 1,
        window_n: Optional[int] = None,
        temperature: float = 0.85,
    ) -> Dict:
        # luôn ép về bản mix (nếu lỡ ai gọi set_algo khác)
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name, self.current_algo_index)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name, self.current_combo_idx)

        algo_name, algo_fn = self.algorithms[self.current_algo_index]
        probs = algo_fn(md5_value=md5_value, alpha=alpha, window_n=window_n, temperature=temperature)
        probs = _safe_probs(probs)

        # chọn label theo max prob
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        prob_label = probs[label]

        # combo
        combo_name, combo_fn = self.combo_strategies[self.current_combo_idx]
        combo, p_combo = combo_fn(md5_value=md5_value, temperature=temperature)

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
        if dice is None:
            dice = [0, 0, 0]
        # guess là dự đoán cuối cùng đã dùng algo hiện tại
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

        # thống kê thuật toán
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
        # luôn khoá lại về MIX
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name)

    def cleanup(self):
        # để tương thích với code cũ (không làm gì)
        return

    # ---------------- internal helpers ----------------
    def _index_of(self, items: List[Tuple[str, Callable]], name: str, default_idx: int = 0) -> int:
        for i, (n, _) in enumerate(items):
            if n == name:
                return i
        return default_idx

    def _stat_inc(self, table: Dict[str, Stats], key: str, win: bool):
        if key not in table:
            table[key] = Stats()
        table[key].add(win)

    # ----- thuật toán xác suất label -----
    def _recent_counts(self, window_n: Optional[int]) -> Dict[Label, int]:
        n = window_n or self.window_default
        last = self.records[-n:] if n > 0 else self.records[:]
        c = {"Tài": 0, "Xỉu": 0, "Triple": 0}
        for r in last:
            c[r.real] = c.get(r.real, 0) + 1
        return c

    def _algo_freq(self, **kw) -> Dict[Label, float]:
        """Tần suất nhãn gần đây + một ít làm trơn."""
        alpha = max(1, int(kw.get("alpha", 1)))
        counts = self._recent_counts(kw.get("window_n"))
        base = {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
        p = {k: base[k] * alpha + counts.get(k, 0) for k in base}
        return _safe_probs(p)

    def _algo_md5_bias(self, **kw) -> Dict[Label, float]:
        """Thiên hướng nhẹ theo đặc trưng MD5 (chỉ để đa dạng hoá)."""
        f = _hash_feature(kw.get("md5_value", ""))
        # lệch 2 phía theo feature
        p_t = 0.45 + 0.15 * f
        p_x = 0.45 + 0.15 * (1.0 - f)
        p_tri = 0.10 * (0.5 - abs(f - 0.5))  # rất nhỏ
        return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": p_tri})

    def _algo_recent_momentum(self, **kw) -> Dict[Label, float]:
        """Nếu chuỗi gần đây nghiêng về phía nào thì tăng xác suất phía đó."""
        last = self.records[-8:]
        bias = sum(1 if r.real == "Tài" else -1 if r.real == "Xỉu" else 0 for r in last)
        p_t = 0.48 + 0.02 * bias
        p_x = 0.48 - 0.02 * bias
        p_tri = 0.04
        return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": p_tri})

    def _algo_dirichlet(self, **kw) -> Dict[Label, float]:
        """Giống Dirichlet prior rất đơn giản."""
        alpha = max(1, int(kw.get("alpha", 1)))
        c = self._recent_counts(kw.get("window_n"))
        prior = {"Tài": alpha, "Xỉu": alpha, "Triple": max(1, alpha // 10)}
        p = {k: prior[k] + c.get(k, 0) for k in prior}
        return _safe_probs(p)

    def _algo_mix(self, **kw) -> Dict[Label, float]:
        """Trung bình các thuật toán khác (đÃ khoá dùng mặc định)."""
        names = [n for n, _ in self.algorithms if n != "algo_mix"]
        ps = []
        for n, fn in self.algorithms:
            if n == "algo_mix":
                continue
            ps.append(fn(**kw))
        if not ps:
            return {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
        agg = {"Tài": 0.0, "Xỉu": 0.0, "Triple": 0.0}
        for p in ps:
            for k, v in p.items():
                agg[k] += v
        for k in agg:
            agg[k] /= len(ps)
        return _safe_probs(agg)

    # ----- combo (dự đoán bộ xúc xắc) -----
    def _combo_center(self, **kw) -> Tuple[List[int], float]:
        """Ưu tiên tổng 10-11 (trung tâm phân phối)."""
        center_sums = [10, 11]
        s = random.choice(center_sums)
        d = self._sample_sum(s)
        return d, self._sum_prob(s)

    def _combo_edges(self, **kw) -> Tuple[List[int], float]:
        """Ưu tiên tổng thấp/cao hiếm."""
        edge_sums = [3, 4, 17, 18]
        s = random.choice(edge_sums)
        d = self._sample_sum(s)
        return d, self._sum_prob(s)

    def _combo_random(self, **kw) -> Tuple[List[int], float]:
        d = [random.randint(1, 6) for _ in range(3)]
        return d, self._sum_prob(sum(d))

    def _combo_mix(self, **kw) -> Tuple[List[int], float]:
        cand = []
        for name, fn in self.combo_strategies:
            if name == "combo_mix":
                continue
            d, p = fn(**kw)
            cand.append((d, p))
        if not cand:
            return self._combo_random(**kw)
        # chọn theo xác suất p lớn nhất
        d, p = max(cand, key=lambda x: x[1])
        return d, p

    # --- xác suất gần đúng của tổng (đếm brute-force) ---
    _sum_cache: Dict[int, float] = {}

    def _sum_prob(self, s: int) -> float:
        if not self._sum_cache:
            # precompute
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
        """Tạo bộ 3 xúc xắc có tổng là s (simple constructive)."""
        s = max(3, min(18, s))
        for _ in range(30):
            a = random.randint(1, 6)
            b = random.randint(1, 6)
            c = s - a - b
            if 1 <= c <= 6:
                return [a, b, c]
        # fallback
        return [random.randint(1, 6) for _ in range(3)]
