# predictor.py — bộ ~50 thuật toán ổn định + pending chuẩn B1→B2
from __future__ import annotations
import csv, hashlib, io, random, time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

Label = str  # "Tài" | "Xỉu" | "Triple"

# ---------- helpers ----------
def _safe_probs(p: Dict[Label, float]) -> Dict[Label, float]:
    total = sum(max(0.0, v) for v in p.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in p.items()}

def _hash_feature(md5: str, byte_idx: int = -1) -> float:
    if not md5:
        return 0.0
    h = hashlib.md5(md5.encode("utf-8")).hexdigest()
    try:
        if byte_idx is None or byte_idx < 0:
            return int(h[-2:], 16) / 255.0
        off = (byte_idx % 16) * 2
        return int(h[off : off + 2], 16) / 255.0
    except Exception:
        return 0.0

# ---------- data ----------
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


# ---------- core ----------
class TaiXiuAI:
    """
    - ~50 thuật toán label “classic” (Dirichlet/Momentum/Markov/EWMA/Streak/Parity/md5/noise)
    - ~15 thuật toán combo (center / band / edges / md5 / random)
    - Luôn ensemble bằng 'algo_mix' + 'combo_mix' (đã khóa)
    - Lưu 'pending' ở B1 → B2 chấm đúng theo guess đã lưu
    """

    def __init__(self):
        self.records: List[Record] = []
        self.per_algo: Dict[str, Stats] = {}
        self.combo_stats: Dict[str, Stats] = {}

        # pending: md5 -> {guess, probs, combo, ts, algo}
        self.pending: Dict[str, Dict] = {}

        self.window_default = 20
        self._sum_cache: Dict[int, float] = {}

        self.algorithms: List[Tuple[str, Callable]] = []
        self.combo_strategies: List[Tuple[str, Callable]] = []

        # ===== register LABEL (≈51 cái) =====
        self._register_dirichlet_classic()  # 7
        self._register_momentum_classic()   # 11
        self._register_markov1_classic()    # 5
        self._register_ewma_classic()       # 4
        self._register_streak_classic()     # 5
        self._register_parity_classic()     # 3
        self._register_md5_bias_classic()   # 7
        self._register_noise_classic()      # 9
        self.algorithms.append(("algo_mix", self._algo_mix))  # 1 mix
        # Tổng xấp xỉ: 51+1 = 52 (đủ > 50)

        # ===== register COMBO (≈15 cái) =====
        self._register_combo_classic()
        self.combo_strategies.append(("combo_mix", self._combo_mix))

        # Khóa luôn về MIX
        self.fixed_algo_name = "algo_mix"
        self.fixed_combo_name = "combo_mix"
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name)

    # ---------- PUBLIC ----------
    def next_prediction(
        self,
        md5_value: str = "",
        alpha: int = 1,
        window_n: Optional[int] = None,
        temperature: float = 0.85,
    ) -> Dict:
        # khóa về mix
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name, self.current_algo_index)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name, self.current_combo_idx)

        # label
        algo_name, algo_fn = self.algorithms[self.current_algo_index]
        probs = _safe_probs(algo_fn(md5_value=md5_value, alpha=alpha, window_n=window_n, temperature=temperature))
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        prob_label = probs[label]

        # combo (ép khớp nhãn)
        combo_name, combo_fn = self.combo_strategies[self.current_combo_idx]
        raw_combo, _ = combo_fn(md5_value=md5_value, temperature=temperature)
        combo, p_combo = self._force_combo_match_label(label, raw_combo)

        res = {
            "label": label,
            "prob_label": prob_label,
            "probs": probs,
            "combo": combo,
            "sum": sum(combo),
            "p_combo": p_combo,
            "algo": algo_name,
            "combo_algo": combo_name,
        }

        # lưu pending theo md5 (nếu có)
        if md5_value:
            self.pending[md5_value] = {
                "guess": label,
                "probs": probs,
                "combo": combo,
                "ts": time.time(),
                "algo": algo_name,
            }
        return res

    def finalize_with_real(self, md5: str, real_label: Label, dice: Optional[List[int]]) -> Dict:
        """Chốt ván ở B2: dùng đúng guess đã lưu trong pending (không tính lại)."""
        if dice is None:
            dice = [0, 0, 0]
        pend = self.pending.pop(md5, None)
        if pend is None:
            # không có pending (nhập lịch sử) → fallback
            guess = self.next_prediction(md5_value=md5)["label"]
            algo_name = self.algorithms[self.current_algo_index][0]
        else:
            guess = pend["guess"]
            algo_name = pend.get("algo", self.algorithms[self.current_algo_index][0])

        win = guess == real_label
        rec = Record(time.time(), md5 or "", dice, guess, real_label, win, algo_name)
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
            "last_record": (
                {
                    "md5": last.md5,
                    "dice": last.dice,
                    "guess": last.guess,
                    "real": last.real,
                    "win": last.win,
                    "algo": last.algo,
                }
                if last
                else None
            ),
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
        self.pending.clear()
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name)

    def cleanup(self):
        return

    # ---------- internal utils ----------
    def _index_of(self, items: List[Tuple[str, Callable]], name: str, default_idx: int = 0) -> int:
        for i, (n, _) in enumerate(items):
            if n == name:
                return i
        return default_idx

    def _stat_inc(self, table: Dict[str, Stats], key: str, win: bool):
        table.setdefault(key, Stats()).add(win)

    # ===== register LABEL (classic ~50) =====
    def _register_dirichlet_classic(self):
        # 7 thuật toán
        setups = [
            (12, 1), (12, 2), (12, 3),
            (20, 1), (20, 2), (20, 3),
            (16, 5),
        ]
        for w, a in setups:
            name = f"algo_dirichlet_w{w}_a{a}"
            def make(w=w, a=a):
                def fn(**kw):
                    c = self._recent_counts(w)
                    prior = {"Tài": a, "Xỉu": a, "Triple": max(1, a // 10)}
                    p = {k: prior[k] + c.get(k, 0) for k in prior}
                    return _safe_probs(p)
                return fn
            self.algorithms.append((name, make()))

    def _register_momentum_classic(self):
        # 11 thuật toán
        setups = [
            (8, 1.0), (8, 1.2), (8, 1.5),
            (12, 1.0), (12, 1.2), (12, 1.5),
            (15, 1.0), (18, 1.0),
            (20, 1.0), (20, 1.2), (20, 1.5),
        ]
        for w, g in setups:
            name = f"algo_momentum_w{w}_g{int(g*10)}"
            def make(w=w, g=g):
                def fn(**kw):
                    last = self.records[-w:]
                    bias = sum(1 if r.real == "Tài" else -1 if r.real == "Xỉu" else 0 for r in last)
                    p_t = 0.48 + 0.02 * g * (bias / (w or 1))
                    p_x = 0.48 - 0.02 * g * (bias / (w or 1))
                    return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": 0.04})
                return fn
            self.algorithms.append((name, make()))

    def _register_markov1_classic(self):
        # 5 thuật toán
        setups = [(20, 1.0), (20, 2.0), (40, 1.0), (40, 2.0), (80, 1.0)]
        for w, s in setups:
            name = f"algo_markov1_w{w}_s{int(s*10)}"
            def make(w=w, s=s):
                def fn(**kw):
                    last = self.records[-w:]
                    trans = {"Tài": {"Tài": s, "Xỉu": s}, "Xỉu": {"Tài": s, "Xỉu": s}}
                    prev = None
                    for r in last:
                        cur = r.real if r.real in ("Tài", "Xỉu") else ("Tài" if r.dice and sum(r.dice) >= 11 else "Xỉu")
                        if prev is not None:
                            trans[prev][cur] += 1
                        prev = cur
                    start = prev or "Tài"
                    p_t = trans[start]["Tài"]
                    p_x = trans[start]["Xỉu"]
                    return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": 0.03})
                return fn
            self.algorithms.append((name, make()))

    def _register_ewma_classic(self):
        # 4 thuật toán
        for d in (0.92, 0.94, 0.96, 0.98):
            name = f"algo_ewma_d{int(d*100)}"
            def make(decay=d):
                def fn(**kw):
                    w = {"Tài": 1.0, "Xỉu": 1.0}
                    for r in self.records:
                        w["Tài"] *= decay
                        w["Xỉu"] *= decay
                        key = r.real if r.real in ("Tài", "Xỉu") else ("Tài" if r.dice and sum(r.dice) >= 11 else "Xỉu")
                        w[key] += 1.0
                    return _safe_probs({"Tài": w["Tài"], "Xỉu": w["Xỉu"], "Triple": 0.03})
                return fn
            self.algorithms.append((name, make()))

    def _register_streak_classic(self):
        # 5 thuật toán
        setups = [(3, 0.6), (3, 0.8), (4, 0.6), (4, 0.8), (5, 0.6)]
        for t, b in setups:
            name = f"algo_streak_t{t}_b{int(b*10)}"
            def make(t=t, b=b):
                def fn(**kw):
                    last = list(reversed(self.records))
                    cur = None
                    cnt = 0
                    for r in last:
                        lab = r.real if r.real in ("Tài", "Xỉu") else ("Tài" if r.dice and sum(r.dice) >= 11 else "Xỉu")
                        if cur is None:
                            cur = lab
                            cnt = 1
                        elif lab == cur:
                            cnt += 1
                        else:
                            break
                    if cnt >= t:
                        if cur == "Tài":
                            p = {"Tài": 0.5 + 0.05 * b, "Xỉu": 0.45 - 0.05 * b, "Triple": 0.05}
                        else:
                            p = {"Tài": 0.45 - 0.05 * b, "Xỉu": 0.5 + 0.05 * b, "Triple": 0.05}
                    else:
                        p = {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
                    return _safe_probs(p)
                return fn
            self.algorithms.append((name, make()))

    def _register_parity_classic(self):
        # 3 thuật toán
        for w in (12, 16, 20):
            name = f"algo_parity_w{w}"
            def make(w=w):
                def fn(**kw):
                    last = self.records[-w:]
                    odd = sum(1 for r in last if r.dice and (sum(r.dice) % 2 == 1))
                    even = len(last) - odd
                    p_t = 0.48 + 0.02 * (even - odd) / (w or 1)
                    p_x = 0.48 - 0.02 * (even - odd) / (w or 1)
                    return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": 0.04})
                return fn
            self.algorithms.append((name, make()))

    def _register_md5_bias_classic(self):
        # 7 thuật toán
        setups = [
            (0, 0.15, 0.0), (0, 0.20, 0.0),
            (3, 0.15, 0.0), (3, 0.20, 0.0),
            (5, 0.15, 0.0), (5, 0.20, 0.0),
            (2, 0.10, 0.0),
        ]
        for bi, sc, sh in setups:
            name = f"algo_md5_b{bi}_s{int(sc*100)}_h{int(sh*10)}"
            def make(bi=bi, sc=sc, sh=sh):
                def fn(**kw):
                    f = _hash_feature(kw.get("md5_value", ""), bi) + sh
                    f = max(0.0, min(1.0, f))
                    p_t = 0.45 + sc * f
                    p_x = 0.45 + sc * (1.0 - f)
                    p_tri = 0.10 * (0.5 - abs(f - 0.5))
                    return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": p_tri})
                return fn
            self.algorithms.append((name, make()))

    def _register_noise_classic(self):
        # 9 thuật toán
        for s in range(9):
            name = f"algo_noise_s{s}"
            rnd = random.Random(s)
            def make(rnd=rnd):
                def fn(**kw):
                    base = {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
                    j = {k: v + (rnd.random() - 0.5) * 0.02 for k, v in base.items()}
                    return _safe_probs(j)
                return fn
            self.algorithms.append((name, make()))

    # ===== COMBO classic (~15) =====
    def _register_combo_classic(self):
        # centers
        for lo, hi in [(9, 11), (10, 11), (10, 12), (9, 12)]:
            name = f"combo_center_{lo}_{hi}"
            def make(lo=lo, hi=hi):
                def fn(**kw):
                    s = self._weighted_choice_by_sum(list(range(lo, hi + 1)))
                    return self._sample_sum(s), self._sum_prob(s)
                return fn
            self.combo_strategies.append((name, make()))

        # bands hẹp
        for lo, hi in [(7, 9), (8, 10), (11, 13), (12, 14), (5, 7), (14, 16)]:
            name = f"combo_band_{lo}_{hi}"
            def make(lo=lo, hi=hi):
                def fn(**kw):
                    rng = list(range(lo, hi + 1))
                    s = self._weighted_choice_by_sum(rng)
                    return self._sample_sum(s), self._sum_prob(s)
                return fn
            self.combo_strategies.append((name, make()))

        # edges
        for a, b, c, d in [(3, 4, 17, 18), (3, 5, 16, 18)]:
            name = f"combo_edges_{a}-{b}-{c}-{d}"
            def make(a=a, b=b, c=c, d=d):
                def fn(**kw):
                    picks = list(range(a, b + 1)) + list(range(c, d + 1))
                    s = self._weighted_choice_by_sum(picks)
                    return self._sample_sum(s), self._sum_prob(s)
                return fn
            self.combo_strategies.append((name, make()))

        # md5-based
        for bi in range(3):
            name = f"combo_md5b{bi}"
            def make(bi=bi):
                def fn(**kw):
                    f = _hash_feature(kw.get("md5_value", ""), bi)
                    if f > 0.66:
                        pool = list(range(11, 18))
                    elif f < 0.33:
                        pool = list(range(3, 10))
                    else:
                        pool = list(range(9, 12))
                    s = self._weighted_choice_by_sum(pool)
                    return self._sample_sum(s), self._sum_prob(s)
                return fn
            self.combo_strategies.append((name, make()))

        # random seeds
        for s in range(5):
            name = f"combo_random_s{s}"
            rnd = random.Random(s)
            def make(rnd=rnd):
                def fn(**kw):
                    d = [rnd.randint(1, 6) for _ in range(3)]
                    return d, self._sum_prob(sum(d))
                return fn
            self.combo_strategies.append((name, make()))

    # ---------- combo: MIX ----------
    def _combo_mix(self, **kw) -> Tuple[List[int], float]:
        cands: List[Tuple[List[int], float]] = []
        for name, fn in self.combo_strategies:
            if name == "combo_mix":
                continue
            try:
                d, p = fn(**kw)
                if not (isinstance(d, list) and len(d) == 3 and all(1 <= int(x) <= 6 for x in d)):
                    continue
                cands.append((d, float(p)))
            except Exception:
                continue
        if not cands:
            d = [random.randint(1, 6) for _ in range(3)]
            return d, self._sum_prob(sum(d))
        d, p = max(cands, key=lambda x: x[1])
        return d, p

    # ---------- label: MIX ----------
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

    # ---------- shared utils ----------
    def _recent_counts(self, window_n: Optional[int]) -> Dict[Label, int]:
        n = window_n or self.window_default
        last = self.records[-n:] if n > 0 else self.records[:]
        c = {"Tài": 0, "Xỉu": 0, "Triple": 0}
        for r in last:
            c[r.real] = c.get(r.real, 0) + 1
        return c

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

    def _force_combo_match_label(self, label: str, combo: List[int]) -> Tuple[List[int], float]:
        if label == "Triple":
            k = random.randint(1, 6)
            return [k, k, k], 6.0 / 216.0
        s = sum(combo)
        if label == "Tài" and s >= 11:
            return combo, self._sum_prob(s)
        if label == "Xỉu" and s <= 10:
            return combo, self._sum_prob(s)
        pool = list(range(11, 19)) if label == "Tài" else list(range(3, 11))
        s_new = self._weighted_choice_by_sum(pool)
        new_combo = self._sample_sum(s_new)
        return new_combo, self._sum_prob(s_new)

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
