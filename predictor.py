# predictor.py  — big library of strategies (240+ variants)
from __future__ import annotations
import csv, hashlib, io, random, time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

Label = str  # "Tài" | "Xỉu" | "Triple"

# ---------- small helpers ----------
def _label_from_sum(s: int) -> Label:
    return "Xỉu" if s <= 10 else "Tài"

def _label_from_dice(d: List[int]) -> Label:
    if len(d) == 3 and d[0] == d[1] == d[2]:
        return "Triple"
    return _label_from_sum(sum(d))

def _safe_probs(p: Dict[Label, float]) -> Dict[Label, float]:
    total = sum(max(0.0, v) for v in p.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in p.items()}

def _hash_feature(md5: str, byte_idx: int = -1) -> float:
    """Stable feature from md5 string (no decryption)."""
    if not md5:
        return 0.0
    h = hashlib.md5(md5.encode("utf-8")).hexdigest()
    try:
        if byte_idx is None or byte_idx < 0:
            return int(h[-2:], 16) / 255.0
        off = (byte_idx % 16) * 2
        return int(h[off:off+2], 16) / 255.0
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
        if win: self.wins += 1

# ---------- core ----------
class TaiXiuAI:
    """
    Huge strategy library:
      - 140+ label (dice) strategies via parameterized families
      - 100+ combo strategies via parameterized families
      - Fixed to use algo_mix + combo_mix
      - Enforces combo to match predicted label
    """

        # ---------- MIX cho combo (quên chưa thêm) ----------
    def _combo_mix(self, **kw) -> Tuple[List[int], float]:
        """
        Gộp tất cả chiến lược combo khác (trừ chính nó) và chọn phương án
        có xác suất tổng (p_combo) cao nhất. Đây là bản 'ensemble' cho combo.
        """
        cands: List[Tuple[List[int], float]] = []
        for name, fn in self.combo_strategies:
            if name == "combo_mix":
                continue
            try:
                d, p = fn(**kw)
                # bảo vệ dữ liệu xấu
                if not (isinstance(d, list) and len(d) == 3 and all(1 <= int(x) <= 6 for x in d)):
                    continue
                cands.append((d, float(p)))
            except Exception:
                # nếu một chiến lược lỗi, bỏ qua
                continue

        if not cands:
            # fallback an toàn
            d = [random.randint(1, 6) for _ in range(3)]
            return d, self._sum_prob(sum(d))

        # chọn combo có p_combo lớn nhất
        d, p = max(cands, key=lambda x: x[1])
        return d, p

    def __init__(self):
        self.records: List[Record] = []
        self.per_algo: Dict[str, Stats] = {}
        self.combo_stats: Dict[str, Stats] = {}
        self.window_default = 20

        self._sum_cache: Dict[int, float] = {}

        # build strategy registries
        self.algorithms: List[Tuple[str, Callable]] = []
        self.combo_strategies: List[Tuple[str, Callable]] = []

        # register families (LABEL side)
        self._register_dirichlet_family()
        self._register_momentum_family()
        self._register_md5_bias_family()
        self._register_markov1_family()
        self._register_ewma_family()
        self._register_streak_family()
        self._register_parity_family()
        self._register_noise_family()
        # final mix
        self.algorithms.append(("algo_mix", self._algo_mix))

        # register combo families
        self._register_combo_center_family()
        self._register_combo_edges_family()
        self._register_combo_bands_family()
        self._register_combo_mod_family()
        self._register_combo_md5_family()
        self._register_combo_random_family()
        # final mix
        self.combo_strategies.append(("combo_mix", self._combo_mix))

        # lock to MIX
        self.fixed_algo_name = "algo_mix"
        self.fixed_combo_name = "combo_mix"
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx = self._index_of(self.combo_strategies, self.fixed_combo_name)

    # ---------- PUBLIC API ----------
    def next_prediction(self, md5_value: str = "", alpha: int = 1,
                        window_n: Optional[int] = None, temperature: float = 0.85) -> Dict:
        # lock indices
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name, self.current_algo_index)
        self.current_combo_idx  = self._index_of(self.combo_strategies, self.fixed_combo_name, self.current_combo_idx)

        # label probabilities
        algo_name, algo_fn = self.algorithms[self.current_algo_index]
        probs = algo_fn(md5_value=md5_value, alpha=alpha, window_n=window_n, temperature=temperature)
        probs = _safe_probs(probs)
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        prob_label = probs[label]

        # raw combo
        combo_name, combo_fn = self.combo_strategies[self.current_combo_idx]
        raw_combo, _ = combo_fn(md5_value=md5_value, temperature=temperature)

        # force combo to match label
        combo, p_combo = self._force_combo_match_label(label, raw_combo)

        return {
            "label": label, "prob_label": prob_label, "probs": probs,
            "combo": combo, "sum": sum(combo), "p_combo": p_combo,
            "algo": algo_name, "combo_algo": combo_name
        }

    def update_with_real(self, md5: str, real_label: Label, dice: Optional[List[int]]) -> Dict:
        if dice is None: dice = [0,0,0]
        last_pred = self.next_prediction(md5_value=md5)
        guess = last_pred["label"]
        win = (guess == real_label)
        rec = Record(time.time(), md5 or "", dice, guess, real_label, win,
                     self.algorithms[self.current_algo_index][0])
        self.records.append(rec)
        self._stat_inc(self.per_algo, rec.algo, rec.win)
        self._stat_inc(self.combo_stats, self.combo_strategies[self.current_combo_idx][0], rec.win)
        return {"md5": rec.md5, "dice": rec.dice, "guess": rec.guess, "real": rec.real, "win": rec.win, "algo": rec.algo}

    def summary(self) -> Dict:
        total = len(self.records); wins = sum(1 for r in self.records if r.win)
        last = self.records[-1] if self.records else None
        return {
            "total_rounds": total, "wins": wins, "win_rate": (wins/total) if total else 0.0,
            "current_algo": self.algorithms[self.current_algo_index][0],
            "current_combo_algo": self.combo_strategies[self.current_combo_idx][0],
            "last_record": ({
                "md5": last.md5, "dice": last.dice, "guess": last.guess,
                "real": last.real, "win": last.win, "algo": last.algo
            } if last else None),
            "per_algo": {k: {"total": v.total, "wins": v.wins, "win_rate": v.win_rate} for k,v in self.per_algo.items()},
        }

    def export_algo_stats_csv(self) -> str:
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["algo","total","wins","win_rate"])
        for k,v in self.per_algo.items():
            w.writerow([k, v.total, v.wins, f"{v.win_rate:.4f}"])
        return buf.getvalue()

    def export_combo_stats_csv(self) -> str:
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["combo_algo","total","wins","win_rate"])
        for k,v in self.combo_stats.items():
            w.writerow([k, v.total, v.wins, f"{v.win_rate:.4f}"])
        return buf.getvalue()

    def reset(self):
        self.records.clear(); self.per_algo.clear(); self.combo_stats.clear()
        self.current_algo_index = self._index_of(self.algorithms, self.fixed_algo_name)
        self.current_combo_idx  = self._index_of(self.combo_strategies, self.fixed_combo_name)

    def cleanup(self): return

    # ---------- INTERNAL ----------
    def _index_of(self, items: List[Tuple[str, Callable]], name: str, default_idx: int = 0) -> int:
        for i,(n,_) in enumerate(items):
            if n==name: return i
        return default_idx

    def _stat_inc(self, table: Dict[str, Stats], key: str, win: bool):
        table.setdefault(key, Stats()).add(win)

    # ======== LABEL families (create 140+ variants) ========
    # 1) Dirichlet frequency
    def _register_dirichlet_family(self):
        windows = [8,12,16,20,24,30,40,50]
        alphas  = [1,2,3,5,8,13]
        for w in windows:
            for a in alphas:
                name = f"algo_dirichlet_w{w}_a{a}"
                def make(w=w, a=a):
                    def fn(**kw):
                        c = self._recent_counts(w)
                        prior = {"Tài": a, "Xỉu": a, "Triple": max(1, a//10)}
                        p = {k: prior[k] + c.get(k,0) for k in prior}
                        return _safe_probs(p)
                    return fn
                self.algorithms.append((name, make()))
        # ~48 variants

    # 2) Momentum (recent bias)
    def _register_momentum_family(self):
        windows = [6,8,10,12,15,18,20,24]
        gains   = [0.5,0.8,1.0,1.2,1.5,2.0]
        for w in windows:
            for g in gains:
                name = f"algo_momentum_w{w}_g{int(g*10)}"
                def make(w=w, g=g):
                    def fn(**kw):
                        last = self.records[-w:]
                        bias = sum(1 if r.real=="Tài" else -1 if r.real=="Xỉu" else 0 for r in last)
                        p_t = 0.48 + 0.02*g*(bias/(w or 1))
                        p_x = 0.48 - 0.02*g*(bias/(w or 1))
                        p_tri = 0.04
                        return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": p_tri})
                    return fn
                self.algorithms.append((name, make()))
        # ~48 variants

    # 3) MD5 bias variants
    def _register_md5_bias_family(self):
        bytes_idx = list(range(8))  # first 8 bytes positions
        scales    = [0.10,0.15,0.20,0.25]
        shifts    = [0.0, 0.1, -0.1]
        for bi in bytes_idx:
            for sc in scales:
                for sh in shifts:
                    name = f"algo_md5_b{bi}_s{int(sc*100)}_h{int(sh*10)}"
                    def make(bi=bi, sc=sc, sh=sh):
                        def fn(**kw):
                            f = _hash_feature(kw.get("md5_value",""), bi) + sh
                            f = max(0.0, min(1.0, f))
                            p_t = 0.45 + sc * f
                            p_x = 0.45 + sc * (1.0 - f)
                            p_tri = 0.10 * (0.5 - abs(f-0.5))
                            return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": p_tri})
                        return fn
                    self.algorithms.append((name, make()))
        # 8*4*3 = 96 variants

    # 4) Markov order-1 (transition prob)
    def _register_markov1_family(self):
        windows = [20,40,80]
        smooths = [0.5,1.0,2.0,3.0]
        for w in windows:
            for s in smooths:
                name = f"algo_markov1_w{w}_s{int(s*10)}"
                def make(w=w, s=s):
                    def fn(**kw):
                        last = self.records[-w:]
                        # states only Tai/Xiu (merge Triple to nearest by sum trend)
                        trans = { "Tài": {"Tài": s, "Xỉu": s},
                                  "Xỉu": {"Tài": s, "Xỉu": s} }
                        prev = None
                        for r in last:
                            cur = r.real if r.real in ("Tài","Xỉu") else ("Tài" if r.dice and sum(r.dice)>=11 else "Xỉu")
                            if prev is not None:
                                trans[prev][cur] += 1
                            prev = cur
                        start = prev or "Tài"
                        p_t = trans[start]["Tài"]; p_x = trans[start]["Xỉu"]
                        return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": 0.03})
                    return fn
                self.algorithms.append((name, make()))
        # 12 variants

    # 5) EWMA family (exponential moving average of wins by label)
    def _register_ewma_family(self):
        decays = [0.85,0.9,0.92,0.94,0.96,0.98]
        for d in decays:
            name = f"algo_ewma_d{int(d*100)}"
            def make(decay=d):
                def fn(**kw):
                    w = {"Tài":1.0, "Xỉu":1.0}
                    for r in self.records:
                        w["Tài"] *= decay
                        w["Xỉu"] *= decay
                        w[r.real if r.real in ("Tài","Xỉu") else ("Tài" if r.dice and sum(r.dice)>=11 else "Xỉu")] += 1.0
                    return _safe_probs({"Tài": w["Tài"], "Xỉu": w["Xỉu"], "Triple": 0.03})
                return fn
            self.algorithms.append((name, make()))
        # 6 variants

    # 6) Streak-aware
    def _register_streak_family(self):
        thresholds = [2,3,4,5,6]
        biases = [0.4,0.6,0.8,1.0]
        for t in thresholds:
            for b in biases:
                name = f"algo_streak_t{t}_b{int(b*10)}"
                def make(t=t, b=b):
                    def fn(**kw):
                        # detect last streak
                        last = list(reversed(self.records))
                        cur = None; cnt = 0
                        for r in last:
                            lab = r.real if r.real in ("Tài","Xỉu") else ("Tài" if r.dice and sum(r.dice)>=11 else "Xỉu")
                            if cur is None: cur = lab; cnt = 1
                            elif lab == cur: cnt += 1
                            else: break
                        if cnt >= t:
                            if cur == "Tài":
                                p = {"Tài": 0.5 + 0.05*b, "Xỉu": 0.45 - 0.05*b, "Triple": 0.05}
                            else:
                                p = {"Tài": 0.45 - 0.05*b, "Xỉu": 0.5 + 0.05*b, "Triple": 0.05}
                        else:
                            p = {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
                        return _safe_probs(p)
                    return fn
                self.algorithms.append((name, make()))
        # 20 variants

    # 7) Parity (sum parity trend)
    def _register_parity_family(self):
        windows = [8,12,16,20,30]
        for w in windows:
            name = f"algo_parity_w{w}"
            def make(w=w):
                def fn(**kw):
                    last = self.records[-w:]
                    odd = sum(1 for r in last if r.dice and (sum(r.dice)%2==1))
                    even = len(last) - odd
                    p_t = 0.48 + 0.02*(even - odd)/(w or 1)
                    p_x = 0.48 - 0.02*(even - odd)/(w or 1)
                    return _safe_probs({"Tài": p_t, "Xỉu": p_x, "Triple": 0.04})
                return fn
            self.algorithms.append((name, make()))
        # 5 variants

    # 8) Small noise ensembles (different seeds/temps)
    def _register_noise_family(self):
        seeds = range(10)
        for s in seeds:
            name = f"algo_noise_s{s}"
            rnd = random.Random(s)
            def make(rnd=rnd):
                def fn(**kw):
                    base = {"Tài": 0.485, "Xỉu": 0.485, "Triple": 0.03}
                    j = {k: v + (rnd.random()-0.5)*0.02 for k,v in base.items()}
                    return _safe_probs(j)
                return fn
            self.algorithms.append((name, make()))
        # 10 variants

    # ======== COMBO families (create 100+ variants) ========
    def _register_combo_center_family(self):
        centers = [(9,11),(10,11),(10,12),(9,12)]
        for lo,hi in centers:
            name = f"combo_center_{lo}_{hi}"
            def make(lo=lo,hi=hi):
                def fn(**kw):
                    s = self._weighted_choice_by_sum(list(range(lo,hi+1)))
                    return self._sample_sum(s), self._sum_prob(s)
                return fn
            self.combo_strategies.append((name, make()))
        # 4

    def _register_combo_edges_family(self):
        bands = [(3,4,17,18),(3,5,16,18)]
        for a,b,c,d in bands:
            name = f"combo_edges_{a}-{b}-{c}-{d}"
            def make(a=a,b=b,c=c,d=d):
                def fn(**kw):
                    picks = list(range(a,b+1))+list(range(c,d+1))
                    s = self._weighted_choice_by_sum(picks)
                    return self._sample_sum(s), self._sum_prob(s)
                return fn
            self.combo_strategies.append((name, make()))
        # 2

    def _register_combo_bands_family(self):
        bands = [(7,9),(8,10),(11,13),(12,14),(5,7),(14,16)]
        temps = [1,2,3,4,5,6,7,8]
        for lo,hi in bands:
            for t in temps:
                name = f"combo_band_{lo}_{hi}_t{t}"
                def make(lo=lo,hi=hi,t=t):
                    def fn(**kw):
                        rng = list(range(lo,hi+1))
                        weights = [self._sum_prob(s)**(1.0/t) for s in rng]
                        total = sum(weights) or 1.0
                        r = random.random()*total; acc=0.0
                        pick = rng[-1]
                        for s,w in zip(rng,weights):
                            acc += w
                            if r <= acc: pick=s; break
                        return self._sample_sum(pick), self._sum_prob(pick)
                    return fn
                self.combo_strategies.append((name, make()))
        # 6*8 = 48

    def _register_combo_mod_family(self):
        for mod in (2,3,4):
            for r in range(mod):
                name = f"combo_mod{mod}_r{r}"
                def make(mod=mod,r=r):
                    def fn(**kw):
                        pool = [s for s in range(3,19) if s%mod==r]
                        if not pool: pool=[10,11]
                        s = self._weighted_choice_by_sum(pool)
                        return self._sample_sum(s), self._sum_prob(s)
                    return fn
                self.combo_strategies.append((name, make()))
        # ~9

    def _register_combo_md5_family(self):
        for bi in range(8):
            name = f"combo_md5b{bi}"
            def make(bi=bi):
                def fn(**kw):
                    f = _hash_feature(kw.get("md5_value",""), bi)
                    if f > 0.66:
                        pool = list(range(11,18))
                    elif f < 0.33:
                        pool = list(range(3,10))
                    else:
                        pool = list(range(9,12))
                    s = self._weighted_choice_by_sum(pool)
                    return self._sample_sum(s), self._sum_prob(s)
                return fn
            self.combo_strategies.append((name, make()))
        # 8

    def _register_combo_random_family(self):
        for s in range(30):
            name = f"combo_random_s{s}"
            rnd = random.Random(s)
            def make(rnd=rnd):
                def fn(**kw):
                    d = [rnd.randint(1,6) for _ in range(3)]
                    return d, self._sum_prob(sum(d))
                return fn
            self.combo_strategies.append((name, make()))
        # 30

    # ---------- combo alignment ----------
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
            return [k, k, k], 6.0/216.0  # group prob of any triple
        s = sum(combo)
        if label == "Tài" and s >= 11:  return combo, self._sum_prob(s)
        if label == "Xỉu" and s <= 10:  return combo, self._sum_prob(s)
        pool = list(range(11,19)) if label=="Tài" else list(range(3,11))
        s_new = self._weighted_choice_by_sum(pool)
        new_combo = self._sample_sum(s_new)
        return new_combo, self._sum_prob(s_new)

    # ---------- label utilities ----------
    def _recent_counts(self, window_n: Optional[int]) -> Dict[Label, int]:
        n = window_n or self.window_default
        last = self.records[-n:] if n>0 else self.records[:]
        c = {"Tài":0,"Xỉu":0,"Triple":0}
        for r in last:
            c[r.real] = c.get(r.real,0)+1
        return c

    def _algo_mix(self, **kw) -> Dict[Label, float]:
        others = [fn for name,fn in self.algorithms if name!="algo_mix"]
        if not others: return {"Tài":0.485,"Xỉu":0.485,"Triple":0.03}
        agg = {"Tài":0.0,"Xỉu":0.0,"Triple":0.0}
        for fn in others:
            p = fn(**kw)
            for k,v in p.items(): agg[k]+=v
        for k in agg: agg[k]/=len(others)
        return _safe_probs(agg)

    # ---------- combo utilities ----------
    def _sum_prob(self, s: int) -> float:
        if not self._sum_cache:
            freq = {k:0 for k in range(3,19)}
            for a in range(1,7):
                for b in range(1,7):
                    for c in range(1,7):
                        freq[a+b+c]+=1
            total = 6**3
            for k,v in freq.items():
                self._sum_cache[k] = v/total
        return float(self._sum_cache.get(s,0.0))

    def _sample_sum(self, s: int) -> List[int]:
        s = max(3, min(18, s))
        for _ in range(30):
            a=random.randint(1,6); b=random.randint(1,6); c=s-a-b
            if 1<=c<=6: return [a,b,c]
        return [random.randint(1,6) for _ in range(3)]

