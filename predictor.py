# predictor.py
import os, json, random, time, tempfile, atexit, math, hashlib
from collections import Counter, deque
from typing import List, Dict, Optional, Tuple, Callable

Label = str  # "Tài" | "Xỉu" | "Triple"

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v

class TaiXiuAI:
    """Core logic: nhiều thuật toán + 'algo_mix' và nhiều combo + 'combo_mix'."""
    def __init__(self):
        # Trạng thái
        self.history: List[Dict] = []
        self.pending: Optional[Dict] = None

        self.win_streak = 0
        self.lose_streak = 0
        self.switch_on_losses = 2
        self.auto_best_window = 40

        self.recent_combos = deque(maxlen=10)
        self.algo_stats: Dict[str, Dict[str,int]] = {}
        self.combo_stats: Dict[str, Dict[str,int]] = {}

        # Thuật toán nhãn
        self.algorithms: List[Tuple[str, Callable[[List[Dict], str], Label]]] = []
        self._build_algorithms()
        # Combo strategies
        self.combo_strategies: List[Tuple[str, Callable[
            [Dict[int,float], Label, deque, Optional[float]],
            Tuple[Tuple[int,int,int], int, float]
        ]]] = []
        self._md5_for_combo_seed = ""
        self._build_combo_strategies()

        # Mặc định: 2 siêu-thuật toán
        self.current_algo_index = self._algo_index_by_name("algo_mix")
        self.current_combo_idx = self._combo_index_by_name("combo_mix")

        # File tạm riêng session
        fd, self._tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        atexit.register(self.cleanup)
        self._save_temp()

    # -------------- persist --------------
    def _save_temp(self):
        tmp = self._tmp_path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"history": self.history, "pending": self.pending,
                       "algo_stats": self.algo_stats, "combo_stats": self.combo_stats},
                      f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._tmp_path)

    def cleanup(self):
        for _ in range(10):
            try: os.remove(self._tmp_path); return
            except PermissionError: time.sleep(0.2)
            except FileNotFoundError: return
        print("[WARN] Cannot remove temp file:", self._tmp_path)

    def reset(self):
        self.history.clear()
        self.pending = None
        self.win_streak = self.lose_streak = 0
        self.recent_combos.clear()
        self.algo_stats.clear()
        self.combo_stats.clear()
        self.current_algo_index = self._algo_index_by_name("algo_mix")
        self.current_combo_idx = self._combo_index_by_name("combo_mix")
        self._save_temp()
        return {"ok": True}

    # -------------- helpers --------------
    def _recent(self, n: Optional[int]) -> List[Dict]:
        return self.history if not n or n<=0 else self.history[-n:]

    @staticmethod
    def _label_from_dice(d: List[int]) -> Label:
        if d[0]==d[1]==d[2]: return "Triple"
        return "Xỉu" if sum(d)<=10 else "Tài"

    # -------------- Bayes/EWMA --------------
    def _bayes_face_probs(self, alpha: int=1, window_n: Optional[int]=None) -> Dict[int,float]:
        r = self._recent(window_n)
        if not r: return {f:1/6 for f in range(1,7)}
        c = Counter([x for rec in r for x in rec["dice"]])
        post = {f:c.get(f,0)+alpha for f in range(1,7)}
        s = sum(post.values())
        return {f: post[f]/s for f in range(1,7)}

    def bayes_group_probs(self, alpha: int=1, window_n: Optional[int]=None) -> Dict[Label,float]:
        p = self._bayes_face_probs(alpha, window_n)
        p_sum = {k:0.0 for k in range(3,19)}
        for a in range(1,7):
            for b in range(1,7):
                for c in range(1,7):
                    pr = p[a]*p[b]*p[c]; p_sum[a+b+c]+=pr
        p_tri = sum(p[i]**3 for i in range(1,7))
        p_xiu = sum(v for s,v in p_sum.items() if s<=10) - p_tri
        p_tai = sum(v for s,v in p_sum.items() if s>=11) - p_tri
        tot = p_tai+p_xiu+p_tri
        if tot>0: p_tai/=tot; p_xiu/=tot; p_tri/=tot
        return {"Tài":p_tai,"Xỉu":p_xiu,"Triple":p_tri}

    def _ewma_face_probs(self, decay:float=0.9, window_n: Optional[int]=None) -> Dict[int,float]:
        r = self._recent(window_n)
        if not r: return {f:1/6 for f in range(1,7)}
        wcnt = {f:1e-6 for f in range(1,7)}
        n = len(r)
        for i,rec in enumerate(r):
            w = decay**(n-1-i)
            for d in rec["dice"]: wcnt[d]+=w
        s = sum(wcnt.values())
        return {f: wcnt[f]/s for f in range(1,7)}

    # -------------- thuật toán nhãn --------------
    def _algo_index_by_name(self, name:str)->int:
        for i,(n,_) in enumerate(self.algorithms):
            if n==name: return i
        return 0

    def _build_algorithms(self):
        A: List[Tuple[str, Callable[[List[Dict], str], Label]]] = []

        # 1) random
        A.append(("algo_random", lambda h,m: random.choice(["Tài","Xỉu"])))

        # 2) Bayes (nhiều cấu hình)
        def bayes_picker(alpha,window):
            def f(h,m):
                probs = self.bayes_group_probs(alpha=alpha, window_n=window)
                return max(probs,key=probs.get)
            return f
        for a in (1,2,3):
            for w in (0,20,40,60):
                A.append((f"algo_bayes_a{a}_w{w}", bayes_picker(a,w)))

        # 3) EWMA
        def ewma_picker(decay,window):
            def f(h,m):
                p = self._ewma_face_probs(decay,window)
                # chuyển sang group nhanh
                p_sum={k:0.0 for k in range(3,19)}
                for a in range(1,7):
                    for b in range(1,7):
                        for c in range(1,7):
                            pr=p[a]*p[b]*p[c]; p_sum[a+b+c]+=pr
                p_tri=sum(p[i]**3 for i in range(1,7))
                p_xiu=sum(v for s,v in p_sum.items() if s<=10)-p_tri
                p_tai=sum(v for s,v in p_sum.items() if s>=11)-p_tri
                tot=p_tai+p_xiu+p_tri
                if tot>0: p_tai/=tot;p_xiu/=tot;p_tri/=tot
                probs={"Tài":p_tai,"Xỉu":p_xiu,"Triple":p_tri}
                return max(probs,key=probs.get)
            return f
        for d,w in ((0.9,30),(0.92,50),(0.95,80)):
            A.append((f"algo_ewma_d{str(d).replace('.','')}_w{w}", ewma_picker(d,w)))

        # 4) Markov outcome (1-step)
        def markov1(h,m):
            if len(h)<2: return random.choice(["Tài","Xỉu"])
            last=h[-1]["real"]; prev=h[-2]["real"]
            return ("Tài" if last=="Xỉu" else "Xỉu") if last==prev else last
        A.append(("algo_markov_1step", markov1))

        # 5) Sum bias (thống kê gần đây)
        def sum_bias(window):
            def f(h,m):
                if not h: return random.choice(["Tài","Xỉu"])
                span = h[-window:] if window>0 else h
                c=Counter()
                for r in span:
                    if r["dice"][0]==r["dice"][1]==r["dice"][2]: c["Triple"]+=1
                    elif sum(r["dice"])<=10: c["Xỉu"]+=1
                    else: c["Tài"]+=1
                return "Tài" if c["Tài"]>=c["Xỉu"] else "Xỉu"
            return f
        for w in (30,60,100): A.append((f"algo_sum_bias_w{w}", sum_bias(w)))

        # 6) MD5 mix (nhiễu nhẹ)
        def md5_mix(weight):
            def f(h,m):
                base = self.bayes_group_probs(alpha=1, window_n=50)
                if not m: return max(base,key=base.get)
                u=int(hashlib.md5(m.encode()).hexdigest()[:8],16)/0xFFFFFFFF
                base["Tài"]=max(0.0, base["Tài"]+weight*math.sin(2*math.pi*u))
                base["Xỉu"]=max(0.0, base["Xỉu"]-weight*math.sin(2*math.pi*u)*0.8)
                s=sum(base.values())
                for k in base: base[k]/=s
                return max(base,key=base.get)
            return f
        for w in (0.02,0.05,0.08): A.append((f"algo_md5_mix_w{int(w*100)}", md5_mix(w)))

        # 7) AUTO-BEST (xếp hạng backtest cửa sổ gần)
        def auto_best(h,m):
            if len(h)<6: return bayes_picker(1,40)(h,m)
            K=min(self.auto_best_window,len(h)); start=len(h)-K
            ranks=[]
            for nm,fn in A:
                if nm.startswith("algo_auto_best") or nm=="algo_mix": continue
                win=tot=0
                for t in range(start,len(h)):
                    pred = fn(h[:t], "")
                    if pred==h[t]["real"]: win+=1
                    tot+=1
                wr=win/tot if tot else 0.0
                ranks.append((wr,nm,fn))
            ranks.sort(reverse=True)
            return ranks[0][2](h,m) if ranks else bayes_picker(1,40)(h,m)
        A.append(("algo_auto_best", auto_best))

        # 8) MIX (bỏ phiếu tất cả)
        def algo_mix(h,m):
            votes=Counter()
            probs_acc={"Tài":0.0,"Xỉu":0.0,"Triple":0.0}; nprob=0
            for nm,fn in A:
                if nm=="algo_mix": continue
                guess = fn(h,m)
                votes[guess]+=1
                # ước xác suất từ Bayes để phá hòa (đủ tốt)
                pr = self.bayes_group_probs(alpha=1, window_n=40)
                for k in probs_acc: probs_acc[k]+=pr.get(k,0.0)
                nprob+=1
            choice, top = votes.most_common(1)[0]
            # hòa? dùng xác suất trung bình
            if list(votes.values()).count(top)>1 and nprob>0:
                avg={k:v/nprob for k,v in probs_acc.items()}
                choice = max(avg,key=avg.get)
            return choice
        A.append(("algo_mix", algo_mix))

        self.algorithms = A

    # -------------- combo strategies --------------
    def _combo_index_by_name(self, name:str)->int:
        for i,(n,_) in enumerate(self.combo_strategies):
            if n==name: return i
        return 0

    def _build_combo_strategies(self):
        S = self.combo_strategies
        def _reg(n,fn):
            S.append((n,fn))
            if n not in self.combo_stats: self.combo_stats[n]={"plays":0,"sum_hits":0,"exact_hits":0}

        def allc():
            for a in range(1,7):
                for b in range(1,7):
                    for c in range(1,7):
                        yield (a,b,c)

        def bayes_top(p,label, recent, temp):
            best=None; bw=-1
            for a,b,c in allc():
                s=a+b+c; tri=(a==b==c)
                if (label=="Triple" and not tri) or (label=="Tài" and (s<11 or tri)) or (label=="Xỉu" and (s>10 or tri)): continue
                w=p[a]*p[b]*p[c]
                if (a,b,c) in recent: w*=0.3
                if w>bw: bw=w; best=(a,b,c)
            prod=p[best[0]]*p[best[1]]*p[best[2]]
            return best, sum(best), prod

        def bayes_sample(p,label,recent,temp):
            cand=[]; wts=[]
            T=temp or 0.85
            for a,b,c in allc():
                s=a+b+c; tri=(a==b==c)
                if (label=="Triple" and not tri) or (label=="Tài" and (s<11 or tri)) or (label=="Xỉu" and (s>10 or tri)): continue
                pr=p[a]*p[b]*p[c]
                if pr<=0: continue
                w=max(1e-12, pr**(1/max(1e-6,T)))
                if (a,b,c) in recent: w*=0.35
                cand.append((a,b,c)); wts.append(w)
            if not cand: return bayes_top(p,label,recent,temp)
            tot=sum(wts); r=random.random()*tot; acc=0; ch=cand[-1]
            for combo,w in zip(cand,wts):
                acc+=w
                if r<=acc: ch=combo; break
            pr=p[ch[0]]*p[ch[1]]*p[ch[2]]
            return ch, sum(ch), pr

        def anti_repeat(p,label,recent,temp):
            cand=[]; wts=[]
            for a,b,c in allc():
                s=a+b+c; tri=(a==b==c)
                if (label=="Triple" and not tri) or (label=="Tài" and (s<11 or tri)) or (label=="Xỉu" and (s>10 or tri)): continue
                pr=p[a]*p[b]*p[c]
                if pr<=0: continue
                w=max(1e-12, pr**(1/0.8))
                if (a,b,c) in recent: w*=0.2
                cand.append((a,b,c)); wts.append(w)
            if not cand: return bayes_sample(p,label,recent,temp)
            tot=sum(wts); r=random.random()*tot; acc=0; ch=cand[-1]
            for combo,w in zip(cand,wts):
                acc+=w
                if r<=acc: ch=combo; break
            pr=p[ch[0]]*p[ch[1]]*p[ch[2]]
            return ch, sum(ch), pr

        def sum_bias(p,label,recent,temp):
            best=None; bw=-1
            for a,b,c in allc():
                s=a+b+c; tri=(a==b==c)
                if (label=="Triple" and not tri) or (label=="Tài" and (s<11 or tri)) or (label=="Xỉu" and (s>10 or tri)): continue
                pr=p[a]*p[b]*p[c]
                w = pr * (1+0.03*(s-11)) if label=="Tài" else pr*(1+0.03*(10-s))
                if (a,b,c) in recent: w*=0.6
                if w>bw: bw=w; best=(a,b,c)
            pr=p[best[0]]*p[best[1]]*p[best[2]]
            return best, sum(best), pr

        def md5_mod(p,label,recent,temp):
            md5=self._md5_for_combo_seed or ""
            seed=int(hashlib.md5(md5.encode()).hexdigest()[:8],16) if md5 else 0
            rnd=random.Random(seed)
            best=None; bw=-1
            for a,b,c in allc():
                s=a+b+c; tri=(a==b==c)
                if (label=="Triple" and not tri) or (label=="Tài" and (s<11 or tri)) or (label=="Xỉu" and (s>10 or tri)): continue
                pr=p[a]*p[b]*p[c]*(1+0.2*(rnd.random()-0.5))
                if (a,b,c) in recent: pr*=0.6
                if pr>bw: bw=pr; best=(a,b,c)
            pr=p[best[0]]*p[best[1]]*p[best[2]]
            return best, sum(best), pr

        # đăng ký ~10 chiến lược (đủ phong phú)
        _reg("combo_bayes_top", bayes_top)
        _reg("combo_bayes_sample", bayes_sample)
        _reg("combo_anti_repeat", anti_repeat)
        _reg("combo_sum_bias", sum_bias)
        _reg("combo_md5_mod", md5_mod)
        # biến thể
        for t in (0.7,0.85,1.2,1.5):
            _reg(f"combo_bayes_sample_t{str(t).replace('.','')}",
                 lambda pf,l,rec,tmp,t=t: bayes_sample(pf,l,rec,t))

        # ---- MIX: vote tất cả ----
        def combo_mix(p,label,recent,temp):
            votes=Counter(); any_list=[]
            for nm,fn in self.combo_strategies:
                if nm=="combo_mix": continue
                c,s,pr = fn(p,label,recent,temp)
                votes[c]+=1; any_list.append((c,s,pr))
            best_combo, top = votes.most_common(1)[0]
            if list(votes.values()).count(top)>1:
                # hòa → chọn combo có p_face^3 lớn nhất
                any_list.sort(key=lambda x: x[2], reverse=True)
                best_combo = any_list[0][0]
            a,b,c = best_combo
            return best_combo, a+b+c, p[a]*p[b]*p[c]
        _reg("combo_mix", combo_mix)

    # -------------- dự đoán/ghi nhận --------------
    def predict_with_name(self, md5_value: str)->Tuple[str,Label]:
        name, func = self.algorithms[self.current_algo_index]
        return name, func(self.history, md5_value)

    def create_pending(self, md5_value: str, alpha:int, window_n: Optional[int], temperature: Optional[float]=None)->Dict:
        algo_name, label = self.predict_with_name(md5_value or "")
        probs = self.bayes_group_probs(alpha=alpha, window_n=window_n)
        p_face = self._bayes_face_probs(alpha=alpha, window_n=window_n)

        self._md5_for_combo_seed = md5_value or ""
        combo_name, combo_fn = self.combo_strategies[self.current_combo_idx]
        combo, total_sum, p_combo = combo_fn(p_face, label, self.recent_combos, temperature)

        self.pending = {"md5": md5_value or "", "label": label, "probs": probs,
                        "combo": combo, "sum": total_sum, "p_combo": p_combo,
                        "algo": algo_name, "combo_algo": combo_name}
        self._save_temp()
        return self.pending

    def update_algo_stats(self, algo_name: str, win: bool):
        st = self.algo_stats.setdefault(algo_name, {"wins":0,"total":0})
        st["total"]+=1; st["wins"]+=1 if win else 0

    def update_with_real(self, md5_value: str, real_label: Label, dice: List[int]):
        if self.pending and (not md5_value or self.pending["md5"]==md5_value):
            use_label = self.pending["label"]; used_algo = self.pending["algo"]
            used_combo_name = self.pending.get("combo_algo"); used_sum = self.pending.get("sum")
            used_combo = tuple(self.pending.get("combo") or ())
        else:
            used_algo, use_label = self.predict_with_name(md5_value or "")
            used_combo_name=None; used_sum=None; used_combo=()

        win = (use_label==real_label)
        rec = {"dice":dice,"md5":md5_value or "", "guess":use_label, "real":real_label,
               "win":win, "algo":used_algo}
        self.history.append(rec)
        self.update_algo_stats(used_algo, win)

        if self.pending and (not md5_value or self.pending["md5"]==md5_value) and used_combo_name:
            st = self.combo_stats.setdefault(used_combo_name, {"plays":0,"sum_hits":0,"exact_hits":0})
            st["plays"]+=1
            if used_sum==sum(dice): st["sum_hits"]+=1
            if used_combo and used_combo==tuple(dice): st["exact_hits"]+=1

        if win: self.win_streak+=1; self.lose_streak=0
        else:
            self.lose_streak+=1; self.win_streak=0
            if self.lose_streak>=self.switch_on_losses:
                self.current_algo_index=(self.current_algo_index+1)%len(self.algorithms)
                self.lose_streak=0

        if self.pending and (not md5_value or self.pending["md5"]==md5_value):
            self.pending=None

        self._save_temp()
        return rec

    def next_prediction(self, md5_value: str, alpha:int, window_n: Optional[int], temperature: Optional[float]=None)->Dict:
        if not self.pending or (md5_value and self.pending.get("md5")!=md5_value):
            self.create_pending(md5_value, alpha, window_n, temperature)
        probs = self.bayes_group_probs(alpha=alpha, window_n=window_n)
        p_label = probs.get(self.pending["label"],0.0)
        return {"label": self.pending["label"], "probs": probs, "prob_label": p_label,
                "combo": self.pending["combo"], "sum": self.pending["sum"], "p_combo": self.pending["p_combo"],
                "algo": self.pending["algo"], "combo_algo": self.pending.get("combo_algo")}

    def window_win_rate(self, window_n:int)->float:
        r=self._recent(window_n); 
        return 0.0 if not r else sum(1 for x in r if x["win"])/len(r)

    def summary(self)->Dict:
        total=len(self.history); wins=sum(1 for r in self.history if r["win"])
        last_record = self.history[-1] if self.history else None
        per_algo = {k: {"wins":v["wins"],"total":v["total"],
                        "win_rate": (v["wins"]/v["total"] if v["total"] else 0.0)}
                    for k,v in self.algo_stats.items()}
        return {"total_rounds": total, "wins": wins, "win_rate": (wins/total if total else 0.0),
                "current_algo": self.algorithms[self.current_algo_index][0],
                "current_combo_algo": self.combo_strategies[self.current_combo_idx][0],
                "last_record": last_record, "per_algo": per_algo,
                "combo_stats": self.combo_stats}

    # CSV
    def export_algo_stats_csv(self)->str:
        lines=["algo,wins,total,win_rate"]
        for k,v in sorted(self.algo_stats.items()):
            wr = (v["wins"]/v["total"]) if v["total"] else 0.0
            lines.append(f"{k},{v['wins']},{v['total']},{wr:.4f}")
        return "\n".join(lines)
    def export_combo_stats_csv(self)->str:
        lines=["combo_algo,plays,sum_hits,exact_hits,sum_hit_rate,exact_hit_rate"]
        for k,v in sorted(self.combo_stats.items()):
            plays=v.get("plays",0) or 0
            shr=(v.get("sum_hits",0)/plays) if plays else 0.0
            ehr=(v.get("exact_hits",0)/plays) if plays else 0.0
            lines.append(f"{k},{plays},{v.get('sum_hits',0)},{v.get('exact_hits',0)},{shr:.4f},{ehr:.6f}")
        return "\n".join(lines)
