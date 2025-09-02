# main.py
from __future__ import annotations
import time, uuid
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from predictor import TaiXiuAI

# ================== CẤU HÌNH ==================
ADMIN_API_KEY = "01279992564Aa!"       # đổi nếu muốn
PER_IP_LIMIT  = (40, 60)                # 40 req / 60s / IP
GLOBAL_LIMIT  = (600, 60)               # toàn server
SESSION_IDLE_TTL = 60*60*12
SESSION_MAX   = 800

app = FastAPI(title="TaiXiu AI", docs_url=None, redoc_url=None, openapi_url=None)

_ip_hits: Dict[str, Deque[float]] = defaultdict(deque)
_global_hits: Deque[float] = deque()

# ================== RATE LIMIT + SECURITY HEADERS ==================
def rate_limiter(ip: str):
    now = time.time()
    per, win = PER_IP_LIMIT
    dq = _ip_hits[ip]
    while dq and now - dq[0] > win:
        dq.popleft()
    if len(dq) >= per:
        return JSONResponse({"detail": "Too Many Requests"}, status_code=429)
    dq.append(now)

    gper, gwin = GLOBAL_LIMIT
    while _global_hits and now - _global_hits[0] > gwin:
        _global_hits.popleft()
    if len(_global_hits) >= gper:
        return JSONResponse({"detail": "Server Busy"}, status_code=429)
    _global_hits.append(now)

@app.middleware("http")
async def sec_limit(request: Request, call_next):
    headers = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "same-origin",
        "Cache-Control": "no-store",
    }
    ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown").split(",")[0].strip()
    if request.url.path not in ("/ui", "/health"):
        rl = rate_limiter(ip)
        if rl:
            for k, v in headers.items(): rl.headers[k] = v
            return rl
    resp = await call_next(request)
    for k, v in headers.items(): resp.headers[k] = v
    return resp

def verify_admin(x_api_key: str = Header(None)):
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ================== SESSION ==================
class SessionItem:
    def __init__(self):
        self.ai = TaiXiuAI()
        self.last = time.time()
        # bankroll + fibonacci
        self.bank_vnd: int = 0
        self.fib_base: int = 0
        self.fib_max: int = 12
        self.fib_idx: int = 0
        self.fib_seq: List[int] = [1,1]
        self.last_bet_vnd: int = 0

    def ensure_seq(self):
        self.fib_seq = [1,1]
        while len(self.fib_seq) < max(2, self.fib_max):
            self.fib_seq.append(self.fib_seq[-1] + self.fib_seq[-2])

    def next_fib_bet(self) -> int:
        if self.fib_base <= 0 or self.bank_vnd <= 0: return 0
        self.ensure_seq()
        idx = max(0, min(self.fib_idx, len(self.fib_seq)-1))
        raw = self.fib_seq[idx] * self.fib_base
        cap = int(max(0, self.bank_vnd) * 0.2)  # không quá 20% vốn
        bet = min(raw, cap if cap>0 else raw)
        return (bet // 1000) * 1000

    def apply_round_result(self, win: bool):
        bet = self.last_bet_vnd
        if bet>0:
            if win:
                self.bank_vnd += bet
                self.fib_idx = max(0, self.fib_idx-2)
            else:
                self.bank_vnd -= bet
                self.fib_idx = min(self.fib_idx+1, self.fib_max-1)
        if self.bank_vnd <= 0:
            self.bank_vnd = 0
            self.fib_idx = 0

SESSIONS: Dict[str, SessionItem] = {}

def _gc_sessions():
    now = time.time()
    for sid in list(SESSIONS.keys()):
        if now - SESSIONS[sid].last > SESSION_IDLE_TTL:
            try: SESSIONS[sid].ai.cleanup()
            except Exception: ...
            SESSIONS.pop(sid, None)
    if len(SESSIONS) > SESSION_MAX:
        for sid,_ in sorted(SESSIONS.items(), key=lambda kv: kv[1].last)[:len(SESSIONS)-SESSION_MAX]:
            try: SESSIONS[sid].ai.cleanup()
            except Exception: ...
            SESSIONS.pop(sid, None)

def get_ai(request: Request, response: Optional[HTMLResponse]=None) -> TaiXiuAI:
    _gc_sessions()
    sid = request.cookies.get("txsid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        SESSIONS[sid] = SessionItem()
        if response is not None:
            response.set_cookie("txsid", sid, httponly=True, samesite="Lax", secure=False)
    SESSIONS[sid].last = time.time()
    return SESSIONS[sid].ai

def get_sess(request: Request) -> SessionItem:
    return SESSIONS[request.cookies.get("txsid")]

# ================== SCHEMAS ==================
class Step1Body(BaseModel):
    prev_dice: Optional[List[int]] = None
    prev_md5: Optional[str] = None
    next_md5: Optional[str] = None
    alpha: int = Field(default=1, ge=1)
    window_n: Optional[int] = None
    temperature: Optional[float] = Field(default=0.85, ge=0.1, le=3.0)
    bankroll_vnd: Optional[int] = None
    fib_base_vnd: Optional[int] = None
    fib_max_step: Optional[int] = Field(default=12, ge=2, le=30)

class Step2Body(BaseModel):
    md5: Optional[str] = None
    real: Optional[str] = None
    dice: Optional[List[int]] = None

class AlgoSelectBody(BaseModel):
    name: str

class ComboSelectBody(BaseModel):
    name: str

# ================== HELPERS ==================
def outcome_from_dice(d: List[int]) -> str:
    if len(d)==3 and d[0]==d[1]==d[2]: return "Triple"
    return "Xỉu" if sum(d)<=10 else "Tài"

# ================== PUBLIC API ==================
public = APIRouter()

@public.get("/health")
def health(): return {"ok": True}

@public.get("/algo_list")
def algo_list(request: Request):
    ai = get_ai(request)
    return {"algorithms": [n for n,_ in ai.algorithms]}

# ---- KHÓA về MIX (bỏ qua lựa chọn) ----
@public.post("/set_algo")
def set_algo(body: AlgoSelectBody, request: Request):
    ai = get_ai(request)
    ai.current_algo_index = next((i for i,(n,_) in enumerate(ai.algorithms) if n=="algo_mix"), ai.current_algo_index)
    return {"ok": True, "current_algo": "algo_mix"}

@public.get("/algo_stats")
def algo_stats(request: Request):
    ai = get_ai(request); return ai.summary().get("per_algo")

@public.get("/export_stats")
def export_stats(request: Request):
    ai = get_ai(request); csv = ai.export_algo_stats_csv()
    return PlainTextResponse(csv, media_type="text/csv")

@public.get("/combo_list")
def combo_list(request: Request):
    ai = get_ai(request)
    return {"combo_algorithms": [n for n,_ in ai.combo_strategies]}

@public.post("/set_combo")
def set_combo(body: ComboSelectBody, request: Request):
    ai = get_ai(request)
    ai.current_combo_idx = next((i for i,(n,_) in enumerate(ai.combo_strategies) if n=="combo_mix"), ai.current_combo_idx)
    return {"ok": True, "current_combo_algo": "combo_mix"}

@public.get("/combo_stats")
def combo_stats(request: Request):
    ai = get_ai(request); return ai.combo_stats

@public.get("/export_combo_stats")
def export_combo_stats(request: Request):
    ai = get_ai(request); csv = ai.export_combo_stats_csv()
    return PlainTextResponse(csv, media_type="text/csv")

@public.post("/step1_predict")
def step1_predict(body: Step1Body, request: Request):
    ai   = get_ai(request)
    sess = get_sess(request)

    # bankroll + fib
    if body.bankroll_vnd is not None: sess.bank_vnd  = max(0, int(body.bankroll_vnd))
    if body.fib_base_vnd is not None: sess.fib_base  = max(0, int(body.fib_base_vnd))
    if body.fib_max_step is not None: sess.fib_max   = int(body.fib_max_step)

    # nếu có dữ liệu ván trước thì cập nhật trước
    if body.prev_dice:
        ai.update_with_real(body.prev_md5 or "", outcome_from_dice(body.prev_dice), body.prev_dice)

    out = ai.next_prediction(md5_value=body.next_md5 or "",
                             alpha=body.alpha, window_n=body.window_n, temperature=body.temperature)

    next_bet = sess.next_fib_bet()
    sess.last_bet_vnd = next_bet

    sumy = ai.summary()
    return {
        "next_guess": out["label"],
        "prob": out["prob_label"],
        "combo": out["combo"],
        "sum": out["sum"],
        "p_combo": out["p_combo"],
        "algo": out["algo"],
        "combo_algo": out.get("combo_algo"),
        "summary": sumy,
        "fib": {
            "bankroll_vnd": sess.bank_vnd,
            "base_vnd": sess.fib_base,
            "idx": sess.fib_idx,
            "max": sess.fib_max,
            "next_bet_vnd": next_bet,
            "seq_preview": _seq_preview(sess),
        },
        "echo": {"promote_next_md5_to_prev": body.next_md5 or ""},
    }

def _seq_preview(sess: SessionItem):
    sess.ensure_seq()
    start = max(0, sess.fib_idx); end = min(start+8, len(sess.fib_seq))
    return [x * sess.fib_base for x in sess.fib_seq[start:end]]

@public.post("/step2_confirm")
def step2_confirm(body: Step2Body, request: Request):
    ai   = get_ai(request)
    sess = get_sess(request)
    if body.dice:
        real = outcome_from_dice(body.dice); dice = body.dice
    elif body.real:
        real = body.real; dice = None
    else:
        return {"error":"Cần real hoặc dice"}

    rec = ai.update_with_real(body.md5 or "", real, dice or [0,0,0])
    sess.apply_round_result(rec["win"])
    return {
        "record": rec,
        "summary": ai.summary(),
        "fib": {"bankroll_vnd": sess.bank_vnd, "idx": sess.fib_idx,
                "next_bet_vnd": sess.next_fib_bet(), "seq_preview": _seq_preview(sess)}
    }

@public.get("/summary")
def summary(request: Request): return get_ai(request).summary()

app.include_router(public)

# ================== ADMIN ==================
admin = APIRouter(dependencies=[Depends(verify_admin)])

@admin.post("/reset")
def reset_all(request: Request):
    ai = get_ai(request); ai.reset()
    # khoá lại về MIX
    ai.current_algo_index = next((i for i,(n,_) in enumerate(ai.algorithms) if n=="algo_mix"), 0)
    ai.current_combo_idx  = next((i for i,(n,_) in enumerate(ai.combo_strategies) if n=="combo_mix"), 0)
    sess = get_sess(request)
    sess.bank_vnd = sess.fib_base = sess.fib_idx = sess.last_bet_vnd = 0
    return {"ok": True}

app.include_router(admin)

# ================== UI ==================
PAGE = r"""
<!doctype html><html lang="vi"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"/>
<title>TaiXiu AI — #tool make by Boring</title>
<style>
:root{--bg:#13070a;--bg2:#2a0f13;--card:#1b0c0f;--txt:#fff3cc;--muted:#ffddb0;--gold:#ffd54a;--accent:#e11d2a;--accent2:#ff3838;--accent3:#f5d200}
*{box-sizing:border-box} html,body{height:100%}
body{margin:0;color:var(--txt);background:radial-gradient(1200px 700px at 70% -10%,#3a0f14 0%,var(--bg2) 30%,var(--bg) 70%) fixed;
font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:12px 16px 88px}
.header{position:sticky;top:0;z-index:10;backdrop-filter:blur(10px);
background:linear-gradient(180deg,rgba(19,7,10,.9),rgba(19,7,10,.6));display:flex;justify-content:space-between;align-items:center;padding:10px 16px;border-bottom:1px solid rgba(255,213,74,.15)}
.logo{font-weight:900;font-size:18px;letter-spacing:.2px}
.badge{background:linear-gradient(90deg,#f5d200,#ffd54a);-webkit-background-clip:text;background-clip:text;color:transparent}
.card{background:linear-gradient(180deg,#1f0e12,var(--card));border:1px solid rgba(255,213,74,.35);border-radius:16px;padding:16px;box-shadow:0 12px 40px rgba(225,29,42,.25), inset 0 0 0 1px rgba(255,255,255,.02);margin:12px 0}
h2{margin:0 0 8px 0;font-size:16px}
label{display:block;margin:8px 0 6px;color:var(--muted);font-size:13px}
input[type=text],input[type=number],select{width:100%;padding:12px 14px;border-radius:12px;border:1px solid rgba(255,213,74,.35);background:#14090c;color:#fff3cc}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.row2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.btn{padding:12px 16px;border:0;border-radius:12px;cursor:pointer;font-weight:700;touch-action:manipulation}
.btn-primary{background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3));color:#2d0b0f}
.btn-ghost{background:#170b0e;border:1px solid rgba(255,213,74,.35);color:#ffe7ad}
.btn-danger{background:linear-gradient(90deg,#a8071a,#e11d2a);color:#ffe7ad}
.small{font-size:12px;color:#ffdca3}.out{margin-top:12px;padding:12px 14px;background:#150a0d;border:1px solid rgba(255,213,74,.35);border-radius:12px}
.hidden{display:none}.vnd{font-variant-numeric:tabular-nums}
@media (max-width: 640px){.wrap{padding-bottom:110px}.row{grid-template-columns:1fr}.row2{grid-template-columns:1fr}.actions{grid-template-columns:1fr}.logo{font-size:16px}}
.fab{position:fixed;right:16px;bottom:20px;z-index:15;width:56px;height:56px;border-radius:50%;display:flex;align-items:center;justify-content:center;border:1px solid rgba(255,213,74,.35);color:#ffe7ad;background:#1b0c0f;box-shadow:0 10px 30px rgba(0,0,0,.45);transition:transform .18s ease, box-shadow .18s ease}
.fab:active{transform:scale(.97);box-shadow:0 6px 16px rgba(0,0,0,.5)} .fab svg{width:22px;height:22px}
.sheet-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.35);backdrop-filter:blur(2px);opacity:0;pointer-events:none;transition:opacity .25s ease;z-index:19}
.sheet{position:fixed;left:0;right:0;bottom:0;z-index:20;border-radius:16px 16px 0 0;transform:translateY(100%);transition:transform .28s ease;border:1px solid rgba(255,213,74,.35);background:linear-gradient(180deg,#1f0e12,#0f0709)}
.sheet .sheet-handle{width:48px;height:5px;border-radius:999px;background:rgba(255,213,74,.35);margin:10px auto}
.sheet .sheet-body{padding:10px 14px 18px}.menu-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.menu-grid button{padding:12px 14px;border-radius:12px;border:1px solid rgba(255,213,74,.35);background:#170b0e;color:#ffe7ad;font-weight:700}
.menu-grid button.danger{background:linear-gradient(90deg,#a8071a,#e11d2a);color:#ffe7ad}
.sheet-open .sheet{transform:none}.sheet-open .sheet-backdrop{opacity:1;pointer-events:auto}
</style>
</head>
<body>
<div class="header">
  <div class="logo">TaiXiu AI <span class="badge">#tool make by Boring</span></div>
  <button class="btn btn-ghost" style="padding:8px 10px" onclick="openMenu()">☰</button>
</div>

<div class="wrap">
  <div class="card" style="display:flex;align-items:center;gap:10px;background:linear-gradient(90deg,#2a0f13,#3a1418)">
    <div style="font-weight:800">Chào mừng Quốc khánh 2/9</div>
    <div class="small">Chúc may mắn – tinh thần rực lửa 🇻🇳</div>
  </div>

  <div id="step1" class="card">
    <h2>B1. Dự đoán ván KẾ (tạo pending)</h2>
    <div class="row">
      <div><label>Dice 1 (ván trước)</label><input id="s1-d1" type="number" min="1" max="6"></div>
      <div><label>Dice 2 (ván trước)</label><input id="s1-d2" type="number" min="1" max="6"></div>
      <div><label>Dice 3 (ván trước)</label><input id="s1-d3" type="number" min="1" max="6"></div>
    </div>
    <label>MD5 ván trước (tuỳ chọn)</label><input id="s1-prev-md5" type="text">
    <label>MD5 ván kế (tuỳ chọn)</label><input id="s1-next-md5" type="text">
    <div class="row">
      <div><label>Alpha (>=1)</label><input id="s1-alpha" type="number" min="1" value="1"></div>
      <div><label>Window N</label><input id="s1-window" type="number" min="1" placeholder="vd 20"></div>
      <div><label>Temperature</label><input id="s1-temp" type="number" step="0.05" min="0.1" max="3.0" value="0.85"></div>
    </div>
    <div class="row2" style="margin-top:6px">
      <div><label>Thuật toán (cố định)</label><select id="algo-select" disabled><option>algo_mix</option></select></div>
      <div><label>Combo (cố định)</label><select id="combo-select" disabled><option>combo_mix</option></select></div>
    </div>
    <div class="row">
      <div><label>Vốn hiện có (VND)</label><input id="bankroll" type="number" min="0" step="1000" value="0"></div>
      <div><label>Đơn vị Fibo (VND)</label><input id="fib-base" type="number" min="0" step="1000" value="0" placeholder="gợi ý: 1% vốn"></div>
      <div><label>Tối đa bước Fibo</label><input id="fib-max" type="number" min="2" max="30" value="12"></div>
    </div>
    <div class="actions">
      <button class="btn btn-primary" onclick="doStep1()">Cập nhật & Dự đoán</button>
      <button class="btn btn-ghost" onclick="clearStep1()">Xoá ô nhập</button>
    </div>
    <div id="s1-out" class="out small"></div>
  </div>

  <div id="step2" class="card hidden">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h2>B2. Xác nhận kết quả thực</h2>
      <button class="btn btn-ghost" onclick="backToStep1()">← Quay về B1</button>
    </div>
    <label>MD5 ván đã dự đoán</label><input id="s2-md5" type="text">
    <div class="row">
      <div><label>Dice 1</label><input id="s2-d1" type="number" min="1" max="6"></div>
      <div><label>Dice 2</label><input id="s2-d2" type="number" min="1" max="6"></div>
      <div><label>Dice 3</label><input id="s2-d3" type="number" min="1" max="6"></div>
    </div>
    <div class="actions" style="grid-template-columns:repeat(3,1fr)">
      <button id="btn-real-tai" class="btn btn-ghost" onclick="confirmResult('Tài')">Thực tế: Tài</button>
      <button id="btn-real-xiu" class="btn btn-ghost" onclick="confirmResult('Xỉu')">Thực tế: Xỉu</button>
      <button id="btn-real-tri" class="btn btn-ghost" onclick="confirmResult('Triple')">Thực tế: Triple</button>
    </div>
    <button id="btn-confirm-dice" class="btn btn-primary" onclick="confirmDice()">Xác nhận theo DICE</button>
    <div id="s2-locked" class="out small hidden">Đã xác nhận. Quay về B1 để nhập ván mới.</div>
    <div id="s2-out" class="out small"></div>
  </div>
</div>

<button class="fab" aria-label="Menu" onclick="openMenu()">
  <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 6h18v2H3zm0 5h18v2H3zm0 5h18v2H3z"/></svg>
</button>
<div id="backdrop" class="sheet-backdrop" onclick="closeMenu()"></div>
<div id="sheet" class="sheet" role="dialog" aria-modal="true" aria-label="Menu">
  <div class="sheet-handle"></div>
  <div class="sheet-body">
    <div class="menu-grid">
      <button onclick="copyLink()">Copy link</button>
      <button onclick="downloadCSV()">Tải thuật toán CSV</button>
      <button onclick="downloadComboCSV()">Tải combo CSV</button>
      <button onclick="loadSummary()">Làm mới</button>
      <button class="danger" onclick="doReset()">Reset (admin)</button>
      <button onclick="closeMenu()">Đóng</button>
    </div>
  </div>
</div>

<div id="toast" style="position:fixed;right:14px;bottom:88px;background:#111827;color:#e5e7eb;border:1px solid #374151;padding:10px 12px;border-radius:10px;opacity:0;transform:translateY(8px);transition:all .3s;z-index:30"></div>

<script>
function g(id){return document.getElementById(id)}
function openMenu(){document.body.classList.add('sheet-open')}
function closeMenu(){document.body.classList.remove('sheet-open')}
function valInt(id){const v=g(id).value.trim();return v===''?null:parseInt(v,10)}
function valStr(id){const v=g(id).value.trim();return v===''?null:v}
function toast(msg){const t=g('toast');t.textContent=msg;t.style.opacity=1;t.style.transform='translateY(0)';setTimeout(()=>{t.style.opacity=0;t.style.transform='translateY(8px)'},1500)}
function swap(hideId,showId){g(hideId).classList.add('hidden');g(showId).classList.remove('hidden')}
function clearStep1(){['s1-d1','s1-d2','s1-d3','s1-next-md5'].forEach(id=>g(id).value='')}
function lockStep2(){g('s2-locked').classList.remove('hidden');['s2-md5','s2-d1','s2-d2','s2-d3','btn-real-tai','btn-real-xiu','btn-real-tri','btn-confirm-dice'].forEach(id=>g(id).setAttribute('disabled','disabled'))}
function unlockStep2(){g('s2-locked').classList.add('hidden');['s2-md5','s2-d1','s2-d2','s2-d3','btn-real-tai','btn-real-xiu','btn-real-tri','btn-confirm-dice'].forEach(id=>g(id).removeAttribute('disabled'))}

function formatVND(n){return (n||0).toLocaleString('vi-VN')+' đ'}

function render(whereId, roundData, nextData){
  const pct=(nextData.prob*100).toFixed(2), combo=nextData.combo?nextData.combo.join('-'):'—', pcombo=(nextData.p_combo*100).toFixed(2);
  const wrAll=(nextData.summary.win_rate*100).toFixed(1);
  const fib = nextData.fib||{}; const seq=(fib.seq_preview||[]).map(x=>formatVND(x)).join(' → ')||'—';
  g(whereId).innerHTML = `
    <div class="out"><div style="font-weight:700;margin-bottom:6px">Ván vừa rồi</div>
      <div>Real: <b>${roundData.record?.real??'—'}</b> • Guess: <b>${roundData.record?.guess??'—'}</b> → ${roundData.record?.win===true?'<b style="color:#10b981">WIN</b>':roundData.record?.win===false?'<b style="color:#ef4444">LOSE</b>':'—'}</div>
      <div class="small">Tổng ván: ${nextData.summary.total_rounds} • Win% all: ${wrAll}% • Algo: ${nextData.summary.current_algo}</div>
    </div>
    <div class="out"><div style="font-weight:700;margin-bottom:6px">Dự đoán ván KẾ</div>
      <div><b>${nextData.next_guess}</b> — <b>${pct}%</b> • Combo: <b>${combo}</b> (sum=${nextData.sum}, p=${pcombo}%)</div>
      <div class="small">Combo strategy: ${nextData.summary.current_combo_algo}</div>
    </div>
    <div class="out"><div style="font-weight:700;margin-bottom:6px">Gợi ý tiền cược (Fibonacci – VND)</div>
      <div>Vốn: <b class="vnd">${formatVND(fib.bankroll_vnd||0)}</b> • Đơn vị: <b class="vnd">${formatVND(fib.base_vnd||0)}</b> • Bước hiện tại: <b>${(fib.idx??0)}/${(fib.max??12)}</b></div>
      <div>Gợi ý cược ván tới: <b class="vnd">${formatVND(fib.next_bet_vnd||0)}</b></div>
      <div class="small">Chuỗi tiếp theo: ${seq}</div>
    </div>`;
}

const uiState={alpha:1,windowN:null,temp:0.85};
let s2Busy=false;
const lastRoundForB1={prevDice:null, prevMd5:null};

async function doStep1(){
  const prevDice=(valInt('s1-d1')&&valInt('s1-d2')&&valInt('s1-d3'))?[valInt('s1-d1'),valInt('s1-d2'),valInt('s1-d3')]:null;
  uiState.alpha=parseInt(g('s1-alpha').value||'1',10);
  uiState.windowN=valInt('s1-window');
  uiState.temp=parseFloat((g('s1-temp').value||'0.85'));

  const body={
    prev_dice: prevDice,
    prev_md5: g('s1-prev-md5').value || null,
    next_md5: g('s1-next-md5').value || null,
    alpha: uiState.alpha, window_n: uiState.windowN, temperature: uiState.temp,
    bankroll_vnd: valInt('bankroll'), fib_base_vnd: valInt('fib-base'), fib_max_step: valInt('fib-max')
  };

  const res=await fetch('/step1_predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();

  if(data.echo&&data.echo.promote_next_md5_to_prev){ g('s2-md5').value=data.echo.promote_next_md5_to_prev; }
  clearStep1();

  let roundData={record:data.summary.last_record, summary:data.summary};
  if(!roundData.record){ roundData={record:{}, summary:data.summary}; }
  render('s2-out', roundData, data);
  swap('step1','step2'); unlockStep2(); toast('Đã tạo dự đoán, sang B2 để xác nhận');
}

async function confirmResult(label){
  if(s2Busy) return; s2Busy=true;
  const md5=g('s2-md5').value || null;

  const recRes=await fetch('/step2_confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({md5, real:label})});
  const recData=await recRes.json();

  lastRoundForB1.prevMd5 = md5 || recData.record?.md5 || null;
  const rdDice = recData.record?.dice;
  if (Array.isArray(rdDice) && rdDice.length===3 && rdDice.every(n=>Number(n)>0)) {
    lastRoundForB1.prevDice = rdDice.slice();
  }

  const pred=await fetch('/step1_predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alpha:uiState.alpha, window_n:uiState.windowN, temperature:uiState.temp})});
  const next=await pred.json();

  render('s2-out', recData, next);
  lockStep2(); toast('Đã cập nhật & tạo dự đoán mới'); s2Busy=false;
}

async function confirmDice(){
  if(s2Busy) return; s2Busy=true;
  const d1=valInt('s2-d1'),d2=valInt('s2-d2'),d3=valInt('s2-d3');
  if(!(d1&&d2&&d3)){toast('Nhập đủ 3 xúc xắc'); s2Busy=false; return;}
  const md5=g('s2-md5').value || null;

  const recRes=await fetch('/step2_confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({md5, dice:[d1,d2,d3]})});
  const recData=await recRes.json();

  lastRoundForB1.prevMd5 = md5 || recData.record?.md5 || null;
  lastRoundForB1.prevDice = [d1,d2,d3];

  const pred=await fetch('/step1_predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({alpha:uiState.alpha, window_n:uiState.windowN, temperature:uiState.temp})});
  const next=await pred.json();

  ['s2-d1','s2-d2','s2-d3'].forEach(id=>g(id).value='');
  render('s2-out', recData, next);
  lockStep2(); toast('Đã cập nhật & tạo dự đoán mới'); s2Busy=false;
}

function backToStep1(){
  if (lastRoundForB1.prevMd5) g('s1-prev-md5').value = lastRoundForB1.prevMd5;
  if (Array.isArray(lastRoundForB1.prevDice) && lastRoundForB1.prevDice.length===3){
    const [a,b,c]=lastRoundForB1.prevDice;
    g('s1-d1').value=a; g('s1-d2').value=b; g('s1-d3').value=c;
  }
  swap('step2','step1'); g('s1-next-md5').focus();
}

async function loadSummary(){const r=await fetch('/summary');const s=await r.json();toast(`Tổng ván: ${s.total_rounds} • Win%: ${(s.win_rate*100).toFixed(1)}% • Algo: ${s.current_algo}`)}
async function downloadCSV(){const r=await fetch('/export_stats');const t=await r.text();const b=new Blob([t],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='algo_stats.csv';document.body.appendChild(a);a.click();a.remove()}
async function downloadComboCSV(){const r=await fetch('/export_combo_stats');const t=await r.text();const b=new Blob([t],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='combo_stats.csv';document.body.appendChild(a);a.click();a.remove()}
function doReset(){const k=prompt('Nhập ADMIN API key để reset:'); if(!k) return; fetch('/reset',{method:'POST',headers:{'X-API-Key':k}}).then(r=>{toast(r.ok?'Đã reset':'Sai key'); closeMenu();})}
</script>
</body></html>
"""

@app.get("/")
def root():
    return {"ok": True, "msg": "Open /ui for interface"}

@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    resp = HTMLResponse(content=PAGE)
    _ = get_ai(request, resp)  # tạo session + cookie
    return resp
