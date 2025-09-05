# main.py
from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from predictor import EnsemblePredictor

app = FastAPI(title="TX Console — Markov+Pattern")

# mỗi client dùng chung một phiên (đơn giản). Có thể thay bằng cookie/session nếu cần.
PRED = EnsemblePredictor()

# ---------- UI ----------
PAGE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Tool soi cầu TX — console UI</title>
<style>
:root{
  --bg:#0b0e14;--panel:#11151c;--ink:#e6e6e6;--muted:#9aa4b2;
  --ok:#22c55e;--bad:#ef4444;--accent:#facc15;--line:#293241;
}
*{box-sizing:border-box}
body{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:radial-gradient(1200px 700px at 80% -10%,#18202b 0%,#0b0e14 60%);color:var(--ink)}
.wrap{max-width:980px;margin:0 auto;padding:18px}
h1{font-size:18px;margin:0 0 10px}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.btn{border:1px solid var(--line);background:#0f131a;color:var(--ink);padding:10px 14px;border-radius:10px;font-weight:700;cursor:pointer}
.btn.primary{background:linear-gradient(90deg,#0ea5e9,#22d3ee);color:#0b0e14;border:0}
.btn.ok{background:linear-gradient(90deg,#22c55e,#86efac);color:#052e12;border:0}
.btn.bad{background:linear-gradient(90deg,#ef4444,#fb7185);color:#2f0707;border:0}
.btn.ghost{background:#10141b}
.grid{display:grid;grid-template-columns:1fr;gap:12px}
.card{background:linear-gradient(180deg,#0f141c,#0b0e14);border:1px solid var(--line);border-radius:14px;padding:14px}
pre{margin:0;white-space:pre-wrap;line-height:1.35}
.kv{display:grid;grid-template-columns:160px 1fr;gap:6px;margin-top:8px}
.badge{display:inline-block;background:#151a23;border:1px solid var(--line);padding:3px 6px;border-radius:8px;font-size:12px;color:var(--muted)}
.hr{height:1px;background:var(--line);margin:8px 0}
.small{color:var(--muted);font-size:12px}
.stat{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.stat .box{background:#0d1118;border:1px solid var(--line);padding:10px;border-radius:12px;text-align:center}
.title{color:#facc15;font-weight:800}
</style>
</head>
<body>
<div class="wrap">
  <h1>📊 Tool soi cầu Tài/Xỉu — <span class="badge">Markov + Pattern + EWMA + Dirichlet + Streak (ensemble)</span></h1>

  <div class="controls">
    <button class="btn ok" onclick="send('T')">+ Tài (T)</button>
    <button class="btn bad" onclick="send('X')">+ Xỉu (X)</button>
    <button class="btn ghost" onclick="undo()">↶ Hoàn tác</button>
    <button class="btn ghost" onclick="resetAll()">⟲ Reset</button>
    <button class="btn primary" onclick="refresh()">Làm mới</button>
  </div>

  <div class="grid">
    <div class="card" id="console"><pre>Đang khởi tạo…</pre></div>
    <div class="card">
      <div class="title">Thống kê nhanh</div>
      <div id="quick" class="stat" style="margin-top:8px"></div>
      <div class="hr"></div>
      <div class="small">* Đây là công cụ thống kê/xác suất. Kết quả game RNG không thể biết trước 100%.</div>
    </div>
  </div>
</div>

<script>
async function api(path, body){
  const opt = body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt);
  return await r.json();
}
async function send(lab){ const d = await api('/update', {r:lab}); render(d); }
async function undo(){ const d = await api('/undo', {}); render(d); }
async function resetAll(){ const d = await api('/reset', {}); render(d); }
async function refresh(){ const d = await api('/state'); render(d); }

function asciiBox(stats){
  const pad = (s,n)=> (s+' '.repeat(n)).slice(0,n);
  const total = stats.total||0, t = stats.tai||0, x = stats.xiu||0;
  const pt = (stats.p_tai||0).toFixed(2).padStart(6,' ');
  const px = (stats.p_xiu||0).toFixed(2).padStart(6,' ');
  return [
    "┌────────┬─────┬─────┬──────────┬──────────┐",
    "│ TỔNG   │  T  │  X  │  % TÀI   │  % XỈU   │",
    "├────────┼─────┼─────┼──────────┼──────────┤",
    `│ ${String(total).padStart(6)} │ ${String(t).padStart(3)} │ ${String(x).padStart(3)} │ ${pt.padStart(8)} │ ${px.padStart(8)} │`,
    "└────────┴─────┴─────┴──────────┴──────────┘",
  ].join("\n");
}

function render(d){
  // console block
  const s = d.stats, p = d.predict, hist = d.history || [];
  const rsn = (p.reasons||[]).join(" • ");
  const conf = (p.confidence*100).toFixed(1);
  const wr20 = (s.wr20||0).toFixed(1);
  const streaks = d.streaks || [];
  let str = "";
  str += "🟡 Đang soi cầu nâng cao (Markov + pattern)…\n\n";
  str += `✅ Dự đoán: ${p.guess}  (độ tin cậy ${conf}%)\n`;
  str += `   ↳ Lý do: ${rsn}\n\n`;
  str += "📘 BẢNG THỐNG KÊ HIỆN TẠI\n";
  str += asciiBox(s) + "\n\n";
  str += `🎯 Win-rate 20 ván gần nhất (ensemble): ${wr20}%\n\n`;
  str += "🔗 CHUỖI GẦN ĐÂY:\n";
  if(streaks.length===0){ str += "  (chưa có dữ liệu)\n"; }
  else{
    for(let i=0;i<streaks.length;i++){
      const it = streaks[i];
      str += `  [${i+1}] ${it[0]} (${it[1]})\n`;
    }
  }
  str += "\n🧾 Lịch sử gần nhất:  " + hist.slice(-40).join(" ") + "\n";
  document.querySelector("#console pre").textContent = str;

  // quick stat boxes
  document.getElementById("quick").innerHTML = `
    <div class="box"><div class="small">Tổng ván</div><div style="font-size:20px;font-weight:800">${s.total||0}</div></div>
    <div class="box"><div class="small">% Tài</div><div style="font-size:20px;font-weight:800">${(s.p_tai||0).toFixed(1)}%</div></div>
    <div class="box"><div class="small">% Xỉu</div><div style="font-size:20px;font-weight:800">${(s.p_xiu||0).toFixed(1)}%</div></div>
    <div class="box"><div class="small">Dự đoán</div><div style="font-size:20px;font-weight:800;color:${p.guess==='T'?'#22c55e':'#ef4444'}">${p.guess}</div></div>
    <div class="box"><div class="small">Tin cậy</div><div style="font-size:20px;font-weight:800">${conf}%</div></div>
  `;
}
refresh();
</script>
</body>
</html>
"""

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(PAGE)

# ---------- API ----------
@app.get("/state")
def state():
    pred = PRED.predict()
    return {
        "history": PRED.history[-200:],
        "stats": PRED.stats(),
        "predict": pred,
        "streaks": PRED.streaks(),
    }

@app.post("/update")
async def update(body: dict):
    r = (body or {}).get("r")
    if r not in ("T", "X"):
        return JSONResponse({"error":"r must be 'T' or 'X'"}, status_code=400)
    PRED.update(r)
    return state()

@app.post("/undo")
async def undo():
    PRED.undo()
    return state()

@app.post("/reset")
async def reset():
    PRED.reset()
    return state()
