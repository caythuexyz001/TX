# main.py
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from predictor_baccarat import BacPredictor
from vision_road import RoadExtractor, DEFAULT_CFG, render_debug_png
import numpy as np
import cv2
import json
import base64

app = FastAPI(title="Baccarat Soi Cầu — Vision + Ensemble")
PRED = BacPredictor()
VISION = RoadExtractor()

# -------- helpers --------
def map_label(token: str):
    t = token.strip().lower()
    if not t: return None
    if t in ("b","banker","nha cai","nhà cái","cai"): return "B"
    if t in ("p","player","nha con","nhà con","con"): return "P"
    if t in ("t","tie","hoa","hoà","hòa"): return "T"
    return None

def parse_bulk(s: str):
    seps = [",",";","|","/","\\","\n","\t"," "]
    for sp in seps[1:]:
        s = s.replace(sp, seps[0])
    toks = [x for x in s.split(seps[0]) if x.strip()!=""]
    out=[]
    for tk in toks:
        lab = map_label(tk)
        if lab in ("B","P","T"): out.append(lab)
    return out

# -------- UI --------
PAGE = r"""<!doctype html><html lang="vi"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Vision Baccarat Soi Cầu</title>
<style>
:root{--bg:#0b0e14;--panel:#11151c;--ink:#e6e6e6;--muted:#97a3b3;--ok:#22c55e;--bad:#ef4444;--tie:#38bdf8;--line:#293241}
*{box-sizing:border-box}body{margin:0;background:#0b0e14;color:var(--ink);font-family:ui-monospace,Consolas,monospace}
.wrap{max-width:1000px;margin:0 auto;padding:16px}
h1{margin:0 0 10px;font-size:18px}
.row{display:flex;gap:8px;flex-wrap:wrap}
.btn{padding:10px 14px;border-radius:12px;border:1px solid var(--line);cursor:pointer;font-weight:800}
.btn.b{background:linear-gradient(90deg,#ef4444,#fb7185);color:#2a0707}
.btn.p{background:linear-gradient(90deg,#22c55e,#86efac);color:#062811}
.btn.t{background:linear-gradient(90deg,#38bdf8,#7dd3fc);color:#061f2a}
.btn.ghost{background:#10141b;color:#e6e6e6}
.card{background:linear-gradient(180deg,#0f141c,#0b0e14);border:1px solid var(--line);border-radius:14px;padding:14px;margin-top:10px}
pre{white-space:pre-wrap;margin:0}
.badge{padding:2px 6px;border:1px solid var(--line);border-radius:8px;color:#9aa4b2;font-size:12px}
input,textarea{width:100%;background:#0b0e14;border:1px solid var(--line);border-radius:10px;color:#e6e6e6;padding:10px}
.kpi{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:8px}
.kpi .box{background:#0d1118;border:1px solid var(--line);border-radius:12px;text-align:center;padding:10px}
.bar{height:10px;border-radius:6px;background:#10141b;overflow:hidden}
.fill{height:100%}
</style></head><body>
<div class="wrap">
  <h1>🎴 Baccarat Soi Cầu — Vision (đọc 4 bảng) + Ensemble</h1>

  <div class="row">
    <button class="btn b" onclick="send('B')">+ Nhà cái (B)</button>
    <button class="btn p" onclick="send('P')">+ Nhà con (P)</button>
    <button class="btn t" onclick="send('T')">+ Hòa (T)</button>
    <button class="btn ghost" onclick="undo()">↶ Hoàn tác</button>
    <button class="btn ghost" onclick="resetAll()">⟲ Reset</button>
    <button class="btn ghost" onclick="refresh()">Làm mới</button>
  </div>

  <div class="card" id="console"><pre>Đang khởi tạo…</pre></div>

  <div class="card">
    <div style="font-weight:800;margin-bottom:6px">📥 Đọc lịch sử từ ẢNH (4 bảng)</div>
    <div class="row">
      <input id="cfg" placeholder="Dán JSON ROI/HSV (để trống dùng mặc định)">
    </div>
    <div class="row" style="margin-top:6px">
      <input id="file" type="file" accept="image/*"/>
      <label style="display:flex;align-items:center;gap:6px">
        <input id="dbg" type="checkbox"> Debug overlay
      </label>
    </div>
    <div class="row" style="margin-top:6px">
      <button class="btn p" onclick="uploadImg('append')">Dò ảnh & Nhập thêm</button>
      <button class="btn b" onclick="uploadImg('replace')">Dò ảnh & Thay thế</button>
      <button class="btn t" onclick="getExample()">Lấy cấu hình mẫu</button>
    </div>
  </div>

  <div class="card">
    <div class="badge">Preview debug</div>
    <img id="debug_img" style="max-width:100%;border-radius:10px;border:1px solid #293241;display:none"/>
  </div>

  <div class="card">
    <div class="row"><textarea id="bulk" placeholder="Nhập nhanh: B,P,P,T,B,..."></textarea></div>
    <div class="row" style="margin-top:6px">
      <button class="btn p" onclick="bulkAppend()">Nhập thêm</button>
      <button class="btn b" onclick="bulkReplace()">Thay thế</button>
      <button class="btn t" onclick="downloadCSV()">Tải CSV</button>
    </div>
  </div>

  <div class="card"><div id="kpi" class="kpi"></div></div>
</div>

<script>
async function call(path, body){
  const opt = body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {};
  const r = await fetch(path, opt); return await r.json();
}
async function send(l){ render(await call('/update',{r:l})); }
async function undo(){ render(await call('/undo',{})); }
async function resetAll(){ render(await call('/reset',{})); }
async function refresh(){ render(await fetch('/state').then(r=>r.json())); }

async function bulkAppend(){ const txt=document.getElementById('bulk').value.trim(); if(!txt) return; render(await call('/bulk',{text:txt,mode:'append'})); }
async function bulkReplace(){ const txt=document.getElementById('bulk').value.trim(); if(!txt) return; render(await call('/bulk',{text:txt,mode:'replace'})); }

async function uploadImg(mode){
  const file = document.getElementById('file').files[0];
  if(!file){ alert('Chọn ảnh trước'); return; }
  const cfg = document.getElementById('cfg').value.trim();
  const dbg = document.getElementById('dbg').checked ? "1" : "0";
  const fd = new FormData();
  fd.append('image', file);
  fd.append('mode', mode);
  fd.append('debug', dbg);
  if(cfg) fd.append('cfg_json', cfg);
  const r = await fetch('/vision/upload', {method:'POST', body:fd});
  const d = await r.json();
  if(d.error){ alert(d.error); return; }
  render(d);
  const img = document.getElementById('debug_img');
  if(d.debug_image){ img.src = d.debug_image; img.style.display = 'block'; }
  else { img.style.display = 'none'; }
}

async function getExample(){
  const d = await fetch('/vision/example_config').then(r=>r.json());
  document.getElementById('cfg').value = JSON.stringify(d);
}

async function downloadCSV(){
  const r = await fetch('/export_csv'); const blob = await r.blob();
  const url = URL.createObjectURL(blob); const a=document.createElement('a');
  a.href=url; a.download='baccarat_report.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}

function bar(p,c){ return `<div class="bar"><div class="fill" style="width:${(p*100).toFixed(1)}%;background:${c}"></div></div>`; }
function box(t,v,u=""){ return `<div class="box"><div class="badge">${t}</div><div style="font-size:22px;font-weight:800">${v}${u}</div></div>`; }

function render(d){
  const s=d.stats, p=d.predict, hist=d.history||[], st=d.streaks||[], pr=p.probs||{B:0.33,P:0.33,T:0.33};
  let txt="";
  txt += "🟡 Soi cầu Baccarat (Vision + Ensemble)…\n\n";
  txt += `✅ Dự đoán: ${p.guess} • Tin cậy ${(p.confidence*100).toFixed(1)}%\n`;
  txt += `   ↳ Lý do: ${(p.reasons||[]).join(" • ")}\n\n`;
  txt += `- Tổng: ${s.total} | B:${s.banker} P:${s.player} T:${s.tie}\n`;
  txt += `- Tỉ lệ: B ${s.p_b.toFixed(1)}% • P ${s.p_p.toFixed(1)}% • T ${s.p_t.toFixed(1)}%\n`;
  txt += `- WR 20: ${s.wr20.toFixed(1)}%\n\n`;
  txt += "🔗 Chuỗi gần đây:\n";
  if(st.length===0) txt+="  (chưa có)\n"; else st.forEach((x,i)=>txt+=`  [${i+1}] ${x[0]} (${x[1]})\n`);
  txt += "\n🧾 Lịch sử: "+hist.slice(-60).join(" ")+"\n";
  document.querySelector("#console pre").textContent = txt;

  document.getElementById('kpi').innerHTML = `
    ${box('Xác suất B', (pr.B*100).toFixed(1),'%') + bar(pr.B,'#fb7185')}
    ${box('Xác suất P', (pr.P*100).toFixed(1),'%') + bar(pr.P,'#86efac')}
    ${box('Xác suất T', (pr.T*100).toFixed(1),'%') + bar(pr.T,'#7dd3fc')}
    ${box('Tổng ván', s.total)}
    ${box('Dự đoán', p.guess)}
    ${box('Tin cậy', (p.confidence*100).toFixed(1),'%')}
  `;
}
refresh();
</script></body></html>
"""

@app.get("/ui", response_class=HTMLResponse)
def ui(): return HTMLResponse(PAGE)

# -------- core APIs --------
@app.get("/state")
def state():
    pred = PRED.predict()
    return {"history": PRED.H[-300:], "stats": PRED.stats(), "predict": pred, "streaks": PRED.streaks()}

@app.post("/update")
def update(body: dict):
    lab = (body or {}).get("r")
    if lab not in ("B","P","T"): return JSONResponse({"error":"r must be B/P/T"}, status_code=400)
    PRED.update(lab); return state()

@app.post("/undo")
def undo(): PRED.undo(); return state()

@app.post("/reset")
def reset(): PRED.reset(); return state()

@app.post("/bulk")
def bulk(body: dict):
    text = (body or {}).get("text","")
    mode = (body or {}).get("mode","append")
    seq = parse_bulk(text)
    if mode == "replace": PRED.reset()
    for lab in seq: PRED.update(lab)
    return state()

@app.get("/export_csv")
def export_csv():
    H = PRED.H[:]
    rows = ["index,true,probB,probP,probT,guess,correct,cum_acc"]
    if not H:
        return PlainTextResponse("\n".join(rows), media_type="text/csv",
                                 headers={"Content-Disposition":"attachment; filename=baccarat_report.csv"})
    wins=total=0
    tmp = BacPredictor()
    for i,true_lab in enumerate(H, start=1):
        pred = tmp.predict()
        pB,pP,pT = pred["probs"]["B"], pred["probs"]["P"], pred["probs"]["T"]
        guess = pred["guess"]; correct = 1 if guess==true_lab else 0
        total += 1; wins += correct
        rows.append(f"{i},{true_lab},{pB:.6f},{pP:.6f},{pT:.6f},{guess},{correct},{wins/total:.6f}")
        tmp.update(true_lab)
    csv = "\n".join(rows)
    return PlainTextResponse(csv, media_type="text/csv",
                             headers={"Content-Disposition":"attachment; filename=baccarat_report.csv"})

# -------- VISION APIs --------
@app.get("/vision/example_config")
def vision_example():
    return DEFAULT_CFG

@app.post("/vision/upload")
async def vision_upload(image: UploadFile = File(...),
                        mode: str = Form("append"),
                        cfg_json: str = Form(None),
                        debug: str = Form("0")):
    try:
        data = await image.read()
        file_bytes = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"error":"Không đọc được ảnh"}, status_code=400)

        if cfg_json:
            try:
                cfg = json.loads(cfg_json)
                VISION.set_config(cfg)
            except Exception:
                return JSONResponse({"error":"Config JSON không hợp lệ"}, status_code=400)

        result = VISION.process(img)
        seq = result.get("combined", [])
        if not seq:
            return JSONResponse({"error":"Không nhận diện được chấm ở các bảng. Hãy chỉnh ROI/HSV."}, status_code=400)

        if mode == "replace":
            PRED.reset()
        for lab in seq:
            if lab in ("B","P","T"):
                PRED.update(lab)

        resp = state()
        resp["vision"] = {
            "bead": len(result["bead"]),
            "big": len(result["big"]),
            "bigeye": len(result["bigeye"]),
            "small": len(result["small"]),
            "strip": result["strip"],
        }

        if debug in ("1","true","True"):
            png_bytes = render_debug_png(img, result["strip"], result["roi_abs"])
            b64 = base64.b64encode(png_bytes).decode("ascii")
            resp["debug_image"] = f"data:image/png;base64,{b64}"

        return resp
    except Exception as e:
        return JSONResponse({"error": f"Lỗi xử lý ảnh: {e}"}, status_code=500)
