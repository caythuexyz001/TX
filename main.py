# main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Predictor
from predictor_baccarat import BacPredictor, PredictResult

# OpenCV & Vision (an toàn)
try:
    import numpy as np
    import cv2
    from vision_road import RoadExtractor, DEFAULT_CFG, render_debug_png
    OPENCV_OK = True
except Exception as e:
    OPENCV_OK = False
    np = None  # type: ignore
    cv2 = None # type: ignore
    RoadExtractor = None # type: ignore
    DEFAULT_CFG = {"note":"opencv not available"} # type: ignore
    def render_debug_png(*args, **kwargs): return b""  # type: ignore
    print("[WARN] OpenCV/Vision disabled:", e)

app = FastAPI(title="TX / Baccarat Predictor (Vision 4 bảng)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

PRED = BacPredictor()
VISION = RoadExtractor() if OPENCV_OK else None

# --------- Helpers ---------
def _valid_side(x: str | None):
    return x in (None, "player", "banker", "tie")

def _bp2side(bp: str) -> str:
    # map P/B/T thành side
    if bp == "P": return "player"
    if bp == "B": return "banker"
    return "tie"

# ----------------- UI cơ bản -----------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK - TX is running. Go to /ui"

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return """
<!doctype html>
<html lang="vi"><meta charset="utf-8"/>
<title>TX — Vision Baccarat</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0b0b10;color:#eee;margin:0;padding:24px}
  .wrap{max-width:960px;margin:0 auto}
  .card{padding:16px;background:#13131a;border:1px solid #262636;border-radius:12px;margin:0 0 12px}
  input,select,button{padding:10px 12px;border-radius:8px;border:1px solid #333;background:#1a1a22;color:#fff}
  button{cursor:pointer}
  .row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
  pre{background:#0f0f14;border:1px solid #262636;padding:10px;border-radius:8px;white-space:pre-wrap}
  .warn{background:#231f1f;border:1px dashed #664;color:#ffb}
</style>
<div class="wrap">
  <div class="card">
    <h2>TX — Baccarat Predictor</h2>
    <div class="row">
      <select id="last">
        <option value="">(Không nhập ván trước)</option>
        <option value="player">Tay con (Player)</option>
        <option value="banker">Nhà cái (Banker)</option>
        <option value="tie">Hòa (Tie)</option>
      </select>
      <button onclick="doUpdate()">Cập nhật ván trước</button>
      <button onclick="doPredict()">Dự đoán ván kế</button>
      <button onclick="doReset()">Reset</button>
    </div>
    <pre id="out">Kết quả sẽ hiển thị ở đây…</pre>
  </div>

  <div class="card warn" id="ocv_warn" style="display:none">⚠️ Máy chủ không có OpenCV – chức năng đọc ảnh 4 bảng tạm tắt.</div>

  <div class="card" id="vision_block">
    <h3>📷 Đọc ảnh 4 bảng (bead/big/big-eye/small)</h3>
    <div class="row">
      <input id="file" type="file" accept="image/*">
      <label><input id="dbg" type="checkbox"> Debug overlay</label>
    </div>
    <div class="row">
      <textarea id="cfg" style="width:100%;min-height:140px" placeholder='Dán JSON cấu hình ROI/HSV (để trống dùng mặc định)'></textarea>
    </div>
    <div class="row">
      <button onclick="getCfg()">Lấy cấu hình mẫu</button>
      <button onclick="uploadImg('append')">Dò ảnh & Nhập thêm</button>
      <button onclick="uploadImg('replace')">Dò ảnh & Thay thế</button>
    </div>
    <img id="dbgimg" style="max-width:100%;display:none;border:1px solid #333;border-radius:8px"/>
  </div>
</div>
<script>
async function doUpdate(){
  const v = document.getElementById('last').value || null;
  const r = await fetch('/api/update', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({last:v})});
  document.getElementById('out').textContent = await r.text();
}
async function doPredict(){
  const r = await fetch('/api/predict'); const d = await r.json();
  document.getElementById('out').textContent = JSON.stringify(d, null, 2);
}
async function doReset(){
  const r = await fetch('/api/reset', {method:'POST'}); document.getElementById('out').textContent = await r.text();
}
async function getCfg(){
  const d = await fetch('/vision/example_config').then(r=>r.json());
  document.getElementById('cfg').value = JSON.stringify(d, null, 2);
}
async function uploadImg(mode){
  const f = document.getElementById('file').files[0];
  if(!f){ alert('Chọn ảnh trước'); return; }
  const fd = new FormData();
  fd.append('image', f);
  fd.append('mode', mode);
  const cfg = document.getElementById('cfg').value.trim();
  if(cfg) fd.append('cfg_json', cfg);
  fd.append('debug', document.getElementById('dbg').checked ? '1' : '0');
  const r = await fetch('/vision/upload', {method:'POST', body:fd});
  const d = await r.json();
  if(d.error){ alert(d.error); return; }
  document.getElementById('out').textContent = JSON.stringify(d, null, 2);
  if(d.debug_image){
    const img = document.getElementById('dbgimg');
    img.src = d.debug_image; img.style.display='block';
  }
}
(async ()=>{
  const s = await fetch('/state').then(r=>r.json());
  if(s.opencv_ok === false){
    document.getElementById('ocv_warn').style.display='block';
    document.getElementById('vision_block').style.display='none';
  }
})();
</script>
</html>"""

# ---------- Core APIs ----------
@app.get("/state")
def state():
    pr = PRED.predict()
    d = pr.__dict__ if hasattr(pr, "__dict__") else pr
    return {"predict": d, "opencv_ok": OPENCV_OK}

@app.post("/api/update", response_class=PlainTextResponse)
async def api_update(body: dict):
    last = body.get("last", None)
    if not _valid_side(last):
        raise HTTPException(400, "last must be one of ['player','banker','tie'] or null")
    PRED.update(last)
    return "updated"

@app.get("/api/predict")
async def api_predict():
    pr = PRED.predict()
    return JSONResponse(pr.__dict__ if hasattr(pr, "__dict__") else pr)

@app.post("/api/reset", response_class=PlainTextResponse)
async def api_reset():
    PRED.reset()
    return "reset"

# ---------- Vision APIs ----------
@app.get("/vision/example_config")
def vision_example():
    return DEFAULT_CFG

@app.post("/vision/upload")
async def vision_upload(
    image: UploadFile = File(...),
    mode: str = Form("append"),
    cfg_json: str = Form(None),
    debug: str = Form("0"),
):
    if not OPENCV_OK or VISION is None:
        return JSONResponse({"error":"OpenCV không khả dụng trên máy chủ"}, status_code=503)

    try:
        data = await image.read()
        arr = np.frombuffer(data, dtype=np.uint8)    # type: ignore
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)    # type: ignore
        if img is None:
            return JSONResponse({"error":"Không đọc được ảnh"}, status_code=400)

        # set cấu hình nếu có
        if cfg_json:
            import json
            try:
                VISION.set_config(json.loads(cfg_json))
            except Exception:
                return JSONResponse({"error":"Config JSON không hợp lệ"}, status_code=400)

        result = VISION.process(img)                 # type: ignore
        seq = result.get("combined", [])
        if not seq:
            return JSONResponse({"error":"Không nhận diện được chấm — cần chỉnh ROI/HSV"}, status_code=400)

        if mode == "replace":
            PRED.reset()
        # nạp chuỗi B/P vào predictor
        for bp in seq:
            PRED.update(_bp2side(bp))

        resp = {"ok": True, "imported": len(seq), "predict": (PRED.predict().__dict__)}
        resp.update({"vision": {
            "bead": len(result.get("bead", [])),
            "big": len(result.get("big", [])),
            "bigeye": len(result.get("bigeye", [])),
            "small": len(result.get("small", [])),
        }})

        if debug in ("1","true","True"):
            png = render_debug_png(img, result["strip"], result["roi_abs"])  # type: ignore
            import base64
            resp["debug_image"] = "data:image/png;base64," + base64.b64encode(png).decode()

        return resp
    except Exception as e:
        return JSONResponse({"error": f"Lỗi xử lý ảnh: {e}"}, status_code=500)
