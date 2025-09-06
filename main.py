# main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Import predictor an toàn (tránh crash nếu file sai)
try:
    from predictor_baccarat import BacPredictor, PredictResult  # type: ignore
except Exception as e:
    class PredictResult:  # fallback dataclass tối giản
        def __init__(self, side="player", probs=None, reason="fallback"):
            self.side = side
            self.probs = probs or {"player": 0.34, "banker": 0.33, "tie": 0.33}
            self.reason = f"{reason}: {e}"

    class BacPredictor:  # fallback predictor
        def __init__(self): self.err = str(e)
        def reset(self): ...
        def update(self, last=None): ...
        def predict(self): return PredictResult()

app = FastAPI(title="TX / Baccarat Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

PRED = BacPredictor()

# ---------- Helpers ----------
def _valid_side(x: str | None):
    return x in (None, "player", "banker", "tie")

# ---------- Routes ----------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "OK - TX is running. Go to /ui"

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return """
<!doctype html>
<html lang="vi"><meta charset="utf-8"/>
<title>TX — Baccarat Predictor</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:#0b0b0f;color:#eee;margin:0;padding:24px}
  .card{max-width:720px;margin:0 auto;padding:20px;background:#13131a;border:1px solid #262636;border-radius:12px}
  input,select,button{padding:10px 12px;border-radius:8px;border:1px solid #333;background:#1a1a22;color:#fff}
  button{cursor:pointer}
  .row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
  pre{background:#0f0f14;border:1px solid #262636;padding:10px;border-radius:8px;white-space:pre-wrap}
</style>
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
</script>
</html>"""

@app.post("/api/update", response_class=PlainTextResponse)
async def api_update(payload: dict):
    last = payload.get("last", None)
    if not _valid_side(last):
        raise HTTPException(400, "last must be one of ['player','banker','tie'] or null")
    try:
        PRED.update(last)
        return "updated"
    except Exception as e:
        raise HTTPException(500, f"update failed: {e}")

@app.get("/api/predict")
async def api_predict():
    try:
        res = PRED.predict()
        if hasattr(res, "__dict__"):
            res = res.__dict__
        return JSONResponse(res)
    except Exception as e:
        raise HTTPException(500, f"predict failed: {e}")

@app.post("/api/reset", response_class=PlainTextResponse)
async def api_reset():
    try:
        PRED.reset()
        return "reset"
    except Exception as e:
        raise HTTPException(500, f"reset failed: {e}")
