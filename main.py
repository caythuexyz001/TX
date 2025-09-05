from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from predictor import MarkovPredictor

app = FastAPI()
predictor = MarkovPredictor()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
      <title>Tool Tài Xỉu - Markov</title>
      <style>
        body { font-family: monospace; background: #111; color: #eee; padding:20px; }
        button { margin:5px; padding:8px 16px; font-weight:bold; }
        .win { color: #4ade80; }
        .lose { color: #f87171; }
        .box { border:1px solid #555; padding:15px; margin:10px 0; background:#222; }
      </style>
    </head>
    <body>
      <h2>📊 Tool soi cầu Tài Xỉu (Markov + Pattern)</h2>
      <div>
        <button onclick="sendResult('T')">+ Tài (T)</button>
        <button onclick="sendResult('X')">+ Xỉu (X)</button>
      </div>
      <div id="out" class="box">Chưa có dữ liệu...</div>

      <script>
        async function sendResult(r){
          let res = await fetch('/update?r='+r);
          let data = await res.json();
          render(data);
        }
        function render(data){
          let html = '';
          html += '<b>=== BẢNG THỐNG KÊ ===</b><br/>';
          html += 'Tổng: '+data.stats.total+' | ';
          html += 'Tài='+data.stats.tai+' ('+data.stats.p_tai.toFixed(1)+'%) | ';
          html += 'Xỉu='+data.stats.xiu+' ('+data.stats.p_xiu.toFixed(1)+'%)<br/><br/>';

          html += '<b>Dự đoán kế:</b> '+data.predict[0]+' | <i>'+data.predict[1]+'</i><br/><br/>';

          html += '<b>Lịch sử gần nhất:</b> '+data.history.join(' ')+'<br/><br/>';

          html += '<b>Chuỗi gần đây:</b><br/>';
          data.streaks.forEach(s=>{
            html += s[0]+' ('+s[1]+')<br/>';
          });

          document.getElementById('out').innerHTML = html;
        }
      </script>
    </body>
    </html>
    """

@app.get("/update")
async def update(r: str):
    predictor.update(r)
    guess, reason = predictor.predict()
    return JSONResponse({
        "history": predictor.history[-50:],  # hiển thị 50 ván gần nhất
        "stats": predictor.stats(),
        "predict": [guess, reason],
        "streaks": predictor.streaks()
    })
