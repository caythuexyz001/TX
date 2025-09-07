async function fetchJSON(url, opts={}){
// attempt to reconstruct labels from recent history instead of counts
const hist = await fetchJSON('/history?limit=200');
const labels = hist.items.map(it => (it.actual_label ? (it.actual_label==='TAI'?'T':'X') : null)).filter(Boolean);
renderStreak(labels);
renderHeat(stats.transition);
}


async function refreshHistory(){
const h = await fetchJSON('/history?limit=50');
const s = await fetchJSON('/summary');
setText('summary', `Wins: ${s.wins} · Losses: ${s.losses} · Winrate: ${fmtPct(s.winrate)}`);
const tb = document.getElementById('hist-body');
tb.innerHTML = '';
h.items.forEach(it =>{
const tr = document.createElement('tr');
const res = it.correct===true ? 'WIN' : (it.correct===false ? 'LOSE' : '—');
tr.innerHTML = `<td>${new Date(it.ts).toLocaleString()}</td>
<td style="font-family:monospace">${it.md5}</td>
<td>${it.label_pred}</td>
<td>${it.p_tai.toFixed(3)}</td>
<td>${it.p_xiu.toFixed(3)}</td>
<td>${it.actual_label||''}</td>
<td>${res}</td>`;
tb.appendChild(tr);
})
}


async function doPredict(){
const md5 = document.getElementById('md5-input').value.trim();
if(!md5){ setText('predict-msg', 'Nhập MD5 trước.'); return; }
try{
const r = await fetchJSON(`/predict?md5=${encodeURIComponent(md5)}`);
setText('predict-msg', `→ Predict: ${r.label_suggest} | P(T)=${r.p_tai.toFixed(3)}, P(X)=${r.p_xiu.toFixed(3)} (ID ${r.prediction_id})`);
document.getElementById('md5-actual').value = md5; // tiện nhập actual cho cùng MD5
document.getElementById('md5-input').select();
refreshHistory();
}catch(e){ setText('predict-msg', 'Lỗi: '+e.message); }
}


async function doIngest(){
const md5 = document.getElementById('md5-actual').value.trim();
const d1 = parseInt(document.getElementById('d1').value, 10);
const d2 = parseInt(document.getElementById('d2').value, 10);
const d3 = parseInt(document.getElementById('d3').value, 10);
try{
const body = {d1, d2, d3};
if(md5) body.md5 = md5;
const r = await fetchJSON('/ingest', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
if(r.resolved_prediction){
const ok = r.resolved_prediction.correct ? 'WIN' : 'LOSE';
setText('ingest-msg', `Lưu vòng #${r.stored_round.id} (${r.stored_round.label}) · gắn với predict ${r.resolved_prediction.id} → ${ok}`);
}else{
setText('ingest-msg', `Lưu vòng #${r.stored_round.id} (${r.stored_round.label})`);
}
refreshStats();
refreshHistory();
}catch(e){ setText('ingest-msg', 'Lỗi: '+e.message); }
}


document.getElementById('btn-predict').addEventListener('click', doPredict);


document.getElementById('btn-ingest').addEventListener('click', doIngest);


document.getElementById('btn-refresh').addEventListener('click', ()=>{refreshStats(); refreshHistory();});


refreshStats();
refreshHistory();
setInterval(()=>{refreshStats(); refreshHistory();}, 5000);
