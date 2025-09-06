async function fetchJSON(url){
  const r = await fetch(url);
  return await r.json();
}

let streakChart, heatChart;

function renderStreak(labels){
  const ctx = document.getElementById('streak');
  const data = labels.map((v,i)=>({x:i+1, y: v==='T'?1:0}));
  const cfg = {type:'line', data:{datasets:[{label:'T=1, X=0', data}]}, options:{scales:{y:{min:-0.1, max:1.1}}}};
  if(streakChart){ streakChart.destroy(); }
  streakChart = new Chart(ctx, cfg);
}

function renderHeat(map){
  const ctx = document.getElementById('heat');
  const data = {
    labels:['T→T','T→X','X→T','X→X'],
    datasets:[{label:'P', data:[map[0][0], map[0][1], map[1][0], map[1][1]]}]
  };
  const cfg = {type:'bar', data};
  if(heatChart){ heatChart.destroy(); }
  heatChart = new Chart(ctx, cfg);
}

function renderPatterns(p){
  const tb = document.getElementById('pattern-body');
  tb.innerHTML = '';
  const rows = [];
  p.runs.forEach(([s,e,ch,len])=>rows.push(['RUN', `${ch} ${len}x (idx ${s}-${e})`, len]));
  p.alternations.forEach(([s,e])=>rows.push(['ALT', `alternation (idx ${s}-${e})`, e-s+1]));
  p.blocks.forEach(([s,e,blk,count])=>rows.push(['BLOCK', `${blk} × ${count} (idx ${s}-${e})`, count*4]));
  rows.slice(0,50).forEach(r=>{
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td>`;
    tb.appendChild(tr);
  });
}

async function refresh(){
  const stats = await fetchJSON('/stats');
  const pats = await fetchJSON('/patterns');
  // Build streak from counts approximation (for a quick view we re-query history via patterns labels length) – in MVP we skip full history endpoint
  const labels = [];
  const total = stats.counts[0][0]+stats.counts[0][1]+stats.counts[1][0]+stats.counts[1][1];
  // Dummy visual: last_label repeated total times (simple placeholder since API hasn't history endpoint yet)
  for(let i=0;i<Math.max(total, 10);i++){ labels.push(stats.last_label||'T'); }
  renderStreak(labels);
  renderHeat(stats.transition);
  renderPatterns(pats);
}

refresh();
setInterval(refresh, 4000);