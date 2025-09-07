from typing import Optional
def ingest_round(session: Session, d1: int, d2: int, d3: int, md5: Optional[str] = None):
r = Round(md5=md5, d1=d1, d2=d2, d3=d3)
r.compute()
out = insert_round(session, r)
# update markov buffer
global _cached_built
_mkv.push('T' if out.label == 'TAI' else 'X')
_cached_built = True


# try resolve last unresolved prediction for this md5
resolved = None
if md5:
pred = latest_unresolved_prediction(session, md5)
if pred:
resolved = resolve_prediction(session, pred, out)
return out, resolved




def get_stats(session: Session):
_ensure_markov(session)
last = last_label(session)
stats = _mkv.stats(last)
return {
'transition': stats.transition,
'counts': stats.counts,
'last_label': stats.last_label,
'p_value_row': stats.p_value_row,
'entropy': stats.entropy,
}




def get_patterns(session: Session, min_run: int = 3):
labels = latest_labels(session, limit=settings.window)
return {
'runs': runs(labels, k=min_run),
'alternations': alternations(labels, L=4),
'blocks': blocks(labels),
}




def predict_by_markov(session: Session, md5: str | None = None):
_ensure_markov(session)
last = last_label(session)
pT, pX = _mkv.probs(last)
label = 'TAI' if pT >= pX else 'XIU'
pred = create_prediction(session, md5 or '-', label, pT, pX)
return {'prediction_id': pred.id, 'label_suggest': label, 'p_tai': pT, 'p_xiu': pX, 'method': 'markov'}




def get_history(session: Session, limit: int = 50):
rows = history(session, limit=limit)
def to_dict(p):
return {
'id': p.id,
'md5': p.md5,
'label_pred': p.label_pred,
'p_tai': p.p_tai,
'p_xiu': p.p_xiu,
'actual_label': p.actual_label,
'correct': p.correct,
'ts': p.ts.isoformat(),
'resolved_ts': p.resolved_ts.isoformat() if p.resolved_ts else None,
}
return [to_dict(x) for x in rows]




def get_summary(session: Session):
rows = history(session, limit=100000)
wins = sum(1 for r in rows if r.correct is True)
losses = sum(1 for r in rows if r.correct is False)
total = wins + losses
winrate = (wins / total) if total else 0.0
return {'wins': wins, 'losses': losses, 'total': total, 'winrate': winrate}
