from typing import Optional
from sqlmodel import Session
from app.db.models import Round, Prediction
from app.db.crud import latest_labels, last_label, insert_round
from app.analytics.markov import MarkovEngine
from app.analytics.patterns import runs, alternations, blocks
from app.config import settings

# Singleton-ish in-memory engines (rebuilt on demand from DB)
_mkv = MarkovEngine(window=settings.window, alpha=1.0)
_cached_built = False


def _ensure_markov(session: Session):
    global _cached_built
    if not _cached_built:
        labels = latest_labels(session, limit=settings.window)
        _mkv.build_from(labels)
        _cached_built = True


def ingest_round(session: Session, d1: int, d2: int, d3: int, md5: Optional[str] = None) -> Round:
    r = Round(md5=md5, d1=d1, d2=d2, d3=d3)
    r.compute()
    out = insert_round(session, r)
    # update markov buffer
    global _cached_built
    _mkv.push('T' if out.label == 'TAI' else 'X')
    _cached_built = True
    return out


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
    pred = Prediction(md5=md5 or '-', label_pred=label, p_tai=pT, p_xiu=pX)
    session.add(pred); session.commit()
    return {'label_suggest': label, 'p_tai': pT, 'p_xiu': pX, 'method': 'markov'}