from fastapi import APIRouter, Depends, HTTPException, Header
from sqlmodel import Session
from app.db.base import get_session
from app.api.schemas import IngestIn, PredictOut, StatsOut
from app.services import ingest_round, get_stats, get_patterns, predict_by_markov
from app.config import settings
from app.core.validation import is_valid_md5, is_valid_dice

router = APIRouter()


def _auth(api_key_header: str | None = Header(default=None, alias="X-API-Key")):
    if settings.api_key and api_key_header != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.post('/ingest')
async def ingest(data: IngestIn, session: Session = Depends(get_session), ok=Depends(_auth)):
    for d in (data.d1, data.d2, data.d3):
        if not is_valid_dice(d):
            raise HTTPException(400, detail="dice must be 1..6")
    if data.md5 and not is_valid_md5(data.md5):
        raise HTTPException(400, detail="md5 must be 32 hex")
    r = ingest_round(session, data.d1, data.d2, data.d3, data.md5)
    return {
        'stored_round': {
            'id': r.id, 'ts': r.ts, 'd1': r.d1, 'd2': r.d2, 'd3': r.d3,
            'total': r.total, 'label': r.label, 'is_triple': r.is_triple
        },
        'updated_stats': get_stats(session)
    }

@router.get('/stats', response_model=StatsOut)
async def stats(session: Session = Depends(get_session)):
    return get_stats(session)

@router.get('/patterns')
async def patterns(min_k: int = 3, session: Session = Depends(get_session)):
    return get_patterns(session, min_run=min_k)

@router.get('/predict', response_model=PredictOut)
async def predict(md5: str | None = None, session: Session = Depends(get_session)):
    if md5 and not is_valid_md5(md5):
        raise HTTPException(400, detail="md5 must be 32 hex")
    return predict_by_markov(session, md5=md5)