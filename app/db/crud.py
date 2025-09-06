from typing import Iterable
from sqlmodel import Session, select
from app.db.models import Round

def insert_round(session: Session, r: Round) -> Round:
    r.compute()
    session.add(r)
    session.commit()
    session.refresh(r)
    return r

def latest_labels(session: Session, limit: int = 200) -> list[str]:
    rows = session.exec(select(Round).order_by(Round.ts.desc()).limit(limit)).all()
    return [ ('T' if row.label == 'TAI' else 'X') for row in reversed(rows) ]

def last_label(session: Session) -> str | None:
    row = session.exec(select(Round).order_by(Round.ts.desc()).limit(1)).first()
    if not row:
        return None
    return 'T' if row.label == 'TAI' else 'X'