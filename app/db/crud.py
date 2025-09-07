from typing import Iterable, Optional
from sqlmodel import Session, select
from app.db.models import Round, Prediction
from datetime import datetime


def insert_round(session: Session, r: Round) -> Round:
r.compute()
session.add(r)
session.commit()
session.refresh(r)
return r


def latest_labels(session: Session, limit: int = 200) -> list[str]:
rows = session.exec(select(Round).order_by(Round.ts.desc()).limit(limit)).all()
return [('T' if row.label == 'TAI' else 'X') for row in reversed(rows)]


def last_label(session: Session) -> str | None:
row = session.exec(select(Round).order_by(Round.ts.desc()).limit(1)).first()
if not row:
return None
return 'T' if row.label == 'TAI' else 'X'


# Prediction helpers


def create_prediction(session: Session, md5: str, label_pred: str, p_tai: float, p_xiu: float) -> Prediction:
pred = Prediction(md5=md5 or '-', label_pred=label_pred, p_tai=p_tai, p_xiu=p_xiu)
session.add(pred)
session.commit()
session.refresh(pred)
return pred


def latest_unresolved_prediction(session: Session, md5: str) -> Optional[Prediction]:
return session.exec(
select(Prediction)
.where(Prediction.md5 == (md5 or '-'))
.where(Prediction.correct.is_(None))
.order_by(Prediction.ts.desc())
.limit(1)
).first()


def resolve_prediction(session: Session, pred: Prediction, round_obj: Round) -> Prediction:
pred.round_id = round_obj.id
pred.actual_label = round_obj.label
pred.correct = (pred.label_pred == round_obj.label)
pred.resolved_ts = datetime.utcnow()
session.add(pred)
session.commit()
session.refresh(pred)
return pred


def history(session: Session, limit: int = 50) -> list[Prediction]:
return session.exec(select(Prediction).order_by(Prediction.ts.desc()).limit(limit)).all()
