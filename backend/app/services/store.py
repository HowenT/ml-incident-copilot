"""Read helpers shared by routers and the diagnosis engine."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import utcnow
from ..db import engine
from ..models import Meta, MLModel

_pred_cache: dict[str, pd.DataFrame] = {}


def clear_cache() -> None:
    _pred_cache.clear()


def model_dict(m: MLModel) -> dict[str, Any]:
    return {c.name: getattr(m, c.name) for c in MLModel.__table__.columns}


def get_model(db: Session, model_id: str) -> dict[str, Any]:
    m = db.get(MLModel, model_id)
    if m is None:
        raise KeyError(model_id)
    return model_dict(m)


def predictions(model: dict[str, Any]) -> pd.DataFrame:
    """Full prediction log for a model (cached; ~100k rows loads in well under a second)."""
    key = model["id"]
    if key not in _pred_cache:
        df = pd.read_sql_table(model["prediction_table"], engine)
        for col in ("ts", "label_available_at"):
            df[col] = pd.to_datetime(df[col])
        _pred_cache[key] = df
    return _pred_cache[key]


def window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df[(df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))]


def meta(db: Session, key: str, default: str | None = None) -> str | None:
    row = db.get(Meta, key)
    return row.value if row else default


def data_end(db: Session) -> datetime:
    value = meta(db, "data_end")
    return datetime.fromisoformat(value) if value else utcnow()


def all_models(db: Session) -> list[dict[str, Any]]:
    return [model_dict(m) for m in db.scalars(select(MLModel).order_by(MLModel.id))]
