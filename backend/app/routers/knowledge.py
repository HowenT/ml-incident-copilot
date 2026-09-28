"""Knowledge base: resolved incidents, search, and how well the diagnosis has been doing."""
from __future__ import annotations

from statistics import median
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import serializers as ser
from ..db import get_db
from ..services.similarity import _cos, _tfidf, _tokens, incident_text, resolved_incidents
from ..services.workflow import now
from ..taxonomy import ROOT_CAUSES

router = APIRouter(prefix="/api", tags=["knowledge"])


@router.get("/taxonomy")
def taxonomy() -> dict[str, Any]:
    return {k: v["label"] for k, v in ROOT_CAUSES.items()}


@router.get("/knowledge")
def knowledge(q: str = "", category: str | None = None, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = resolved_incidents(db)
    if category:
        rows = [r for r in rows if r.resolution.root_cause_category == category]
    t = now(db)
    items = [ser.incident(r, t) for r in rows]
    if q.strip():
        vecs = _tfidf([_tokens(q)] + [_tokens(incident_text(r)) for r in rows])
        scores = [_cos(vecs[0], v) for v in vecs[1:]]
        ranked = sorted(zip(items, scores), key=lambda x: -x[1])
        return [{**i, "match": round(s, 3)} for i, s in ranked if s > 0]
    return sorted(items, key=lambda i: i["resolved_at"] or "", reverse=True)


@router.get("/knowledge/stats")
def stats(db: Session = Depends(get_db)) -> dict[str, Any]:
    rows = resolved_incidents(db)
    fb = [r.resolution.diagnosis_feedback for r in rows]
    judged = [f for f in fb if f in ("correct", "partially_correct", "incorrect")]
    by_cat: dict[str, dict[str, Any]] = {}
    for r in rows:
        c = r.resolution.root_cause_category
        d = by_cat.setdefault(c, {"category": c, "label": ROOT_CAUSES[c]["label"], "count": 0, "ttm": []})
        d["count"] += 1
        if r.resolution.time_to_mitigate_min:
            d["ttm"].append(r.resolution.time_to_mitigate_min)
    ttms = [r.resolution.time_to_mitigate_min for r in rows if r.resolution.time_to_mitigate_min]
    return {
        "resolved": len(rows),
        "top1_accuracy": sum(f == "correct" for f in judged) / len(judged) if judged else None,
        "partial": sum(f == "partially_correct" for f in judged),
        "incorrect": sum(f == "incorrect" for f in judged),
        "median_ttm_min": median(ttms) if ttms else None,
        "by_category": sorted(({**d, "median_ttm_min": median(d["ttm"]) if d["ttm"] else None, "ttm": None}
                               for d in by_cat.values()), key=lambda d: -d["count"]),
        "feedback_timeline": [{"code": r.code, "resolved_at": r.resolved_at.isoformat(timespec="minutes"),
                               "feedback": r.resolution.diagnosis_feedback}
                              for r in sorted(rows, key=lambda r: r.resolved_at)],
    }
