"""Similar-incident retrieval over the knowledge base (resolved incidents).

similarity = 0.65 · cosine(diagnosis signature) + 0.35 · cosine(TF-IDF of incident text)

The signature captures *how* an incident looked to the diagnosis engine (which layers were
abnormal); the text captures what people said about it. Together they surface past fixes
that are worth reusing, and give the diagnosis a historical prior per root-cause category.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Incident
from ..taxonomy import label

SIGNATURE_KEYS = [
    "missing_spike", "missing_segment_concentration", "missing_drives_decision", "data_source_change_near_onset",
    "data_errors", "feature_drift", "drift_segment_concentration", "business_event_near_onset", "prediction_shift",
    "decision_shift_without_score_shift", "performance_drop", "performance_segment_concentration", "latency_spike",
    "serving_errors", "version_correlated", "deploy_near_onset", "config_change_near_onset", "infra_errors",
    "no_change_events", "no_feature_drift",
]
STOP = set("the a an and or of to in on for with from by at is was were be been are this that these those it its as "
           "not no our their we they after before since into than more less per via up".split())


def signature(signals: dict[str, float], model_type: str) -> dict[str, float]:
    sig = {k: round(float(signals.get(k, 0.0)), 3) for k in SIGNATURE_KEYS if signals.get(k, 0.0) > 0}
    sig[f"model_{model_type}"] = 1.0
    return sig


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9_]+", text.lower()) if t not in STOP and len(t) > 2]


def incident_text(inc: Incident) -> str:
    parts = [inc.title, inc.customer_impact, inc.business_context, inc.affected_segments]
    if inc.resolution:
        parts += [inc.resolution.root_cause_detail, inc.resolution.fix_applied, " ".join(inc.resolution.tags or [])]
    return " ".join(p for p in parts if p)


def _tfidf(docs: list[list[str]]) -> list[dict[str, float]]:
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    out = []
    for d in docs:
        tf = Counter(d)
        out.append({t: (c / len(d)) * math.log((1 + n) / (1 + df[t])) for t, c in tf.items()} if d else {})
    return out


def resolved_incidents(db: Session, exclude_id: int | None = None) -> list[Incident]:
    rows = db.scalars(select(Incident).where(Incident.status == "resolved")).all()
    return [r for r in rows if r.resolution is not None and r.id != exclude_id]


def find_similar(db: Session, inc: Incident, sig: dict[str, float] | None, k: int = 3) -> list[dict]:
    pool = resolved_incidents(db, exclude_id=inc.id)
    if not pool:
        return []
    vecs = _tfidf([_tokens(incident_text(inc))] + [_tokens(incident_text(p)) for p in pool])
    results = []
    for p, v in zip(pool, vecs[1:]):
        text_sim = _cos(vecs[0], v)
        sig_sim = _cos(sig, p.signature or {}) if sig else 0.0
        sim = 0.65 * sig_sim + 0.35 * text_sim if sig else text_sim
        r = p.resolution
        results.append({
            "incident_id": p.id, "code": p.code, "title": p.title, "model_id": p.model_id,
            "similarity": round(sim, 3), "signature_similarity": round(sig_sim, 3), "text_similarity": round(text_sim, 3),
            "root_cause_category": r.root_cause_category, "root_cause_label": label(r.root_cause_category),
            "root_cause_detail": r.root_cause_detail, "fix_applied": r.fix_applied, "prevention": r.prevention,
            "diagnosis_feedback": r.diagnosis_feedback, "time_to_mitigate_min": r.time_to_mitigate_min,
            "created_at": p.created_at.isoformat(),
        })
    results.sort(key=lambda x: -x["similarity"])
    return results[:k]


def category_prior(similar: list[dict], min_sim: float = 0.5) -> dict[str, float]:
    """Similarity-weighted vote over the *confirmed* root causes of close past incidents."""
    votes: dict[str, float] = {}
    for s in similar:
        if s["similarity"] >= min_sim:
            votes[s["root_cause_category"]] = votes.get(s["root_cause_category"], 0.0) + s["similarity"]
    total = sum(votes.values())
    return {k: v / total for k, v in votes.items()} if total else {}
