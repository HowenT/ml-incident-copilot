"""Diagnosis orchestration: windows -> onset -> checks -> hypotheses -> impact."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..config import utcnow
from ..models import Alert, Incident, LogEvent
from ..services.similarity import category_prior, find_similar, signature
from ..services.store import data_end, get_model, predictions, window
from ..stats import mean_shift_changepoint
from ..taxonomy import label
from .checks import ALL_CHECKS
from .context import Ctx, iso
from .hypotheses import PRIOR_WEIGHT, score_hypotheses
from .impact import assess_impact

ANALYSIS_HORIZON = timedelta(hours=72)
BASELINE_DAYS = 7
RULE_PRIORITY = ["missing_values", "serving_errors", "latency_slo", "decision_rate_shift", "prediction_drift",
                 "feature_drift", "performance_drop"]


def _onset_series(rule: str, alert: Alert):
    if rule == "missing_values":
        f = alert.features[0]
        return f"{f} missing rate", lambda d: d[f].isna().mean()
    if rule == "latency_slo":
        return "p95 latency", lambda d: d["latency_ms"].quantile(0.95)
    if rule == "serving_errors":
        return "fallback rate", lambda d: (d["status"] != "ok").mean()
    return "decision rate", lambda d: d["decision"].mean()


def refine_onset(preds: pd.DataFrame, alerts: list[Alert], end: datetime) -> tuple[datetime, str]:
    """Alerts fire late (rolling windows, persistence rules). Walk back to where the change began."""
    hourly = [a for a in alerts if a.rule != "performance_drop"] or alerts  # daily alerts start at midnight
    first = min(a.started_at for a in hourly)
    primary = sorted(hourly, key=lambda a: (RULE_PRIORITY.index(a.rule) if a.rule in RULE_PRIORITY else 99,
                                            a.started_at))[0]
    name, func = _onset_series(primary.rule, primary)
    lo = max(min(first, primary.started_at) - timedelta(hours=96), preds["ts"].min().to_pydatetime())
    hi = min(max(first, primary.started_at) + timedelta(hours=6), end)
    span = window(preds, lo, hi)
    hours = pd.date_range(pd.Timestamp(lo).floor("h"), pd.Timestamp(hi), freq="h", inclusive="left")
    y = span.groupby(span["ts"].dt.floor("h")).apply(func).reindex(hours).interpolate(limit_direction="both").to_numpy()
    k, gain = mean_shift_changepoint(y, min_size=4)
    if k is None or gain < 0.2:
        return first, f"first alert ({primary.rule})"
    m0, m1 = float(np.mean(y[:k])), float(np.mean(y[k:min(k + 24, len(y))]))
    thr = m0 + 0.25 * (m1 - m0)
    trailing = pd.Series(y).rolling(2, min_periods=1).mean().to_numpy()  # never looks ahead
    j = k
    while j - 1 >= 0 and (trailing[j - 1] - thr) * np.sign(m1 - m0) > 0:
        j -= 1
    onset = min(hours[j].to_pydatetime(), first)
    return onset, f"change-point on hourly {name} (explains {gain:.0%} of variance); first alert {iso(first)}"


def run_diagnosis(db: Session, incident: Incident) -> dict[str, Any]:
    t0 = time.perf_counter()
    model = get_model(db, incident.model_id)
    end = data_end(db)
    preds = predictions(model)
    alerts = list(incident.alerts)
    if alerts:
        onset, method = refine_onset(preds, alerts, end)
        window_end = end if any(a.status != "resolved" for a in alerts) else max(a.ended_at or end for a in alerts)
    else:
        onset, method = incident.created_at - timedelta(hours=12), "no linked alerts: 12h before incident creation"
        window_end = end
    data_start = preds["ts"].min().to_pydatetime()
    baseline_start = max(onset - timedelta(days=BASELINE_DAYS), data_start)
    analysis_end = min(window_end, onset + ANALYSIS_HORIZON)

    logs = [
        {"ts": r.ts, "model_id": r.model_id, "service": r.service, "level": r.level, "event_type": r.event_type,
         "message": r.message}
        for r in db.scalars(select(LogEvent).where(
            LogEvent.ts >= onset - timedelta(days=14), LogEvent.ts < analysis_end,
            or_(LogEvent.model_id == model["id"], LogEvent.model_id.is_(None))).order_by(LogEvent.ts))
    ]
    ctx = Ctx(
        model=model, preds=window(preds, baseline_start, analysis_end), base=window(preds, baseline_start, onset),
        cur=window(preds, onset, analysis_end), cur_full=window(preds, onset, window_end), onset=onset,
        window_end=analysis_end, baseline_start=baseline_start, data_end=end, logs=logs,
    )
    ctx.add_evidence("onset", f"Onset detected at {iso(onset)} UTC", method.capitalize() + ".", kind="context")
    for check in ALL_CHECKS:
        check(ctx)

    sig = signature(ctx.signals, ctx.model_type)
    similar = find_similar(db, incident, sig)
    prior = category_prior(similar)
    hypotheses = score_hypotheses(ctx, prior)
    for h in hypotheses:
        if h["prior"] > 0:
            refs = [s for s in similar if s["root_cause_category"] == h["category"] and s["similarity"] >= 0.5]
            h["supporting"].append({
                "signal": "historical_precedent", "weight": PRIOR_WEIGHT, "strength": h["prior"],
                "contribution": round(PRIOR_WEIGHT * h["prior"], 3),
                "evidence": [{"id": "H", "check": "knowledge", "kind": "finding", "chart": None,
                              "title": f"{len(refs)} similar past incident(s) confirmed as {h['label']}",
                              "detail": "; ".join(f"{s['code']} ({s['similarity']:.0%} similar): {s['title']}" for s in refs)}],
            })
    top = hypotheses[0]
    impact = assess_impact(ctx, top["category"])

    return {
        "incident_id": incident.id, "model_id": model["id"], "model_type": ctx.model_type,
        "ran_at": utcnow().isoformat(timespec="seconds"),
        "runtime_ms": round((time.perf_counter() - t0) * 1000),
        "onset": iso(onset), "onset_method": method,
        "window": {"start": iso(onset), "end": iso(analysis_end), "full_end": iso(window_end)},
        "baseline": {"start": iso(baseline_start), "end": iso(onset)},
        "rows": {"baseline": len(ctx.base), "incident": len(ctx.cur)},
        "checks": ctx.checks, "evidence": ctx.evidence, "signals": ctx.signals, "signature": sig,
        "hypotheses": hypotheses,
        "top": {"category": top["category"], "label": label(top["category"]), "confidence": top["confidence"],
                "summary": top["summary"]},
        "impact": impact, "similar": similar, "prior": prior, "facts": _jsonable(ctx.facts),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return iso(obj)
    return obj
