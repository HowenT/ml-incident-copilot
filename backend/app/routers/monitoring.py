"""Monitoring read APIs: fleet overview, model metrics, feature health, logs, alerts."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .. import serializers as ser
from ..db import engine, get_db
from ..models import Alert, Incident, LogEvent
from ..services.knowledge_seed import SAMPLE_REPORTS, sample_report_key
from ..services.store import all_models, data_end, get_model, meta
from ..services.workflow import now

router = APIRouter(prefix="/api", tags=["monitoring"])

SEVERITY_RANK = {"critical": 3, "high": 2, "medium": 1, "low": 0}


def _metrics_df(model_id: str, since=None) -> pd.DataFrame:
    q = "SELECT * FROM metrics_hourly WHERE model_id = :m ORDER BY ts"
    df = pd.read_sql_query(q, engine, params={"m": model_id})
    df["ts"] = pd.to_datetime(df["ts"])
    if since is not None:
        df = df[df["ts"] >= pd.Timestamp(since)]
    return df


def _json_safe(df: pd.DataFrame) -> pd.DataFrame:
    """NaN -> None so the frame serialises to valid JSON."""
    return df.astype(object).where(pd.notna(df), None)


def _nan(x: Any) -> Any:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def _model_kpis(db: Session, model: dict[str, Any]) -> dict[str, Any]:
    end = data_end(db)
    m = _metrics_df(model["id"], end - timedelta(days=8))
    last24 = m[m["ts"] >= pd.Timestamp(end - timedelta(hours=24))]
    prior = m[(m["ts"] < pd.Timestamp(end - timedelta(hours=24)))]
    scale = 1.0 / (model.get("log_sample_rate") or 1.0)
    vol24 = float(last24["volume"].sum())
    dec24 = float((last24["decision_rate"] * last24["volume"]).sum() / max(vol24, 1))
    dec_prior = float((prior["decision_rate"] * prior["volume"]).sum() / max(prior["volume"].sum(), 1))
    open_alerts = db.scalars(select(Alert).where(Alert.model_id == model["id"], Alert.status != "resolved")).all()
    worst = max((SEVERITY_RANK[a.severity] for a in open_alerts), default=-1)
    return {
        "id": model["id"], "display_name": model["display_name"], "model_type": model["model_type"],
        "version": model["current_version"], "client": model["client"], "use_case": model["use_case"],
        "owner": model["owner"], "decision_name": model["decision_name"], "threshold": model["threshold"],
        "latency_slo_ms": model["latency_slo_ms"], "label_lag_hours": model["label_lag_hours"],
        "log_sample_rate": model["log_sample_rate"],
        "features": model["features"], "segments": model["segments"],
        "kpis": {
            "volume_24h": round(vol24 * scale), "decision_rate_24h": dec24, "decision_rate_7d": dec_prior,
            "latency_p95_24h": _nan(float(last24["latency_p95"].median())) if len(last24) else None,
            "latency_p95_last": _nan(float(m["latency_p95"].iloc[-1])) if len(m) else None,
            "timeout_rate_24h": float((last24["timeout_rate"] * last24["volume"]).sum() / max(vol24, 1)),
            "max_missing_24h": float(last24["max_missing_rate"].max()) if len(last24) else 0.0,
            "max_psi_now": float(m["max_psi_24h"].iloc[-1]) if len(m) else 0.0,
            "open_alerts": len(open_alerts),
        },
        "health": "critical" if worst >= 3 else "degraded" if worst >= 1 else "healthy",
    }


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    from ..config import ANTHROPIC_MODEL, llm_available

    return {"status": "ok", "data_start": meta(db, "data_start"), "data_end": meta(db, "data_end"),
            "now": now(db).isoformat(timespec="minutes"), "client": meta(db, "client"),
            "llm_enabled": llm_available(), "llm_model": ANTHROPIC_MODEL}


@router.get("/overview")
def overview(db: Session = Depends(get_db)) -> dict[str, Any]:
    models = [_model_kpis(db, m) for m in all_models(db)]
    live = db.scalars(select(Incident).where(Incident.historical == 0)).all()
    open_inc = [i for i in live if i.status != "resolved"]
    return {
        "models": models,
        "open_alerts": sum(m["kpis"]["open_alerts"] for m in models),
        "open_incidents": len(open_inc),
        "incidents_by_status": {s: sum(1 for i in live if i.status == s) for s in
                                ("open", "investigating", "mitigated", "resolved")},
        "now": now(db).isoformat(timespec="minutes"),
    }


@router.get("/models")
def list_models(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [_model_kpis(db, m) for m in all_models(db)]


@router.get("/models/{model_id}")
def model_detail(model_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return _model_kpis(db, get_model(db, model_id))
    except KeyError:
        raise HTTPException(404, "Unknown model")


@router.get("/models/{model_id}/metrics")
def model_metrics(model_id: str, days: int = Query(30, ge=1, le=60), db: Session = Depends(get_db)) -> dict[str, Any]:
    end = data_end(db)
    df = _metrics_df(model_id, end - timedelta(days=days))
    df = df.drop(columns=["id", "model_id"])
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M")
    df = _json_safe(df)
    return {"columns": df.columns.tolist(), "rows": df.to_dict(orient="list")}


@router.get("/models/{model_id}/features")
def feature_stats(model_id: str, days: int = Query(30, ge=1, le=60), db: Session = Depends(get_db)) -> dict[str, Any]:
    end = data_end(db)
    df = pd.read_sql_query("SELECT ts, feature, missing_rate, psi_24h FROM feature_stats_hourly WHERE model_id = :m",
                           engine, params={"m": model_id})
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[df["ts"] >= pd.Timestamp(end - timedelta(days=days))]
    missing = df.pivot(index="ts", columns="feature", values="missing_rate")
    psi = df.pivot(index="ts", columns="feature", values="psi_24h")
    daily_psi = psi.resample("D").last()
    return {
        "ts": [t.strftime("%Y-%m-%dT%H:%M") for t in missing.index],
        "missing": {c: missing[c].round(4).tolist() for c in missing.columns},
        "psi": {c: psi[c].round(4).tolist() for c in psi.columns},
        "daily_psi": {"days": [t.strftime("%Y-%m-%d") for t in daily_psi.index],
                      "features": daily_psi.columns.tolist(),
                      "values": daily_psi.round(3).fillna(0).to_numpy().T.tolist()},
    }


@router.get("/models/{model_id}/performance")
def performance(model_id: str) -> list[dict[str, Any]]:
    df = pd.read_sql_query("SELECT day, labeled, positives, auc, precision, recall FROM performance_daily "
                           "WHERE model_id = :m ORDER BY day", engine, params={"m": model_id})
    df["day"] = pd.to_datetime(df["day"]).dt.strftime("%Y-%m-%d")
    return _json_safe(df).to_dict(orient="records")


@router.get("/logs")
def logs(model_id: str | None = None, level: str | None = None, event_type: str | None = None,
         limit: int = Query(200, le=2000), db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    q = select(LogEvent).order_by(LogEvent.ts.desc()).limit(limit)
    if model_id:
        q = q.where(or_(LogEvent.model_id == model_id, LogEvent.model_id.is_(None)))
    if level:
        q = q.where(LogEvent.level.in_(level.split(",")))
    if event_type:
        q = q.where(LogEvent.event_type.in_(event_type.split(",")))
    return [{"ts": r.ts.isoformat(timespec="minutes"), "model_id": r.model_id, "service": r.service, "level": r.level,
             "event_type": r.event_type, "message": r.message} for r in db.scalars(q)]


@router.get("/alerts")
def alerts(status: str | None = None, model_id: str | None = None, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    q = select(Alert)
    if status:
        q = q.where(Alert.status.in_(status.split(",")))
    if model_id:
        q = q.where(Alert.model_id == model_id)
    rows = db.scalars(q).all()
    rows = sorted(rows, key=lambda a: (a.status == "resolved", -SEVERITY_RANK[a.severity], -a.started_at.timestamp()))
    codes = {i.id: i.code for i in db.scalars(select(Incident))}
    return [{**ser.alert(a), "incident_code": codes.get(a.incident_id)} for a in rows]


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    a = db.get(Alert, alert_id)
    if a is None:
        raise HTTPException(404, "Unknown alert")
    if a.status == "open":
        a.status = "acknowledged"
        db.commit()
    return ser.alert(a)


@router.get("/alerts/sample-report")
def sample_report(alert_ids: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Pre-filled customer report for the demo (the 'Use sample customer report' button)."""
    ids = [int(x) for x in alert_ids.split(",") if x]
    rows = db.scalars(select(Alert).where(Alert.id.in_(ids))).all()
    if not rows:
        raise HTTPException(404, "No alerts")
    key = sample_report_key(rows[0].model_id, {a.rule for a in rows})
    if key is None:
        raise HTTPException(404, "No sample report for this alert")
    return SAMPLE_REPORTS[key]
