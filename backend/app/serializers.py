"""ORM -> JSON-friendly dicts."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import Action, Alert, Incident, IncidentEvent, Resolution, Summary
from .taxonomy import label


def _iso(d: datetime | None) -> str | None:
    return d.isoformat(timespec="minutes") if d else None


def alert(a: Alert) -> dict[str, Any]:
    return {"id": a.id, "model_id": a.model_id, "rule": a.rule, "title": a.title, "metric": a.metric,
            "features": a.features, "severity": a.severity, "status": a.status, "started_at": _iso(a.started_at),
            "ended_at": _iso(a.ended_at), "peak_value": a.peak_value, "baseline_value": a.baseline_value,
            "threshold": a.threshold, "incident_id": a.incident_id}


def action(a: Action) -> dict[str, Any]:
    return {"id": a.id, "priority": a.priority, "kind": a.kind, "title": a.title, "detail": a.detail, "owner": a.owner,
            "rationale": a.rationale, "customer_text": a.customer_text, "ours": bool(a.ours), "status": a.status,
            "updated_at": _iso(a.updated_at)}


def event(e: IncidentEvent) -> dict[str, Any]:
    return {"id": e.id, "ts": _iso(e.ts), "actor": e.actor, "kind": e.kind, "message": e.message, "data": e.data}


def summary(s: Summary) -> dict[str, Any]:
    return {"id": s.id, "audience": s.audience, "content": s.content, "generator": s.generator,
            "created_at": _iso(s.created_at), "sent_at": _iso(s.sent_at)}


def resolution(r: Resolution | None) -> dict[str, Any] | None:
    if r is None:
        return None
    return {"root_cause_category": r.root_cause_category, "root_cause_label": label(r.root_cause_category),
            "root_cause_detail": r.root_cause_detail, "fix_applied": r.fix_applied, "prevention": r.prevention,
            "diagnosis_feedback": r.diagnosis_feedback, "time_to_mitigate_min": r.time_to_mitigate_min,
            "tags": r.tags, "resolved_by": r.resolved_by, "created_at": _iso(r.created_at)}


def incident(i: Incident, now: datetime | None = None) -> dict[str, Any]:
    end = i.resolved_at or now
    return {
        "id": i.id, "code": i.code, "title": i.title, "model_id": i.model_id, "severity": i.severity,
        "status": i.status, "customer": i.customer, "customer_impact": i.customer_impact,
        "business_context": i.business_context, "affected_segments": i.affected_segments,
        "reported_by": i.reported_by, "customer_contact": i.customer_contact, "created_at": _iso(i.created_at),
        "updated_at": _iso(i.updated_at), "mitigated_at": _iso(i.mitigated_at), "resolved_at": _iso(i.resolved_at),
        "historical": bool(i.historical), "diagnosed_category": i.diagnosed_category,
        "diagnosed_label": label(i.diagnosed_category) if i.diagnosed_category else None,
        "diagnosed_confidence": i.diagnosed_confidence,
        "age_hours": round((end - i.created_at).total_seconds() / 3600, 1) if end else None,
        "alert_count": len(i.alerts),
        "actions_open": sum(1 for a in i.actions if a.status in ("todo", "in_progress")),
        "actions_total": len(i.actions),
        "resolution": resolution(i.resolution),
    }
