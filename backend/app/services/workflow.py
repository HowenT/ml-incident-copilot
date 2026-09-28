"""Incident lifecycle: create -> diagnose -> act -> communicate -> resolve (+ learn).

Every state change writes an IncidentEvent, so the timeline is the audit trail.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import utcnow
from ..diagnosis.engine import run_diagnosis
from ..models import Action, Alert, Diagnosis, Incident, IncidentEvent, Resolution, Summary
from ..taxonomy import ROOT_CAUSES, STATUSES, label
from . import llm
from .actions import generate_actions
from .similarity import signature
from .store import data_end, get_model, meta
from .summaries import build_summaries

COPILOT = "Incident Copilot"


class WorkflowError(ValueError):
    pass


def now(db: Session) -> datetime:
    """Simulated clock: the data ends at seed time, so 'now' advances from there."""
    seeded = meta(db, "seeded_at")
    end = data_end(db)
    if not seeded:
        return utcnow()
    return end + (utcnow() - datetime.fromisoformat(seeded))


def log_event(db: Session, inc: Incident, actor: str, kind: str, message: str, data: dict | None = None,
              ts: datetime | None = None) -> None:
    db.add(IncidentEvent(incident_id=inc.id, ts=ts or now(db), actor=actor, kind=kind, message=message, data=data))
    inc.updated_at = ts or now(db)


def next_code(db: Session, year: int) -> str:
    seqs = [int(m.group(1)) for c in db.scalars(select(Incident.code))
            if (m := re.match(rf"INC-{year}-(\d+)", c))]
    return f"INC-{year}-{max(max(seqs, default=0), 230) + 1:04d}"  # other accounts use 0216-0230


# --------------------------------------------------------------------------------------
# Create
# --------------------------------------------------------------------------------------


def create_incident(db: Session, data: dict[str, Any], actor: str | None = None,
                    created_at: datetime | None = None) -> Incident:
    alert_ids = data.get("alert_ids") or []
    alerts = db.scalars(select(Alert).where(Alert.id.in_(alert_ids))).all() if alert_ids else []
    if len(alerts) != len(set(alert_ids)):
        raise WorkflowError("Unknown alert id(s).")
    model_id = data.get("model_id") or (alerts[0].model_id if alerts else None)
    if not model_id:
        raise WorkflowError("Pick at least one alert or a model.")
    if any(a.model_id != model_id for a in alerts):
        raise WorkflowError("All linked alerts must belong to the same model.")
    taken = [a for a in alerts if a.incident_id is not None]
    if taken:
        raise WorkflowError(f"Alert(s) {', '.join(str(a.id) for a in taken)} already linked to an incident.")
    model = get_model(db, model_id)
    ts = created_at or now(db)
    inc = Incident(
        code=next_code(db, ts.year), title=data["title"].strip(), model_id=model_id,
        severity=data.get("severity", "SEV2"), status="open", customer=model["client"],
        customer_impact=data.get("customer_impact", ""), business_context=data.get("business_context", ""),
        affected_segments=data.get("affected_segments", ""), reported_by=data.get("reported_by", ""),
        customer_contact=data.get("customer_contact", ""), created_at=ts, updated_at=ts,
    )
    db.add(inc)
    db.flush()
    who = actor or inc.reported_by or "FDE"
    log_event(db, inc, who, "created", f"Incident opened ({inc.severity}) for {model_id}.", ts=ts)
    for a in alerts:
        a.incident_id = inc.id
        if a.status == "open":
            a.status = "acknowledged"
    if alerts:
        log_event(db, inc, who, "alerts_linked", f"Linked {len(alerts)} alert(s): " + "; ".join(a.title for a in alerts),
                  {"alert_ids": [a.id for a in alerts]}, ts=ts)
    db.commit()
    db.refresh(inc)
    return inc


# --------------------------------------------------------------------------------------
# Diagnose
# --------------------------------------------------------------------------------------


def latest_diagnosis(db: Session, inc: Incident) -> dict[str, Any] | None:
    row = db.scalars(select(Diagnosis).where(Diagnosis.incident_id == inc.id).order_by(Diagnosis.id.desc())).first()
    return row.result if row else None


def diagnose(db: Session, inc: Incident, actor: str = COPILOT, ts: datetime | None = None) -> dict[str, Any]:
    result = run_diagnosis(db, inc)
    ts = ts or now(db)
    db.add(Diagnosis(incident_id=inc.id, created_at=ts, result=result))
    top = result["hypotheses"][0]
    inc.diagnosed_category = top["category"]
    inc.diagnosed_confidence = top["confidence"]
    inc.signature = result["signature"]
    log_event(db, inc, actor, "diagnosis",
              f"Diagnosis: {top['label']} ({top['confidence']:.0%}). {result['impact']['headline']}",
              {"top": top["category"], "confidence": top["confidence"], "runtime_ms": result["runtime_ms"]}, ts=ts)
    if inc.status == "open":
        inc.status = "investigating"
        log_event(db, inc, actor, "status", "Status → investigating", ts=ts)

    model = get_model(db, inc.model_id)
    keep = {a.title for a in inc.actions if a.status != "todo"}
    for a in list(inc.actions):
        if a.status == "todo":
            db.delete(a)
    db.flush()
    for i, a in enumerate(generate_actions(result, model, inc.customer)):
        if a["title"] in keep:
            continue
        db.add(Action(incident_id=inc.id, sort_order=i, updated_at=ts, **a))
    db.flush()
    db.refresh(inc)
    _store_summaries(db, inc, model, result, ts)
    db.commit()
    db.refresh(inc)
    return result


def _store_summaries(db: Session, inc: Incident, model: dict, diag: dict, ts: datetime) -> None:
    for audience, text in build_summaries(inc, model, diag, list(inc.actions), ts).items():
        db.add(Summary(incident_id=inc.id, audience=audience, content=text, generator="template", created_at=ts))


def latest_summaries(db: Session, inc: Incident) -> dict[str, Summary]:
    out: dict[str, Summary] = {}
    for s in db.scalars(select(Summary).where(Summary.incident_id == inc.id).order_by(Summary.id)):
        out[s.audience] = s
    return out


def regenerate_summaries(db: Session, inc: Incident, use_llm: bool, actor: str) -> dict[str, Summary]:
    diag = latest_diagnosis(db, inc)
    if diag is None:
        raise WorkflowError("Run the diagnosis first.")
    model = get_model(db, inc.model_id)
    ts = now(db)
    drafts = build_summaries(inc, model, diag, list(inc.actions), ts)
    generator = "template"
    if use_llm:
        facts = {"incident": {"code": inc.code, "title": inc.title, "severity": inc.severity, "status": inc.status,
                              "customer_impact": inc.customer_impact, "business_context": inc.business_context},
                 "top_hypothesis": diag["top"], "impact": diag["impact"], "facts": diag["facts"]}
        polished = {}
        for audience, draft in drafts.items():
            polished[audience], generator = llm.polish(audience, draft, facts)
        drafts = polished
    for audience, text in drafts.items():
        db.add(Summary(incident_id=inc.id, audience=audience, content=text, generator=generator, created_at=ts))
    log_event(db, inc, actor, "summary", f"Summaries regenerated ({generator}).")
    db.commit()
    return latest_summaries(db, inc)


def mark_summary_sent(db: Session, inc: Incident, audience: str, actor: str) -> Summary:
    s = latest_summaries(db, inc).get(audience)
    if s is None:
        raise WorkflowError("No summary yet.")
    s.sent_at = now(db)
    to = (inc.customer_contact or inc.customer) if audience == "customer" else "#ml-oncall"
    log_event(db, inc, actor, "communication", f"{audience.capitalize()} update sent to {to}.")
    comm = next((a for a in inc.actions if a.kind == "communicate" and a.status != "done"), None)
    if comm and audience == "customer":
        comm.status = "done"
        comm.updated_at = now(db)
    db.commit()
    return s


# --------------------------------------------------------------------------------------
# Act
# --------------------------------------------------------------------------------------


def update_action(db: Session, inc: Incident, action_id: int, status: str, actor: str) -> Action:
    action = db.get(Action, action_id)
    if action is None or action.incident_id != inc.id:
        raise WorkflowError("Action not found for this incident.")
    if status not in {"todo", "in_progress", "done", "skipped"}:
        raise WorkflowError("Invalid action status.")
    action.status = status
    action.updated_at = now(db)
    log_event(db, inc, actor, "action", f"[{action.priority}] {action.title} → {status.replace('_', ' ')}")
    mitigations = [a for a in inc.actions if a.priority == "P0" and a.kind == "mitigate"]
    if inc.status == "investigating" and mitigations and all(a.status in ("done", "skipped") for a in mitigations):
        inc.status = "mitigated"
        inc.mitigated_at = now(db)
        log_event(db, inc, COPILOT, "status", "All P0 mitigations complete → status mitigated.")
    db.commit()
    return action


def set_status(db: Session, inc: Incident, status: str, actor: str) -> Incident:
    if status not in STATUSES:
        raise WorkflowError("Invalid status.")
    if status == "resolved":
        raise WorkflowError("Resolve via the resolution form so the root cause is recorded.")
    inc.status = status
    if status == "mitigated" and not inc.mitigated_at:
        inc.mitigated_at = now(db)
    log_event(db, inc, actor, "status", f"Status → {status}")
    db.commit()
    return inc


def set_severity(db: Session, inc: Incident, severity: str, actor: str) -> Incident:
    old = inc.severity
    inc.severity = severity
    log_event(db, inc, actor, "severity", f"Severity {old} → {severity}")
    db.commit()
    return inc


def add_note(db: Session, inc: Incident, actor: str, message: str) -> None:
    log_event(db, inc, actor, "note", message)
    db.commit()


# --------------------------------------------------------------------------------------
# Resolve + learn
# --------------------------------------------------------------------------------------


def resolve(db: Session, inc: Incident, data: dict[str, Any]) -> Resolution:
    cat = data["root_cause_category"]
    if cat not in ROOT_CAUSES:
        raise WorkflowError("Unknown root-cause category.")
    ts = now(db)
    diagnosed = inc.diagnosed_category
    feedback = data.get("diagnosis_feedback") or (
        "correct" if diagnosed == cat else "incorrect" if diagnosed else "not_diagnosed")
    ttm = data.get("time_to_mitigate_min")
    if ttm is None and inc.mitigated_at:
        ttm = int((inc.mitigated_at - inc.created_at).total_seconds() // 60)
    res = Resolution(
        incident_id=inc.id, root_cause_category=cat, root_cause_detail=data.get("root_cause_detail", ""),
        fix_applied=data.get("fix_applied", ""), prevention=data.get("prevention", ""), diagnosis_feedback=feedback,
        time_to_mitigate_min=ttm, tags=data.get("tags", []), resolved_by=data.get("resolved_by", "FDE"), created_at=ts,
    )
    if inc.resolution is not None:
        db.delete(inc.resolution)
        db.flush()
    db.add(res)
    inc.status = "resolved"
    inc.resolved_at = ts
    if inc.signature is None:
        inc.signature = signature({}, get_model(db, inc.model_id)["model_type"])
    for a in inc.alerts:
        a.status = "resolved"
    for a in inc.actions:
        if a.kind == "prevent" and "feedback loop" in a.title and a.status != "done":
            a.status = "done"
            a.updated_at = ts
    verdict = {"correct": "matched", "partially_correct": "partially matched", "incorrect": "did not match"}.get(
        feedback, "n/a")
    log_event(db, inc, res.resolved_by, "resolution",
              f"Resolved. Root cause: {label(cat)}. Diagnosis {verdict} ({label(diagnosed)}). Added to knowledge base.",
              {"root_cause_category": cat, "diagnosis_feedback": feedback}, ts=ts)
    db.commit()
    return res
