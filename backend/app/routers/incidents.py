"""Incident workflow API."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import serializers as ser
from ..db import get_db
from ..models import Incident
from ..services import workflow as wf
from ..services.llm import LLMUnavailable
from ..services.similarity import find_similar

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


class IncidentIn(BaseModel):
    title: str = Field(min_length=3, max_length=256)
    alert_ids: list[int] = []
    model_id: str | None = None
    severity: Literal["SEV1", "SEV2", "SEV3"] = "SEV2"
    customer_impact: str = ""
    business_context: str = ""
    affected_segments: str = ""
    reported_by: str = ""
    customer_contact: str = ""


class IncidentPatch(BaseModel):
    status: Literal["open", "investigating", "mitigated"] | None = None
    severity: Literal["SEV1", "SEV2", "SEV3"] | None = None
    actor: str = "FDE"


class NoteIn(BaseModel):
    actor: str = "FDE"
    message: str = Field(min_length=1)


class ActionPatch(BaseModel):
    status: Literal["todo", "in_progress", "done", "skipped"]
    actor: str = "FDE"


class RegenerateIn(BaseModel):
    use_llm: bool = False
    actor: str = "FDE"


class ResolveIn(BaseModel):
    root_cause_category: str
    root_cause_detail: str = ""
    fix_applied: str = ""
    prevention: str = ""
    diagnosis_feedback: Literal["correct", "partially_correct", "incorrect"] | None = None
    time_to_mitigate_min: int | None = None
    tags: list[str] = []
    resolved_by: str = "FDE"


def _get(db: Session, incident_id: int) -> Incident:
    inc = db.get(Incident, incident_id)
    if inc is None:
        raise HTTPException(404, "Incident not found")
    return inc


def _detail(db: Session, inc: Incident) -> dict[str, Any]:
    db.refresh(inc)
    summaries = wf.latest_summaries(db, inc)
    return {
        **ser.incident(inc, wf.now(db)),
        "alerts": [ser.alert(a) for a in inc.alerts],
        "events": [ser.event(e) for e in inc.events],
        "actions": [ser.action(a) for a in inc.actions],
        "summaries": {k: ser.summary(v) for k, v in summaries.items()},
        "diagnosis": wf.latest_diagnosis(db, inc),
    }


@router.get("")
def list_incidents(status: str | None = None, include_historical: bool = False,
                   db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    q = select(Incident).order_by(Incident.created_at.desc())
    if not include_historical:
        q = q.where(Incident.historical == 0)
    if status:
        q = q.where(Incident.status.in_(status.split(",")))
    now = wf.now(db)
    return [ser.incident(i, now) for i in db.scalars(q)]


@router.post("", status_code=201)
def create(body: IncidentIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        inc = wf.create_incident(db, body.model_dump())
    except wf.WorkflowError as e:
        raise HTTPException(400, str(e))
    return _detail(db, inc)


@router.get("/{incident_id}")
def get_incident(incident_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    return _detail(db, _get(db, incident_id))


@router.patch("/{incident_id}")
def patch_incident(incident_id: int, body: IncidentPatch, db: Session = Depends(get_db)) -> dict[str, Any]:
    inc = _get(db, incident_id)
    try:
        if body.severity and body.severity != inc.severity:
            wf.set_severity(db, inc, body.severity, body.actor)
        if body.status and body.status != inc.status:
            wf.set_status(db, inc, body.status, body.actor)
    except wf.WorkflowError as e:
        raise HTTPException(400, str(e))
    return _detail(db, inc)


@router.post("/{incident_id}/notes")
def add_note(incident_id: int, body: NoteIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    inc = _get(db, incident_id)
    wf.add_note(db, inc, body.actor, body.message)
    return _detail(db, inc)


@router.post("/{incident_id}/diagnose")
def diagnose(incident_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    inc = _get(db, incident_id)
    if inc.historical:
        raise HTTPException(400, "Historical incidents have no raw data to diagnose.")
    wf.diagnose(db, inc)
    return _detail(db, inc)


@router.get("/{incident_id}/diagnosis")
def get_diagnosis(incident_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    d = wf.latest_diagnosis(db, _get(db, incident_id))
    if d is None:
        raise HTTPException(404, "Not diagnosed yet")
    return d


@router.patch("/{incident_id}/actions/{action_id}")
def patch_action(incident_id: int, action_id: int, body: ActionPatch, db: Session = Depends(get_db)) -> dict[str, Any]:
    inc = _get(db, incident_id)
    try:
        wf.update_action(db, inc, action_id, body.status, body.actor)
    except wf.WorkflowError as e:
        raise HTTPException(400, str(e))
    return _detail(db, inc)


@router.post("/{incident_id}/summaries/regenerate")
def regenerate(incident_id: int, body: RegenerateIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    inc = _get(db, incident_id)
    try:
        wf.regenerate_summaries(db, inc, body.use_llm, body.actor)
    except LLMUnavailable as e:
        raise HTTPException(503, str(e))
    except wf.WorkflowError as e:
        raise HTTPException(400, str(e))
    return _detail(db, inc)


@router.post("/{incident_id}/summaries/{audience}/sent")
def mark_sent(incident_id: int, audience: Literal["engineer", "customer"], actor: str = "FDE",
              db: Session = Depends(get_db)) -> dict[str, Any]:
    inc = _get(db, incident_id)
    try:
        wf.mark_summary_sent(db, inc, audience, actor)
    except wf.WorkflowError as e:
        raise HTTPException(400, str(e))
    return _detail(db, inc)


@router.get("/{incident_id}/similar")
def similar(incident_id: int, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    inc = _get(db, incident_id)
    return find_similar(db, inc, inc.signature, k=5)


@router.post("/{incident_id}/resolve")
def resolve(incident_id: int, body: ResolveIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    inc = _get(db, incident_id)
    try:
        wf.resolve(db, inc, body.model_dump())
    except wf.WorkflowError as e:
        raise HTTPException(400, str(e))
    return _detail(db, inc)
