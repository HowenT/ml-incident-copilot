"""ORM tables.

Raw prediction logs live in per-model wide tables (``pred_<model_id>``) written with
pandas, because each model has its own feature columns. Everything else is here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class MLModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128))
    model_type: Mapped[str] = mapped_column(String(16))  # credit | fraud
    client: Mapped[str] = mapped_column(String(128))
    use_case: Mapped[str] = mapped_column(Text)
    current_version: Mapped[str] = mapped_column(String(32))
    owner: Mapped[str] = mapped_column(String(128))
    decision_name: Mapped[str] = mapped_column(String(64))  # "Approval rate" / "Block rate"
    threshold: Mapped[float] = mapped_column(Float)
    label_lag_hours: Mapped[int] = mapped_column(Integer)
    latency_slo_ms: Mapped[float] = mapped_column(Float)
    log_sample_rate: Mapped[float] = mapped_column(Float, default=1.0)  # share of requests kept in the prediction log
    features: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    segments: Mapped[list[str]] = mapped_column(JSON)
    reference_profile: Mapped[dict[str, Any]] = mapped_column(JSON)
    prediction_table: Mapped[str] = mapped_column(String(64))


class MetricHourly(Base):
    __tablename__ = "metrics_hourly"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    volume: Mapped[int] = mapped_column(Integer)
    mean_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_rate: Mapped[float] = mapped_column(Float)
    decision_rate_24h: Mapped[float] = mapped_column(Float)
    score_psi_24h: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_p50: Mapped[float] = mapped_column(Float)
    latency_p95: Mapped[float] = mapped_column(Float)
    latency_p99: Mapped[float] = mapped_column(Float)
    timeout_rate: Mapped[float] = mapped_column(Float)
    max_missing_rate: Mapped[float] = mapped_column(Float)
    max_psi_24h: Mapped[float] = mapped_column(Float)


class FeatureStatHourly(Base):
    __tablename__ = "feature_stats_hourly"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    feature: Mapped[str] = mapped_column(String(64))
    missing_rate: Mapped[float] = mapped_column(Float)
    psi_24h: Mapped[float] = mapped_column(Float)


class PerformanceDaily(Base):
    __tablename__ = "performance_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    day: Mapped[datetime] = mapped_column(DateTime)
    labeled: Mapped[int] = mapped_column(Integer)
    positives: Mapped[int] = mapped_column(Integer)
    auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)


class LogEvent(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    model_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    service: Mapped[str] = mapped_column(String(64))
    level: Mapped[str] = mapped_column(String(8))  # INFO | WARN | ERROR
    event_type: Mapped[str] = mapped_column(String(16))  # change | error | info | business
    message: Mapped[str] = mapped_column(Text)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    rule: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(256))
    metric: Mapped[str] = mapped_column(String(64))
    features: Mapped[list[str]] = mapped_column(JSON, default=list)
    severity: Mapped[str] = mapped_column(String(16))  # critical | high | medium | low
    status: Mapped[str] = mapped_column(String(16))  # open | acknowledged | resolved
    started_at: Mapped[datetime] = mapped_column(DateTime)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    peak_value: Mapped[float] = mapped_column(Float)
    baseline_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float] = mapped_column(Float)
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"), nullable=True)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    title: Mapped[str] = mapped_column(String(256))
    model_id: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(8))  # SEV1 | SEV2 | SEV3
    status: Mapped[str] = mapped_column(String(16))  # open | investigating | mitigated | resolved
    customer: Mapped[str] = mapped_column(String(128))
    customer_impact: Mapped[str] = mapped_column(Text, default="")
    business_context: Mapped[str] = mapped_column(Text, default="")
    affected_segments: Mapped[str] = mapped_column(Text, default="")
    reported_by: Mapped[str] = mapped_column(String(128), default="")
    customer_contact: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    mitigated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    historical: Mapped[int] = mapped_column(Integer, default=0)
    # Filled by the latest diagnosis; used by similarity search and the knowledge base.
    diagnosed_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diagnosed_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    signature: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)

    alerts: Mapped[list[Alert]] = relationship(foreign_keys=[Alert.incident_id])
    events: Mapped[list["IncidentEvent"]] = relationship(
        back_populates="incident", order_by="IncidentEvent.ts", cascade="all, delete-orphan"
    )
    actions: Mapped[list["Action"]] = relationship(
        back_populates="incident", order_by="Action.sort_order", cascade="all, delete-orphan"
    )
    resolution: Mapped["Resolution | None"] = relationship(
        back_populates="incident", uselist=False, cascade="all, delete-orphan"
    )


class IncidentEvent(Base):
    __tablename__ = "incident_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    actor: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    incident: Mapped[Incident] = relationship(back_populates="events")


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    priority: Mapped[str] = mapped_column(String(4))  # P0 | P1 | P2
    kind: Mapped[str] = mapped_column(String(16))  # verify | mitigate | fix | prevent | communicate
    title: Mapped[str] = mapped_column(String(256))
    detail: Mapped[str] = mapped_column(Text)
    owner: Mapped[str] = mapped_column(String(128))
    rationale: Mapped[str] = mapped_column(Text)
    customer_text: Mapped[str] = mapped_column(Text, default="")  # plain-language line for the client update
    ours: Mapped[int] = mapped_column(Integer, default=1)  # 1 = our team owns it, 0 = client-owned
    status: Mapped[str] = mapped_column(String(16), default="todo")  # todo | in_progress | done | skipped
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    incident: Mapped[Incident] = relationship(back_populates="actions")


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), index=True)
    audience: Mapped[str] = mapped_column(String(16))  # engineer | customer
    content: Mapped[str] = mapped_column(Text)
    generator: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Resolution(Base):
    __tablename__ = "resolutions"

    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id"), primary_key=True)
    root_cause_category: Mapped[str] = mapped_column(String(64))
    root_cause_detail: Mapped[str] = mapped_column(Text)
    fix_applied: Mapped[str] = mapped_column(Text)
    prevention: Mapped[str] = mapped_column(Text, default="")
    diagnosis_feedback: Mapped[str] = mapped_column(String(32))  # correct | partially_correct | incorrect
    time_to_mitigate_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    resolved_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime)

    incident: Mapped[Incident] = relationship(back_populates="resolution")


class Meta(Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
