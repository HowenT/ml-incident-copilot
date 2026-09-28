"""Build the demo environment from scratch.

    python -m backend.app.seed            # data ends "now"
    python -m backend.app.seed --anchor 2026-09-28T12:00

Trains both models, simulates 30 days of traffic with three incidents, runs the monitoring
job and alert rules, loads the knowledge base, and opens one incident (TravelPlus) that a
teammate is already investigating. The credit and latency incidents are left for the demo.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
from sqlalchemy import inspect, select, text

from .config import CLIENT_NAME, MODEL_ARTIFACT_DIR, RANDOM_SEED, utcnow
from .db import Base, SessionLocal, engine
from .models import Alert, Incident, IncidentEvent, LogEvent, Meta, MLModel, Resolution
from .services import simulator as sim
from .services import store
from .services.knowledge_seed import HISTORICAL_INCIDENTS, SAMPLE_REPORTS
from .services.monitoring import build_reference_profile, compute_monitoring, detect_alerts
from .services.workflow import create_incident, diagnose, log_event, update_action


def _reset_schema() -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        for t in insp.get_table_names():
            if t.startswith("pred_"):
                conn.execute(text(f'DROP TABLE IF EXISTS "{t}"'))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def seed(anchor: datetime | None = None, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(RANDOM_SEED)
    tl = sim.build_timeline(anchor)
    store.clear_cache()
    _reset_schema()

    # 1. Train the two production models on a reference population.
    say("• training models …")
    cref = sim.credit_population(rng, 30_000)
    cmodel = sim.train(sim.CREDIT_FEATURES, cref)
    cref_scores = sim.score(cmodel, sim.CREDIT_FEATURES, cref)
    c_thr = float(np.quantile(cref_scores, 0.62))  # approve if PD below cut-off -> ~62% approval
    fref = sim.fraud_reference(rng, 80_000)
    fmodel = sim.train(sim.FRAUD_FEATURES, fref)
    fref_scores = sim.score(fmodel, sim.FRAUD_FEATURES, fref)
    f_thr = float(np.quantile(fref_scores, 0.98))  # block top 2% of risk
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cmodel, MODEL_ARTIFACT_DIR / "credit-risk-3.4.1.joblib")
    joblib.dump(fmodel, MODEL_ARTIFACT_DIR / "fraud-detect-7.x.joblib")

    c_profile = build_reference_profile(sim.CREDIT_FEATURES, cref, cref_scores)
    c_profile["__decision_rate__"] = float((cref_scores < c_thr).mean())
    f_profile = build_reference_profile(sim.FRAUD_FEATURES, fref, fref_scores)
    f_profile["__decision_rate__"] = float((fref_scores >= f_thr).mean())
    registry = [
        MLModel(id="credit-risk", display_name="Credit default risk (PD)", model_type="credit", client=CLIENT_NAME,
                use_case="Scores consumer loan applications in real time; applications below the PD cut-off are "
                         "auto-approved, the rest are declined or referred.",
                current_version="3.4.1", owner="ML Platform · credit squad", decision_name="Approval rate",
                threshold=c_thr, label_lag_hours=14 * 24, latency_slo_ms=300.0, log_sample_rate=1.0, features=sim.CREDIT_FEATURES,
                segments=sim.CREDIT_SEGMENTS,
                reference_profile=c_profile,
                prediction_table="pred_credit_risk"),
        MLModel(id="fraud-detect", display_name="Card transaction fraud", model_type="fraud", client=CLIENT_NAME,
                use_case="Scores card authorisations in the payment path; transactions above the threshold are "
                         "blocked. On timeout the gateway applies fallback policy ALLOW.",
                current_version="7.3.0", owner="ML Platform · fraud squad", decision_name="Block rate",
                threshold=f_thr, label_lag_hours=48, latency_slo_ms=250.0, log_sample_rate=0.1, features=sim.FRAUD_FEATURES,
                segments=sim.FRAUD_SEGMENTS,
                reference_profile=f_profile,
                prediction_table="pred_fraud_detect"),
    ]

    # 2. Simulate production traffic.
    say("• simulating 30 days of traffic …")
    frames = {"credit-risk": sim.simulate_credit(rng, tl, cmodel, c_thr),
              "fraud-detect": sim.simulate_fraud(rng, tl, fmodel, f_thr)}

    with SessionLocal() as db:
        db.add_all(registry)
        db.commit()
        for m in registry:
            frames[m.id].to_sql(m.prediction_table, engine, index=False, if_exists="replace", chunksize=5000)

        # 3. Monitoring job + alert rules.
        say("• computing monitoring metrics and alerts …")
        n_alerts = 0
        for m in registry:
            md = store.model_dict(m)
            metrics, feats, perf = compute_monitoring(md, frames[m.id], tl.start, tl.end)
            for df, table in ((metrics, "metrics_hourly"), (feats, "feature_stats_hourly"), (perf, "performance_daily")):
                df = df.copy()
                df.insert(0, "model_id", m.id)
                df.to_sql(table, engine, index=False, if_exists="append", chunksize=5000)
            for a in detect_alerts(md, metrics, feats, perf, tl.end):
                db.add(Alert(**a))
                n_alerts += 1
        logs = sim.generate_logs(rng, tl, frames["credit-risk"], frames["fraud-detect"])
        db.add_all(LogEvent(**r) for r in logs)
        db.add_all([
            Meta(key="data_start", value=tl.start.isoformat()),
            Meta(key="data_end", value=tl.end.isoformat()),
            Meta(key="seeded_at", value=utcnow().isoformat()),
            Meta(key="client", value=CLIENT_NAME),
            # Ground truth for tests only — the app never reads it.
            Meta(key="ground_truth", value=json.dumps({
                "credit-risk": {"onset": tl.onset_bureau.isoformat(), "category": "upstream_data_change"},
                "fraud-detect/latency": {"onset": tl.onset_deploy.isoformat(), "category": "model_release_regression"},
                "fraud-detect/travelplus": {"onset": tl.onset_travelplus.isoformat(), "category": "population_drift"},
            })),
        ])
        db.commit()

        # 4. Knowledge base: incidents resolved before this month.
        say("• loading knowledge base …")
        for h in HISTORICAL_INCIDENTS:
            r = h["resolution"]
            created = h["created_at"]
            resolved = created + timedelta(minutes=(r["time_to_mitigate_min"] or 60 * 24 * 9) + 180)
            inc = Incident(code=h["code"], title=h["title"], model_id=h["model_id"], severity=h["severity"],
                           status="resolved", customer=CLIENT_NAME, customer_impact=h["customer_impact"],
                           business_context=h["business_context"], affected_segments=h["affected_segments"],
                           reported_by=r["resolved_by"], customer_contact="", created_at=created, updated_at=resolved,
                           resolved_at=resolved, historical=1, diagnosed_category=h["diagnosed_category"],
                           diagnosed_confidence=h["diagnosed_confidence"], signature=h["signature"])
            db.add(inc)
            db.flush()
            db.add(Resolution(incident_id=inc.id, created_at=resolved, **r))
            db.add(IncidentEvent(incident_id=inc.id, ts=created, actor=r["resolved_by"], kind="created",
                                 message="Incident opened (imported from previous tracker)."))
            db.add(IncidentEvent(incident_id=inc.id, ts=resolved, actor=r["resolved_by"], kind="resolution",
                                 message=f"Resolved: {r['root_cause_detail']}"))
        db.commit()

        # 5. A teammate is already working the TravelPlus incident.
        say("• opening the in-flight TravelPlus incident …")
        drift_rules = {"feature_drift", "decision_rate_shift", "performance_drop", "prediction_drift"}
        tp_alerts = db.scalars(select(Alert).where(Alert.model_id == "fraud-detect",
                                                   Alert.started_at >= tl.onset_travelplus - timedelta(hours=6),
                                                   Alert.started_at < tl.onset_deploy - timedelta(hours=12))).all()
        tp_alerts = [a for a in tp_alerts if a.rule in drift_rules]
        opened = tl.onset_deploy - timedelta(hours=3)  # before the latency incident starts
        report = dict(SAMPLE_REPORTS["fraud-travelplus"])
        inc = create_incident(db, {**report, "alert_ids": [a.id for a in tp_alerts], "model_id": "fraud-detect"},
                              created_at=opened)
        diagnose(db, inc, ts=opened + timedelta(minutes=6))
        verify = next(a for a in inc.actions if a.kind == "verify")
        update_action(db, inc, verify.id, "done", "Maya Chen (FDE)")
        ev = db.scalars(select(IncidentEvent).where(IncidentEvent.incident_id == inc.id)
                        .order_by(IncidentEvent.id.desc())).first()
        ev.ts = opened + timedelta(minutes=48)
        log_event(db, inc, "Maya Chen (FDE)", "note",
                  "Confirmed with Harborline: TravelPlus cardholders are KYC-verified at onboarding. Waiting on fraud "
                  "ops to approve step-up instead of hard declines.", ts=opened + timedelta(minutes=55))
        db.commit()
        alerts = db.scalars(select(Alert)).all()

    summary = {"data_start": tl.start.isoformat(), "data_end": tl.end.isoformat(),
               "predictions": {k: len(v) for k, v in frames.items()}, "alerts": n_alerts, "logs": len(logs),
               "seconds": round(time.perf_counter() - t0, 1)}
    say(f"✓ seeded in {summary['seconds']}s — {summary['predictions']} predictions, {n_alerts} alerts, {len(logs)} log lines")
    for a in alerts:
        say(f"   [{a.status:12}] {a.model_id:12} {a.rule:20} {a.started_at:%m-%d %H:%M} {a.title}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchor", help="ISO timestamp for the end of the simulated data (default: now, UTC)")
    args = parser.parse_args()
    seed(datetime.fromisoformat(args.anchor) if args.anchor else None)


if __name__ == "__main__":
    main()
