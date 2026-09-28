"""The diagnosis engine must recover each injected root cause, its onset, and the key evidence."""
from datetime import datetime, timedelta

from sqlalchemy import select

from backend.app.db import SessionLocal
from backend.app.models import Alert, Incident
from backend.app.services.workflow import latest_diagnosis


def _onset_error_h(diag, truth_iso):
    return abs((datetime.fromisoformat(diag["onset"]) - datetime.fromisoformat(truth_iso)).total_seconds()) / 3600


def _titles(diag):
    return " | ".join(e["title"] for e in diag["evidence"])


def test_seed_produces_expected_alerts(seeded):
    rules = seeded["alerts_at_seed"]
    assert ("credit-risk", "missing_values", True) in rules
    assert ("credit-risk", "decision_rate_shift", True) in rules
    assert ("fraud-detect", "latency_slo", True) in rules
    assert ("fraud-detect", "serving_errors", True) in rules
    assert ("fraud-detect", "feature_drift", True) in rules
    # the network blip auto-resolved
    assert ("fraud-detect", "latency_slo", False) in rules


def test_credit_bureau_incident_is_upstream_data_change(seeded, scenario_incidents):
    d = scenario_incidents["credit-bureau"]["diagnosis"]
    top = d["hypotheses"][0]
    assert top["category"] == "upstream_data_change"
    assert top["confidence"] > 0.8
    assert _onset_error_h(d, seeded["credit-risk"]["onset"]) <= 1
    assert d["facts"]["missing_segment"]["value"] == "kestrel_v2"
    assert "CHG-8812" in _titles(d) + str(d["facts"]["changes"])
    assert d["impact"]["numbers"]["excess_approvals"] > 0


def test_release_incident_is_model_release_regression(seeded, scenario_incidents):
    d = scenario_incidents["fraud-latency"]["diagnosis"]
    top = d["hypotheses"][0]
    assert top["category"] == "model_release_regression"
    assert d["hypotheses"][1]["category"] == "serving_infrastructure"  # the plausible runner-up
    assert _onset_error_h(d, seeded["fraud-detect/latency"]["onset"]) <= 1
    assert d["facts"]["new_version"] == "7.3.0"
    assert d["impact"]["numbers"]["fallback"] > 0


def test_travelplus_incident_is_population_drift(seeded):
    with SessionLocal() as db:
        inc = db.scalars(select(Incident).where(Incident.historical == 0, Incident.model_id == "fraud-detect",
                                                Incident.title.contains("TravelPlus"))).one()
        d = latest_diagnosis(db, inc)
    assert d["hypotheses"][0]["category"] == "population_drift"
    assert d["facts"]["drift_segment"]["value"] == "travelplus"
    assert _onset_error_h(d, seeded["fraud-detect/travelplus"]["onset"]) <= 12  # gradual ramp
    # the later release incident must not leak into this diagnosis
    assert "latency_spike" not in d["signals"]


def test_actions_are_evidence_linked_and_prioritised(scenario_incidents):
    with SessionLocal() as db:
        inc = db.get(Incident, scenario_incidents["credit-bureau"]["id"])
        acts = inc.actions
        assert acts[0].priority == "P0"
        kinds = {a.kind for a in acts}
        assert {"verify", "mitigate", "fix", "prevent", "communicate"} <= kinds
        assert any(not a.ours for a in acts), "client-owned actions become 'what we need from you'"
        assert all(a.rationale for a in acts)


def test_resolved_network_blip_diagnoses_as_infrastructure(seeded):
    from backend.app.services.workflow import create_incident, diagnose

    with SessionLocal() as db:
        blip = [a for a in db.scalars(select(Alert).where(Alert.model_id == "fraud-detect", Alert.status == "resolved",
                                                          Alert.incident_id.is_(None)))
                if a.rule in ("latency_slo", "serving_errors")
                and a.started_at < datetime.fromisoformat(seeded["fraud-detect/travelplus"]["onset"]) - timedelta(days=1)]
        inc = create_incident(db, {"title": "Short latency spike", "alert_ids": [a.id for a in blip]})
        d = diagnose(db, inc)
    assert d["hypotheses"][0]["category"] == "serving_infrastructure"
