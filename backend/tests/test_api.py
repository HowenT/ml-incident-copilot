"""End-to-end workflow through the HTTP API: act -> communicate -> resolve -> learn."""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client(scenario_incidents):
    with TestClient(app) as c:
        yield c


def test_health_and_overview(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["llm_enabled"] is False
    ov = client.get("/api/overview").json()
    assert {m["id"] for m in ov["models"]} == {"credit-risk", "fraud-detect"}
    assert all(m["health"] != "healthy" for m in ov["models"])


def test_monitoring_endpoints_are_json_safe(client):
    for mid in ("credit-risk", "fraud-detect"):
        assert client.get(f"/api/models/{mid}/metrics?days=7").status_code == 200
        feats = client.get(f"/api/models/{mid}/features?days=7").json()
        assert feats["daily_psi"]["features"]
        assert client.get(f"/api/models/{mid}/performance").status_code == 200
    assert client.get("/api/logs?level=WARN,ERROR&limit=20").status_code == 200


def test_create_incident_validates_alerts(client):
    alerts = client.get("/api/alerts").json()
    linked = next(a for a in alerts if a["incident_id"])
    r = client.post("/api/incidents", json={"title": "dup", "alert_ids": [linked["id"]]})
    assert r.status_code == 400
    credit = next(a for a in alerts if a["model_id"] == "credit-risk")
    fraud = next(a for a in alerts if a["model_id"] == "fraud-detect" and not a["incident_id"])
    r = client.post("/api/incidents", json={"title": "mixed", "alert_ids": [credit["id"], fraud["id"]]})
    assert r.status_code == 400


def test_sample_report(client):
    alerts = client.get("/api/alerts?model_id=credit-risk").json()
    r = client.get(f"/api/alerts/sample-report?alert_ids={alerts[0]['id']}").json()
    assert "West" in r["title"]


def test_full_incident_lifecycle(client, scenario_incidents):
    iid = scenario_incidents["credit-bureau"]["id"]
    inc = client.get(f"/api/incidents/{iid}").json()
    assert inc["status"] == "investigating"
    assert set(inc["summaries"]) == {"engineer", "customer"}
    assert "Kestrel" in inc["summaries"]["customer"]["content"]
    assert "PSI" not in inc["summaries"]["customer"]["content"]  # no jargon for the client

    # LLM polish is optional; without credentials the API says so and keeps the template.
    assert client.post(f"/api/incidents/{iid}/summaries/regenerate", json={"use_llm": True}).status_code == 503

    # Completing every P0 mitigation moves the incident to 'mitigated'.
    for a in inc["actions"]:
        if a["priority"] == "P0" and a["kind"] == "mitigate":
            client.patch(f"/api/incidents/{iid}/actions/{a['id']}", json={"status": "done", "actor": "test"})
    inc = client.get(f"/api/incidents/{iid}").json()
    assert inc["status"] == "mitigated"

    inc = client.post(f"/api/incidents/{iid}/summaries/customer/sent").json()
    assert inc["summaries"]["customer"]["sent_at"]

    before = client.get("/api/knowledge/stats").json()
    r = client.post(f"/api/incidents/{iid}/resolve", json={
        "root_cause_category": "upstream_data_change",
        "root_cause_detail": "Kestrel v2 renamed bureau_score to riskScoreV2; connector emitted nulls.",
        "fix_applied": "Connector mapping patched; 640 approvals re-reviewed.",
        "tags": ["schema-change"], "resolved_by": "test"})
    assert r.status_code == 200 and r.json()["status"] == "resolved"
    assert r.json()["resolution"]["diagnosis_feedback"] == "correct"
    after = client.get("/api/knowledge/stats").json()
    assert after["resolved"] == before["resolved"] + 1
    assert after["top1_accuracy"] >= before["top1_accuracy"]

    kinds = [e["kind"] for e in r.json()["events"]]
    for k in ("created", "alerts_linked", "diagnosis", "action", "communication", "resolution"):
        assert k in kinds

    hits = client.get("/api/knowledge?q=kestrel bureau schema").json()
    assert hits and hits[0]["id"] == iid


def test_resolved_incident_feeds_future_similarity(client, scenario_incidents):
    fid = scenario_incidents["fraud-latency"]["id"]
    sims = client.get(f"/api/incidents/{fid}/similar").json()
    assert sims[0]["root_cause_category"] in ("model_release_regression", "serving_infrastructure")
