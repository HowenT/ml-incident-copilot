"""Alert triage: pick related alerts, capture the customer's report, open a tracked incident."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api
import ui

st.title("Alerts & triage")
st.markdown("<div class='sub'>Group the alerts that describe one problem, add what the customer is telling you, "
            "and open a tracked incident.</div>", unsafe_allow_html=True)
st.write("")

f1, f2, _ = st.columns([3, 3, 4], vertical_alignment="bottom")
scope = f1.segmented_control("Show", ["Needs triage", "All"], default="Needs triage", key="al_scope") or "Needs triage"
models = {m["id"]: m for m in api.cached("/api/models")}
model_filter = f2.segmented_control("Model", ["All", *models], default="All", key="al_model") or "All"

alerts = api.get("/api/alerts", model_id=None if model_filter == "All" else model_filter)
if scope == "Needs triage":
    alerts = [a for a in alerts if a["status"] != "resolved" and not a["incident_id"]]

if not alerts:
    st.success("Nothing to triage — every active alert is already attached to an incident.")
    st.stop()

rows = pd.DataFrame([{
    "Severity": f"{ui.SEVERITY[a['severity']][1]} {ui.SEVERITY[a['severity']][2]}",
    "Model": a["model_id"], "Alert": a["title"], "Started": ui.fmt_dt(a["started_at"]),
    "Status": a["status"].capitalize() + (" (auto)" if a["status"] == "resolved" else ""),
    "Incident": a["incident_code"] or "—",
} for a in alerts])
st.caption("Select one or more alerts from the same model.")
event = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun", selection_mode="multi-row",
                     key="al_table", column_config={"Alert": st.column_config.TextColumn(width="large")})
picked = [alerts[i] for i in event.selection.rows]

if not picked:
    st.stop()

model_ids = {a["model_id"] for a in picked}
if len(model_ids) > 1:
    st.warning("Pick alerts from a single model — one incident per affected model keeps ownership clear.")
    st.stop()
if any(a["incident_id"] for a in picked):
    st.info("Some selected alerts are already linked to an incident.")
    st.stop()

model = models[next(iter(model_ids))]

# ---- preview of the selected signals -------------------------------------------------
st.subheader(f"{len(picked)} alert(s) on {model['display_name']}")
met = api.cached(f"/api/models/{model['id']}/metrics", days=7)["rows"]
metric_for = {"missing_values": ("max_missing_rate", "pct", "Max missing rate"),
              "decision_rate_shift": ("decision_rate_24h", "pct", f"{model['decision_name']} (24h)"),
              "latency_slo": ("latency_p95", "ms", "p95 latency"),
              "serving_errors": ("timeout_rate", "pct", "Fallback rate"),
              "feature_drift": ("max_psi_24h", "num", "Max input PSI (24h)"),
              "prediction_drift": ("score_psi_24h", "num", "Score PSI (24h)"),
              "performance_drop": ("decision_rate_24h", "pct", f"{model['decision_name']} (24h)")}
pc = st.columns(min(len(picked), 3))
for i, a in enumerate(picked[:3]):
    col, fmt, label = metric_for[a["rule"]]
    band = [{"x0": max(a["started_at"], met["ts"][0]), "x1": a["ended_at"] or met["ts"][-1],
             "color": ui.SEVERITY[a["severity"]][0]}]
    with pc[i].container(border=True):
        st.markdown(ui.severity_pill(a["severity"]) + f" <span class='sub'>{ui.esc(a['title'])}</span>",
                    unsafe_allow_html=True)
        ui.plot(ui.line_chart(met["ts"], {label: met[col]}, fmt, None, 180, bands=band,
                              threshold=a["threshold"] if a["rule"] != "decision_rate_shift" else None),
                key=f"al_prev_{a['id']}")

# ---- incident form -------------------------------------------------------------------
FIELDS = ["title", "severity", "customer_impact", "business_context", "affected_segments", "reported_by",
          "customer_contact"]
sel_key = ",".join(str(a["id"]) for a in picked)
if st.session_state.get("form_for") != sel_key:
    st.session_state["form_for"] = sel_key
    worst = max(picked, key=lambda a: ["low", "medium", "high", "critical"].index(a["severity"]))
    st.session_state["f_title"] = worst["title"].split(":")[0].split("(")[0].strip()
    st.session_state["f_severity"] = {"critical": "SEV1", "high": "SEV2"}.get(worst["severity"], "SEV3")
    for f in FIELDS[2:]:
        st.session_state[f"f_{f}"] = ""


def fill_sample() -> None:
    try:
        sample = api.get("/api/alerts/sample-report", alert_ids=sel_key)
    except api.ApiError:
        return
    for f in FIELDS:
        st.session_state[f"f_{f}"] = sample.get(f, "")


st.subheader("Open an incident")
st.button("Use sample customer report", icon=":material/edit_note:", on_click=fill_sample,
          help="Fills the form with a realistic report from the client for this demo scenario.")
with st.form("incident_form", border=True):
    c1, c2 = st.columns([4, 1])
    c1.text_input("Title", key="f_title")
    c2.selectbox("Severity", ["SEV1", "SEV2", "SEV3"], key="f_severity")
    st.text_area("Customer impact — what is the client seeing?", key="f_customer_impact", height=90)
    st.text_area("Business context — recent changes, launches, deadlines", key="f_business_context", height=80)
    c3, c4, c5 = st.columns(3)
    c3.text_input("Affected segments", key="f_affected_segments")
    c4.text_input("Reported by", key="f_reported_by")
    c5.text_input("Customer contact", key="f_customer_contact")
    run_now = st.checkbox("Run the diagnosis right away", value=True)
    submitted = st.form_submit_button("Open incident", type="primary", icon=":material/add_alert:")

if submitted:
    body = {f: st.session_state[f"f_{f}"] for f in FIELDS}
    body["alert_ids"] = [a["id"] for a in picked]
    inc = api.guard(api.post, "/api/incidents", body)
    if inc:
        if run_now:
            with st.spinner("Diagnosing: data quality, drift, decisions, labels, serving, changes, logs …"):
                api.guard(api.post, f"/api/incidents/{inc['id']}/diagnose")
        st.session_state["incident_id"] = inc["id"]
        st.session_state["ws_view"] = "Diagnosis" if run_now else "Overview"
        st.session_state.pop("form_for", None)
        api.clear_cache()
        st.switch_page("views/incident.py")
