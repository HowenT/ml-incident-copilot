"""Incident queue."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api
import ui

st.title("Incidents")
f1, _ = st.columns([4, 6])
show = f1.segmented_control("Show", ["Active", "All this month"], default="Active", key="inc_scope") or "Active"
all_incs = api.get("/api/incidents")
incs = [i for i in all_incs if i["status"] != "resolved"] if show == "Active" else all_incs

c = st.columns(4)
by = {s: sum(1 for i in all_incs if i["status"] == s) for s in ("open", "investigating", "mitigated", "resolved")}
for col, s in zip(c, by):
    col.metric(ui.INC_STATUS[s][1], by[s], border=True)

if not incs:
    st.info("No active incidents. Open one from **Alerts & triage**.")
    st.page_link("views/alerts.py", label="Go to Alerts & triage", icon=":material/notifications_active:")
    st.stop()

rows = pd.DataFrame([{
    "Code": i["code"], "Title": i["title"], "Sev": i["severity"], "Status": ui.INC_STATUS[i["status"]][1],
    "Model": i["model_id"], "Likely cause": (f"{i['diagnosed_label']} ({i['diagnosed_confidence']:.0%})"
                                             if i["diagnosed_label"] else "not diagnosed"),
    "Open actions": f"{i['actions_open']}/{i['actions_total']}", "Age": ui.fmt_age(i["age_hours"]),
    "Opened": ui.fmt_dt(i["created_at"]),
} for i in incs])
st.caption("Select an incident to open its workspace.")
ev = st.dataframe(rows, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                  key="inc_table", column_config={"Title": st.column_config.TextColumn(width="large")})
if ev.selection.rows:
    st.session_state["incident_id"] = incs[ev.selection.rows[0]]["id"]
    st.session_state["ws_view"] = "Overview"
    st.switch_page("views/incident.py")
