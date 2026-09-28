"""Streamlit front end.   streamlit run frontend/app.py"""
from __future__ import annotations

import streamlit as st

import api
import ui

st.set_page_config(page_title="ML Incident Copilot", page_icon=":material/troubleshoot:", layout="wide")
ui.inject_css()

pages = {
    "Monitor": [
        st.Page("views/monitoring.py", title="Model health", icon=":material/monitor_heart:", default=True),
        st.Page("views/alerts.py", title="Alerts & triage", icon=":material/notifications_active:"),
    ],
    "Respond": [
        st.Page("views/incidents.py", title="Incidents", icon=":material/list_alt:"),
        st.Page("views/incident.py", title="Incident workspace", icon=":material/troubleshoot:"),
    ],
    "Learn": [
        st.Page("views/knowledge.py", title="Knowledge base", icon=":material/menu_book:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### ML Incident Copilot")
    try:
        h = api.get("/api/health")
        st.markdown(f"<div class='sub'>Client: <b>{ui.esc(h['client'])}</b></div>"
                    f"<div class='sub'>Data as of {ui.fmt_dt(h['data_end'])} UTC</div>", unsafe_allow_html=True)
        llm = "on" if h["llm_enabled"] else "off (template mode)"
        st.markdown(f"<div class='sub'>Claude polish: {llm}</div>", unsafe_allow_html=True)
    except api.ApiError as e:
        st.error(e.detail)
        st.caption("Start the API:  `uvicorn backend.app.main:app`")
        st.stop()
    st.divider()
    with st.popover("Reset demo data", icon=":material/restart_alt:", width="stretch"):
        st.caption("Regenerates 30 days of data ending now and discards incidents you created. Takes ~15 s.")
        if st.button("Reset now", type="primary"):
            with st.spinner("Re-seeding …"):
                api.guard(api.post, "/api/admin/reseed")
            api.clear_cache()
            st.session_state.clear()
            st.rerun()

nav.run()
