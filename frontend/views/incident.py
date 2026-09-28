"""Incident workspace: overview -> diagnosis -> action plan -> updates -> resolve & learn."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import api
import ui

ACTOR = "Alex Rivera (FDE)"
VIEWS = ["Overview", "Diagnosis", "Action plan", "Updates", "Resolve & learn"]

incs = api.get("/api/incidents")
if not incs:
    st.title("Incident workspace")
    st.info("No incidents yet. Open one from **Alerts & triage**.")
    st.page_link("views/alerts.py", label="Go to Alerts & triage", icon=":material/notifications_active:")
    st.stop()

ids = [i["id"] for i in incs]
current = st.session_state.get("incident_id")
by_id = {i["id"]: i for i in incs}
iid = st.selectbox("Incident", ids, index=ids.index(current) if current in ids else 0,
                   format_func=lambda i: f"{by_id[i]['code']} · {by_id[i]['title']} ({by_id[i]['status']})",
                   label_visibility="collapsed")
st.session_state["incident_id"] = iid
inc = api.get(f"/api/incidents/{iid}")
health = api.cached("/api/health")
diag = inc["diagnosis"]


def refresh() -> None:
    api.clear_cache()
    st.rerun()


# ---- header --------------------------------------------------------------------------
h1, h2 = st.columns([7, 3], vertical_alignment="top")
with h1:
    st.markdown(f"<div class='kicker'>{ui.esc(inc['code'])} · {ui.esc(inc['model_id'])} · {ui.esc(inc['customer'])}</div>",
                unsafe_allow_html=True)
    st.markdown(f"## {ui.esc(inc['title'])}")
    meta = [ui.severity_pill(inc["severity"]), ui.status_pill(inc["status"]),
            f"<span class='sub'>opened {ui.fmt_dt(inc['created_at'])} UTC · {ui.fmt_age(inc['age_hours'])} "
            f"{'to resolve' if inc['resolved_at'] else 'ago'} · by {ui.esc(inc['reported_by'] or '—')}</span>"]
    st.markdown(" ".join(meta), unsafe_allow_html=True)
with h2:
    if inc["status"] != "resolved":
        label = "Re-run diagnosis" if diag else "Run diagnosis"
        if st.button(label, type="secondary" if diag else "primary", icon=":material/troubleshoot:", width="stretch"):
            with st.spinner("Checking data quality, drift, decisions, labels, serving, change events and logs …"):
                api.guard(api.post, f"/api/incidents/{iid}/diagnose")
            st.session_state["ws_view"] = "Diagnosis"
            refresh()
        c1, c2 = st.columns(2)
        new_status = c1.selectbox("Status", ["open", "investigating", "mitigated"],
                                  index=["open", "investigating", "mitigated"].index(inc["status"]),
                                  format_func=lambda s: ui.INC_STATUS[s][1], key=f"st_{iid}_{inc['status']}")
        new_sev = c2.selectbox("Severity", ["SEV1", "SEV2", "SEV3"], index=["SEV1", "SEV2", "SEV3"].index(inc["severity"]),
                               key=f"sev_{iid}_{inc['severity']}")  # keyed on server state so it never goes stale
        if new_status != inc["status"] or new_sev != inc["severity"]:
            api.guard(api.patch, f"/api/incidents/{iid}", {"status": new_status, "severity": new_sev, "actor": ACTOR})
            refresh()

order = ["open", "investigating", "mitigated", "resolved"]
pos = order.index(inc["status"])
st.markdown("<div class='stepper'>" + "".join(
    f"<div class='step {'done' if i < pos else 'now' if i == pos else ''}'>{ui.INC_STATUS[s][1]}</div>"
    for i, s in enumerate(order)) + "</div>", unsafe_allow_html=True)

if st.session_state.get("ws_view") not in VIEWS:
    st.session_state["ws_view"] = "Diagnosis" if diag else "Overview"
view = st.segmented_control("View", VIEWS, key="ws_view", label_visibility="collapsed") or "Overview"
st.write("")


# ---- helpers -------------------------------------------------------------------------
def evidence_card(ev: dict, contribution: float | None = None, key: str = "") -> None:
    with st.container(border=True):
        chip = f"<span class='contrib'>{contribution:+.2f}</span>" if contribution is not None else ""
        st.markdown(f"<div class='ev-title'>{ui.esc(ev['title'])}{chip}</div>"
                    f"<div class='ev-detail'>{ui.esc(ev['detail'])}</div>", unsafe_allow_html=True)
        if ev.get("chart"):
            ui.plot(ui.chart_from_spec(ev["chart"], 210), key=f"ev_{key}_{ev['id']}")


def impact_block(d: dict) -> None:
    st.markdown(f"**Blast radius.** {ui.esc(d['impact']['headline'])}")
    ms = d["impact"]["metrics"]
    cols = st.columns(min(len(ms), 3))
    for i, m in enumerate(ms):
        cols[i % len(cols)].metric(m["label"], m["value"], help=m["detail"] or None, border=True)


# ---- Overview ------------------------------------------------------------------------
if view == "Overview":
    left, right = st.columns([1, 1])
    with left.container(border=True):
        st.markdown("#### Customer report")
        st.markdown(f"**Impact.** {ui.esc(inc['customer_impact'] or '—')}")
        st.markdown(f"**Context.** {ui.esc(inc['business_context'] or '—')}")
        st.markdown(f"**Affected segments.** {ui.esc(inc['affected_segments'] or '—')}  \n"
                    f"**Customer contact.** {ui.esc(inc['customer_contact'] or '—')}")
    with right.container(border=True):
        st.markdown("#### Linked alerts")
        if inc["alerts"]:
            st.dataframe(pd.DataFrame([{"Severity": ui.SEVERITY[a["severity"]][2], "Alert": a["title"],
                                        "Started": ui.fmt_dt(a["started_at"]), "Status": a["status"]}
                                       for a in inc["alerts"]]), hide_index=True, width="stretch")
        else:
            st.caption("No alerts linked.")
        if diag:
            st.markdown(f"**Copilot diagnosis:** {ui.esc(diag['top']['label'])} "
                        f"({diag['top']['confidence']:.0%}) — onset {ui.fmt_dt(diag['onset'])} UTC")
    if diag:
        with st.container(border=True):
            impact_block(diag)
    with st.container(border=True):
        st.markdown("#### Timeline")
        icons = {"created": "＋", "alerts_linked": "⎘", "diagnosis": "⌕", "status": "→", "action": "✓", "note": "✎",
                 "summary": "✉", "communication": "✉", "resolution": "★", "severity": "!"}
        st.html("".join(
            f"<div class='tl-row'><div class='tl-time'>{ui.fmt_dt(e['ts'])}</div>"
            f"<div class='tl-actor'>{icons.get(e['kind'], '•')} {ui.esc(e['actor'])}</div>"
            f"<div>{ui.esc(e['message'])}</div></div>" for e in inc["events"]))
        with st.form(f"note_{iid}", clear_on_submit=True, border=False):
            n1, n2 = st.columns([6, 1], vertical_alignment="bottom")
            note = n1.text_input("Add a note", placeholder="e.g. Harborline confirmed the routing change went live at 01:46 UTC")
            if n2.form_submit_button("Add", width="stretch") and note.strip():
                api.guard(api.post, f"/api/incidents/{iid}/notes", {"actor": ACTOR, "message": note})
                refresh()

# ---- Diagnosis -----------------------------------------------------------------------
elif view == "Diagnosis":
    if not diag:
        with st.container(border=True):
            st.markdown("#### No diagnosis yet")
            st.markdown("The copilot checks **data quality, input drift, predictions & decisions, labelled performance, "
                        "serving health, change events and error logs** around the alert window, then ranks root-cause "
                        "hypotheses with the evidence for and against each one.")
            if inc["status"] != "resolved" and st.button("Run diagnosis", type="primary", icon=":material/troubleshoot:"):
                with st.spinner("Diagnosing …"):
                    api.guard(api.post, f"/api/incidents/{iid}/diagnose")
                refresh()
        st.stop()

    top = diag["hypotheses"][0]
    a, b = st.columns([3, 2])
    with a.container(border=True):
        st.markdown("<div class='kicker'>Most likely root cause</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='display:flex;align-items:baseline;gap:16px'><div class='hero'>{top['confidence']:.0%}</div>"
                    f"<div class='hyp-title'>{ui.esc(top['label'])}</div></div>", unsafe_allow_html=True)
        st.markdown(ui.esc(top["summary"]))
        st.markdown(f"<div class='sub'>Onset <b>{ui.fmt_dt(diag['onset'])} UTC</b> — {ui.esc(diag['onset_method'])}. "
                    f"Compared {diag['rows']['incident']:,} requests after onset with {diag['rows']['baseline']:,} in the "
                    f"7 days before · ran in {diag['runtime_ms']} ms.</div>", unsafe_allow_html=True)
    with b.container(border=True):
        hs = diag["hypotheses"]
        ui.plot(ui.bar_chart([h["label"] for h in hs], {"confidence": [h["confidence"] for h in hs]}, "pct",
                             "Hypotheses ranked", 250, horizontal=True, highlight=0,
                             text=[f"{h['confidence']:.0%}" for h in hs]), key="hyp_rank")

    icon = {"flagged": ("▲", ui.STATUS["serious"]), "clear": ("✓", ui.STATUS["good"]), "n/a": ("–", ui.MUTED)}
    st.html("<div class='checks'>" + "".join(
        f"<div class='check'><b><span style='color:{icon[c['status']][1]}'>{icon[c['status']][0]}</span> "
        f"{ui.esc(c['label'])}</b><span class='muted'>{ui.esc(c['headline'])}</span></div>"
        for c in diag["checks"]) + "</div>")
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Why this fits")
        seen = set()
        for s in top["supporting"]:
            for ev in s["evidence"]:
                if ev["id"] in seen:
                    continue
                seen.add(ev["id"])
                evidence_card(ev, s["contribution"] if ev is s["evidence"][0] else None, key="top")
    with right:
        st.markdown("#### What argues against it")
        against = [ev for s in top["contradicting"] for ev in s["evidence"]]
        if against or top["missing"]:
            for ev in against:
                evidence_card(ev, key="against")
            for m in top["missing"]:
                st.markdown(f"- {ui.esc(m)}")
        else:
            st.caption("No counter-evidence found.")
        st.markdown("#### How to confirm")
        st.info(top["verify"], icon=":material/fact_check:")
        st.markdown("#### Alternatives")
        for h in diag["hypotheses"][1:]:
            with st.expander(f"{h['label']} — {h['confidence']:.0%}"):
                st.markdown(ui.esc(h["summary"]))
                for s in h["supporting"]:
                    for ev in s["evidence"][:1]:
                        st.markdown(f"- ✓ {ui.esc(ev['title'])}")
                for s in h["contradicting"]:
                    for ev in s["evidence"][:1]:
                        st.markdown(f"- ✗ {ui.esc(ev['title'])}")
                for m in h["missing"]:
                    st.markdown(f"- ✗ {ui.esc(m)}")
        st.markdown("#### Similar past incidents")
        if not diag["similar"]:
            st.caption("No resolved incidents yet.")
        for s in diag["similar"]:
            with st.container(border=True):
                st.markdown(f"**{ui.esc(s['code'])}** · {s['similarity']:.0%} similar · "
                            f"<span class='sub'>{ui.esc(s['root_cause_label'])}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='ev-detail'>{ui.esc(s['title'])}<br><b>Fix:</b> {ui.esc(s['fix_applied'])}</div>",
                            unsafe_allow_html=True)

    with st.container(border=True):
        impact_block(diag)
    with st.expander("All evidence collected"):
        st.dataframe(pd.DataFrame([{"Check": e["check"], "Finding": e["title"], "Kind": e["kind"], "Detail": e["detail"]}
                                   for e in diag["evidence"]]), hide_index=True, width="stretch")
        st.download_button("Download diagnosis JSON", json.dumps(diag, indent=2), f"{inc['code']}-diagnosis.json",
                           "application/json", icon=":material/download:")

# ---- Action plan ---------------------------------------------------------------------
elif view == "Action plan":
    acts = inc["actions"]
    if not acts:
        st.info("Run the diagnosis to generate an evidence-linked action plan.")
        st.stop()
    done = sum(a["status"] in ("done", "skipped") for a in acts)
    p0 = [a for a in acts if a["priority"] == "P0"]
    p0_done = sum(a["status"] in ("done", "skipped") for a in p0)
    c1, c2 = st.columns(2)
    c1.progress(p0_done / max(len(p0), 1), text=f"P0 — do now: {p0_done}/{len(p0)} complete")
    c2.progress(done / len(acts), text=f"All actions: {done}/{len(acts)} complete")
    st.caption("When every P0 mitigation is done the incident moves to **Mitigated** automatically.")

    labels = {"todo": "To do", "in_progress": "Doing", "done": "Done", "skipped": "Skip"}

    def set_status(action_id: int, key: str) -> None:
        value = st.session_state.get(key)
        if value:
            api.guard(api.patch, f"/api/incidents/{iid}/actions/{action_id}", {"status": value, "actor": ACTOR})

    groups = {"P0": "P0 — do now", "P1": "P1 — fix", "P2": "P2 — prevent recurrence"}
    for pr, heading in groups.items():
        items = [a for a in acts if a["priority"] == pr]
        if not items:
            continue
        st.markdown(f"#### {heading}")
        for a in items:
            with st.container(border=True):
                l, r = st.columns([6, 3], vertical_alignment="center")
                owner = ui.pill("Client" if not a["ours"] else "Us", ui.SERIES[1] if not a["ours"] else ui.SERIES[0])
                strike = "text-decoration:line-through;color:#898781" if a["status"] in ("done", "skipped") else ""
                l.markdown(f"<div class='ev-title' style='{strike}'>{ui.esc(a['title'])}</div>"
                           f"<div class='sub'>{owner} {ui.esc(a['owner'])} · <span class='muted'>{ui.esc(a['kind'])}</span></div>",
                           unsafe_allow_html=True)
                key = f"act_{a['id']}_{a['status']}"
                r.segmented_control("Status", list(labels), format_func=labels.get, default=a["status"], key=key,
                                    label_visibility="collapsed", on_change=set_status, args=(a["id"], key))
                l.markdown(f"<div class='ev-detail'>{ui.esc(a['detail'])}</div>"
                           f"<div class='ev-detail' style='margin-top:4px'><b>Why:</b> {ui.esc(a['rationale'])}</div>",
                           unsafe_allow_html=True)

# ---- Updates -------------------------------------------------------------------------
elif view == "Updates":
    sums = inc["summaries"]
    if not sums:
        st.info("Run the diagnosis to draft the engineer and customer updates.")
        st.stop()
    aud = st.segmented_control("Audience", ["customer", "engineer"], default="customer", key="ws_aud",
                               format_func={"customer": "Customer update (plain language)",
                                            "engineer": "Engineering diagnosis (technical)"}.get) or "customer"
    s = sums[aud]
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Regenerate from latest state", icon=":material/refresh:", width="stretch",
                 help="Rebuild both drafts from the current diagnosis and action statuses."):
        api.guard(api.post, f"/api/incidents/{iid}/summaries/regenerate", {"use_llm": False, "actor": ACTOR})
        refresh()
    if b2.button("Polish wording with Claude", icon=":material/auto_awesome:", width="stretch",
                 disabled=not health["llm_enabled"],
                 help=("Rewrites the drafts for each audience without adding facts." if health["llm_enabled"]
                       else "Set ANTHROPIC_API_KEY on the API server to enable.")):
        with st.spinner("Polishing …"):
            api.guard(api.post, f"/api/incidents/{iid}/summaries/regenerate", {"use_llm": True, "actor": ACTOR})
        refresh()
    if b3.button("Mark as sent", icon=":material/send:", width="stretch", disabled=bool(s["sent_at"])):
        api.guard(api.post, f"/api/incidents/{iid}/summaries/{aud}/sent", actor=ACTOR)
        refresh()
    b4.download_button("Download .md", s["content"], f"{inc['code']}-{aud}.md", "text/markdown",
                       icon=":material/download:", width="stretch")
    sent = f" · sent {ui.fmt_dt(s['sent_at'])}" if s["sent_at"] else " · not sent yet"
    st.caption(f"Drafted {ui.fmt_dt(s['created_at'])} by {s['generator']}{sent}")
    with st.container(border=True):
        st.markdown(s["content"])
    with st.expander("Copy as text"):
        st.code(s["content"], language="markdown", wrap_lines=True)

# ---- Resolve & learn -----------------------------------------------------------------
elif view == "Resolve & learn":
    taxonomy = api.cached("/api/taxonomy")
    res = inc["resolution"]
    if res:
        with st.container(border=True):
            st.markdown("#### Resolved — added to the knowledge base")
            verdict = {"correct": ("✓", ui.STATUS["good"], "Diagnosis was correct"),
                       "partially_correct": ("●", ui.STATUS["warning"], "Diagnosis was partially correct"),
                       "incorrect": ("▲", ui.STATUS["critical"], "Diagnosis was wrong")}.get(res["diagnosis_feedback"])
            if verdict:
                st.markdown(ui.pill(verdict[2], verdict[1], verdict[0]), unsafe_allow_html=True)
            st.markdown(f"**Root cause:** {ui.esc(res['root_cause_label'])} — {ui.esc(res['root_cause_detail'])}")
            st.markdown(f"**Fix applied:** {ui.esc(res['fix_applied'])}")
            st.markdown(f"**Prevention:** {ui.esc(res['prevention'] or '—')}")
            ttm = f"{res['time_to_mitigate_min']} min" if res["time_to_mitigate_min"] else "—"
            st.markdown(f"**Time to mitigate:** {ttm} · **Tags:** {', '.join(res['tags']) or '—'} · "
                        f"**Resolved by:** {ui.esc(res['resolved_by'])}")
        st.markdown("The next incident with a similar signature will surface this one in *Similar past incidents*, "
                    "and its confirmed root cause adds to that category's prior in the ranking.")
        st.page_link("views/knowledge.py", label="Open the knowledge base", icon=":material/menu_book:")
        st.stop()

    cats = list(taxonomy)
    guess = diag["top"]["category"] if diag else "other"
    pre = f"res_{iid}_"
    if pre + "init" not in st.session_state:
        st.session_state[pre + "init"] = True
        st.session_state[pre + "cat"] = guess
        st.session_state[pre + "detail"] = diag["top"]["summary"] if diag else ""
        st.session_state[pre + "fix"] = "; ".join(a["title"] for a in inc["actions"]
                                                  if a["status"] == "done" and a["kind"] in ("mitigate", "fix"))
        st.session_state[pre + "prev"] = "; ".join(a["title"] for a in inc["actions"]
                                                   if a["kind"] == "prevent" and "feedback" not in a["title"])
    if diag:
        st.markdown(f"The copilot concluded **{ui.esc(diag['top']['label'])}** ({diag['top']['confidence']:.0%}). "
                    "Record what it actually was: the confirmed root cause is what future diagnoses learn from.")
    with st.form(f"resolve_{iid}", border=True):
        cat = st.selectbox("Confirmed root cause", cats, format_func=taxonomy.get, key=pre + "cat")
        st.text_area("What exactly happened", key=pre + "detail", height=80)
        st.text_area("Fix applied", key=pre + "fix", height=70)
        st.text_area("Prevention", key=pre + "prev", height=70)
        c1, c2 = st.columns(2)
        fb = c1.radio("Was the copilot's diagnosis right?", ["auto", "correct", "partially_correct", "incorrect"],
                      format_func={"auto": "Decide from the category", "correct": "Correct",
                                   "partially_correct": "Partially", "incorrect": "Wrong"}.get, horizontal=False)
        tags = c2.text_input("Tags (comma-separated)", placeholder="schema-change, vendor-api")
        who = c2.text_input("Resolved by", value=ACTOR)
        if st.form_submit_button("Resolve incident", type="primary", icon=":material/task_alt:"):
            body = {"root_cause_category": cat, "root_cause_detail": st.session_state[pre + "detail"],
                    "fix_applied": st.session_state[pre + "fix"], "prevention": st.session_state[pre + "prev"],
                    "diagnosis_feedback": None if fb == "auto" else fb,
                    "tags": [t.strip() for t in tags.split(",") if t.strip()], "resolved_by": who}
            if api.guard(api.post, f"/api/incidents/{iid}/resolve", body):
                st.toast("Resolved and added to the knowledge base", icon=":material/menu_book:")
                refresh()
