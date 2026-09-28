"""Knowledge base: every resolved incident, searchable, plus how the diagnosis is performing."""
from __future__ import annotations

import streamlit as st

import api
import ui

st.title("Knowledge base")
st.markdown("<div class='sub'>Resolved incidents with their confirmed root cause and fix. Similar-incident search and "
            "the diagnosis ranking both learn from this list.</div>", unsafe_allow_html=True)
st.write("")

stats = api.get("/api/knowledge/stats")
taxonomy = api.cached("/api/taxonomy")
c = st.columns(4)
c[0].metric("Resolved incidents", stats["resolved"], border=True)
acc = stats["top1_accuracy"]
c[1].metric("Diagnosis top-1 accuracy", ui.pct(acc, 0) if acc is not None else "—",
            help=f"Share of resolved incidents where the copilot's top hypothesis matched the confirmed root cause "
                 f"({stats['partial']} partial, {stats['incorrect']} wrong).", border=True)
c[2].metric("Median time to mitigate", f"{stats['median_ttm_min']:.0f} min" if stats["median_ttm_min"] else "—",
            border=True)
top_cat = (f"{stats['by_category'][0]['label'].split(' / ')[0]} ({stats['by_category'][0]['count']})"
           if stats["by_category"] else "—")
c[3].metric("Most common root cause", top_cat, border=True)

left, right = st.columns([3, 2])
with right.container(border=True):
    cats = stats["by_category"]
    ui.plot(ui.bar_chart([d["label"] for d in cats], {"incidents": [d["count"] for d in cats]}, "int",
                         "Resolved incidents by root cause", 60 + 38 * len(cats), horizontal=True,
                         text=[str(d["count"]) for d in cats]), key="kb_cats")
    st.markdown("**Diagnosis feedback, oldest → newest**")
    mark = {"correct": ("✓", ui.STATUS["good"]), "partially_correct": ("●", ui.STATUS["warning"]),
            "incorrect": ("▲", ui.STATUS["critical"])}
    st.html("<div class='checks'>" + "".join(
        f"<div class='check'><span style='color:{mark.get(f['feedback'], ('•', ui.MUTED))[1]}'>"
        f"{mark.get(f['feedback'], ('•', ui.MUTED))[0]}</span> {ui.esc(f['code'])}</div>"
        for f in stats["feedback_timeline"]) + "</div>")

with left:
    f1, f2 = st.columns([3, 2])
    q = f1.text_input("Search", placeholder="e.g. vendor schema change, fallback timeouts, promotion")
    cat = f2.selectbox("Root cause", ["all", *taxonomy], format_func=lambda k: "All" if k == "all" else taxonomy[k])
    items = api.get("/api/knowledge", q=q or None, category=None if cat == "all" else cat)
    if not items:
        st.caption("No matches.")
    for it in items:
        r = it["resolution"]
        with st.container(border=True):
            match = f" · match {it['match']:.0%}" if "match" in it else ""
            st.markdown(f"<div class='kicker'>{ui.esc(it['code'])} · {ui.esc(it['model_id'])} · "
                        f"{ui.fmt_dt(it['created_at'])}{match}</div>"
                        f"<div class='ev-title'>{ui.esc(it['title'])}</div>", unsafe_allow_html=True)
            fb = {"correct": "diagnosis correct", "partially_correct": "diagnosis partially correct",
                  "incorrect": "diagnosis wrong"}.get(r["diagnosis_feedback"], "")
            st.markdown(f"{ui.pill(r['root_cause_label'], ui.SERIES[0])} "
                        f"<span class='sub'>{fb}{' · mitigated in ' + str(r['time_to_mitigate_min']) + ' min' if r['time_to_mitigate_min'] else ''}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='ev-detail'><b>Cause:</b> {ui.esc(r['root_cause_detail'])}<br>"
                        f"<b>Fix:</b> {ui.esc(r['fix_applied'])}<br><b>Prevention:</b> {ui.esc(r['prevention'] or '—')}</div>",
                        unsafe_allow_html=True)
