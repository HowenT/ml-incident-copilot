"""Model health dashboard: fleet status, KPIs, and the signals the diagnosis engine reads."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api
import ui

ov = api.guard(api.cached, "/api/overview")
if ov is None:
    st.stop()
models = {m["id"]: m for m in ov["models"]}

st.title("Model health")
st.markdown(f"<div class='sub'>{ui.esc(ov['models'][0]['client'])} · {len(models)} production models · "
            f"{ov['open_alerts']} open alerts · {ov['open_incidents']} active incidents</div>", unsafe_allow_html=True)
st.write("")

# ---- fleet cards ---------------------------------------------------------------------
cols = st.columns(len(models))
for col, m in zip(cols, models.values()):
    k = m["kpis"]
    with col.container(border=True):
        st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<div><div class='kicker'>{ui.esc(m['id'])} · v{ui.esc(m['version'])}</div>"
                    f"<div style='font-weight:650;font-size:1.05rem'>{ui.esc(m['display_name'])}</div></div>"
                    f"{ui.health_pill(m['health'])}</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests 24h", ui.compact(k["volume_24h"]))
        c2.metric(m["decision_name"], ui.pct(k["decision_rate_24h"]),
                  f"{(k['decision_rate_24h'] - k['decision_rate_7d']) * 100:+.1f} pp", delta_color="off")
        c3.metric("p95 latency", f"{k['latency_p95_last']:.0f} ms" if k["latency_p95_last"] else "—")
        c4.metric("Open alerts", k["open_alerts"])

# ---- filter row ----------------------------------------------------------------------
f1, f2, _ = st.columns([4, 2, 4], vertical_alignment="bottom")
mid = f1.segmented_control("Model", list(models), format_func=lambda i: models[i]["display_name"],
                           default=list(models)[0], key="mon_model") or list(models)[0]
days = f2.segmented_control("Window", [7, 14, 30], format_func=lambda d: f"{d}d", default=14, key="mon_days") or 14
m = models[mid]
k = m["kpis"]

met = api.cached(f"/api/models/{mid}/metrics", days=days)["rows"]
feats = api.cached(f"/api/models/{mid}/features", days=days)
perf = api.cached(f"/api/models/{mid}/performance")
alerts = api.cached("/api/alerts", model_id=mid)
changes = [c for c in api.cached("/api/logs", model_id=mid, event_type="change,business", limit=40)
           if c["ts"] >= met["ts"][0]]
x = met["ts"]
end = x[-1]

t = st.columns(6)
t[0].metric("Requests 24h", ui.compact(k["volume_24h"]),
            help="Extrapolated from a 10% prediction-log sample." if m["log_sample_rate"] < 1 else None, border=True)
t[1].metric(f"{m['decision_name']} 24h", ui.pct(k["decision_rate_24h"]),
            f"{(k['decision_rate_24h'] - k['decision_rate_7d']) * 100:+.1f} pp vs 7d", delta_color="off", border=True)
t[2].metric("p95 latency", f"{k['latency_p95_last']:.0f} ms", f"SLO {m['latency_slo_ms']:.0f} ms",
            delta_color="off", border=True)
t[3].metric("Fallbacks 24h", ui.pct(k["timeout_rate_24h"]),
            help="Requests answered by the gateway fallback policy instead of the model.", border=True)
t[4].metric("Max missing 24h", ui.pct(k["max_missing_24h"]), help="Highest hourly missing rate of any input.",
            border=True)
t[5].metric("Max drift (PSI)", f"{k['max_psi_now']:.2f}", help="24h rolling PSI vs the training reference.",
            border=True)


def bands(rules: set[str]) -> list[dict]:
    out = []
    for a in alerts:
        if a["rule"] in rules and (a["ended_at"] or end) >= x[0]:
            color = ui.SEVERITY[a["severity"]][0]
            out.append({"x0": max(a["started_at"], x[0]), "x1": a["ended_at"] or end, "color": color,
                        "label": f"{ui.SEVERITY[a['severity']][2]} alert"})
    return out


def change_markers() -> list[dict]:
    """One marker per burst of changes (within 6h), labelled by the first change."""
    short = {"deploy-bot": "deploy", "client-change-feed": "client change", "bureau-connector": "connector",
             "config-service": "config"}
    out: list[dict] = []
    for c in sorted(changes, key=lambda c: c["ts"]):
        ts = pd.Timestamp(c["ts"])
        if out and (ts - pd.Timestamp(out[-1]["x"])).total_seconds() < 6 * 3600:
            out[-1]["n"] += 1
            continue
        out.append({"x": c["ts"], "label": short.get(c["service"], c["service"]), "n": 1})
    return [{"x": o["x"], "label": o["label"] + (f" +{o['n'] - 1}" if o["n"] > 1 else "")} for o in out]


st.write("")
r1 = st.columns(2)
with r1[0].container(border=True):
    ui.plot(ui.line_chart(x, {f"{m['decision_name']} (24h rolling)": met["decision_rate_24h"]}, "pct",
                          f"{m['decision_name']}, 24h rolling", 270, bands=bands({"decision_rate_shift"}),
                          markers=change_markers()), key="mon_dec")
with r1[1].container(border=True):
    ui.plot(ui.line_chart(x, {"p95 latency": met["latency_p95"]}, "ms", "p95 latency (hourly)", 270,
                          bands=bands({"latency_slo"}), threshold=m["latency_slo_ms"],
                          threshold_label=f"SLO {m['latency_slo_ms']:.0f} ms", markers=change_markers()), key="mon_lat")

r2 = st.columns(2)
with r2[0].container(border=True):
    miss = feats["missing"]
    top = [f for f in sorted(miss, key=lambda f: -max(miss[f]))[:3] if max(miss[f]) >= 0.005]
    if not top:
        ui.plot(ui.line_chart(feats["ts"], {"max missing rate": met["max_missing_rate"]}, "pct",
                              "Input completeness — max missing rate across features", 240), key="mon_miss")
    else:
        ui.plot(ui.line_chart(feats["ts"], {f: miss[f] for f in top}, "pct", "Missing rate by input (hourly)", 240,
                              bands=bands({"missing_values"}), threshold=0.05, threshold_label="alert 5%"),
                key="mon_miss")
with r2[1].container(border=True):
    ui.plot(ui.line_chart(x, {"fallback rate": met["timeout_rate"]}, "pct",
                          "Requests answered by the fallback policy (hourly)", 240, bands=bands({"serving_errors"}),
                          threshold=0.01, threshold_label="alert 1%"), key="mon_fb")

with st.container(border=True):
    d = feats["daily_psi"]
    ui.plot(ui.heatmap(d["days"], d["features"], d["values"], "Input drift vs training reference — daily PSI by feature",
                       height=60 + 26 * len(d["features"])), key="mon_heat")
    st.caption("PSI below 0.10 is stable, 0.10–0.25 worth watching, above 0.25 a significant shift.")

with st.container(border=True):
    p = pd.DataFrame(perf)
    if len(p):
        lag = m["label_lag_hours"]
        pending = [{"x0": p["day"].iloc[-1], "x1": end[:10], "color": ui.BASELINE,
                    "label": f"labels pending (≈{lag // 24}d lag)"}]
        ui.plot(ui.line_chart(p["day"].tolist(), {"precision": p["precision"].tolist(), "AUC": p["auc"].tolist()},
                              "pct", "Quality on matured labels — daily, all history (outcomes arrive late)", 250,
                              bands=pending), key="mon_perf")
    else:
        st.info(f"No matured labels yet (label lag ≈{m['label_lag_hours'] // 24} days).")
with st.container(border=True):
    st.markdown("**Change feed** — deploys, config and client-side changes in this window")
    if changes:
        st.dataframe(pd.DataFrame([{"When (UTC)": ui.fmt_dt(c["ts"]), "Source": c["service"], "Change": c["message"]}
                                   for c in changes]), hide_index=True, width="stretch",
                     column_config={"Change": st.column_config.TextColumn(width="large")})
    else:
        st.caption("No deploys, config or client changes in this window.")

with st.expander("Model card"):
    st.markdown(f"**Use case.** {ui.esc(m['use_case'])}")
    st.markdown(f"**Owner** {ui.esc(m['owner'])} · **Decision threshold** {m['threshold']:.3f} · "
                f"**Label lag** {m['label_lag_hours'] // 24 if m['label_lag_hours'] >= 48 else m['label_lag_hours']}"
                f"{'d' if m['label_lag_hours'] >= 48 else 'h'} · **Prediction-log sample** {m['log_sample_rate']:.0%}")
    st.dataframe(pd.DataFrame(m["features"]).rename(columns=str.capitalize), hide_index=True, width="stretch")
