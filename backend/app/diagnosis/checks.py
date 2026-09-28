"""Diagnostic checks. Each one inspects one layer of the system and emits signals + evidence.

Layers: data quality -> input drift -> predictions/decisions -> labelled performance
        -> serving -> change events -> error logs.
"""
from __future__ import annotations

import math
import re
from datetime import timedelta

import numpy as np
import pandas as pd

from ..services.monitoring import flagged
from ..stats import auc, pct, pp, psi_between
from .context import Ctx, bar_chart, iso, line_chart

DATA_SERVICES = {"bureau-connector", "feature-pipeline", "etl", "client-change-feed", "batch-scoring"}
INFRA_SERVICES = {"feature-store", "api-gateway", "model-serving", "cloud-status"}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def localise(df: pd.DataFrame, mask: pd.Series, segments: list[str], min_rows: int = 30) -> dict | None:
    """Find the segment value that best explains where `mask` is true.

    score = coverage (share of flagged rows inside the segment) x relative lift.
    """
    total = int(mask.sum())
    if total == 0:
        return None
    best = None
    for col in segments:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col).groups.items():
            if len(idx) < min_rows or len(idx) == len(df):
                continue
            inside = mask.loc[idx]
            rate_in = float(inside.mean())
            out_mask = mask.drop(index=idx)
            rate_out = float(out_mask.mean()) if len(out_mask) else 0.0
            coverage = float(inside.sum() / total)
            if rate_in <= rate_out:
                continue
            score = coverage * (rate_in - rate_out) / max(rate_in, 1e-9)
            if best is None or score > best["score"]:
                best = {"column": col, "value": str(val), "rate_in": rate_in, "rate_out": rate_out,
                        "coverage": coverage, "score": score, "rows": int(len(idx))}
    return best


def _fmt_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}k"
    return f"${x:,.0f}"


def _decision_word(ctx: Ctx) -> str:
    return "approval" if ctx.model_type == "credit" else "block"


# --------------------------------------------------------------------------------------
# 1. Data quality
# --------------------------------------------------------------------------------------


def check_data_quality(ctx: Ctx) -> None:
    rows = []
    for f in ctx.feature_names:
        b, c = float(ctx.base[f].isna().mean()), float(ctx.cur[f].isna().mean())
        rows.append((f, b, c, c - b))
    rows.sort(key=lambda r: -r[3])
    degraded = [r for r in rows if r[3] > 0.03 and r[2] > 0.05]

    vol_b = len(ctx.base) / max((ctx.onset - ctx.baseline_start).total_seconds() / 3600, 1)
    vol_c = len(ctx.cur) / max((ctx.window_end - ctx.onset).total_seconds() / 3600, 1)
    if vol_b and not 0.7 < vol_c / vol_b < 1.3:
        ctx.add_evidence("data_quality", f"Request volume changed {vol_b:.0f}/h → {vol_c:.0f}/h",
                         "Volume shifts often accompany routing or integration changes.", kind="context")

    if not degraded:
        worst = rows[0]
        ctx.add_check("data_quality", "Data quality", "clear",
                      f"All {len(rows)} inputs complete (largest missing-rate change {pp(worst[3])}).")
        ev = ctx.add_evidence("data_quality", "Input completeness is normal",
                              f"No feature's missing rate rose by more than 3pp (largest: {worst[0]} {pp(worst[3])}).",
                              kind="clear")
        ctx.signal("inputs_complete", 1.0, [ev])
        return

    f0, b0, c0, d0 = degraded[0]
    names = [r[0] for r in degraded]
    stable = [r for r in rows if r[0] not in names]
    x, y0 = ctx.hourly(lambda d: d[f0].isna().mean())
    series = {f0: y0}
    for f, *_ in degraded[1:3]:
        series[f] = ctx.hourly(lambda d, f=f: d[f].isna().mean())[1]
    chart = line_chart(f"Hourly missing rate — {', '.join(series)}", x, series, "pct", ctx.onset, 0.05, "alert limit 5%")
    ev = ctx.add_evidence(
        "data_quality", f"{f0} missing rate {pct(b0)} → {pct(c0)} after onset",
        f"{len(degraded)} input(s) degraded: " + ", ".join(f"{f} {pct(b)} → {pct(c)}" for f, b, c, _ in degraded)
        + f". The other {len(stable)} inputs stayed complete (max change {pp(max((r[3] for r in stable), default=0))}).",
        chart,
    )
    ctx.signal("missing_spike", min(1.0, d0 / 0.15), [ev])
    ctx.facts.update(missing_features=names, missing_feature=f0, missing_before=b0, missing_after=c0)

    miss_mask = ctx.cur[f0].isna()
    seg = localise(ctx.cur, miss_mask, ctx.model["segments"])
    if seg and seg["coverage"] > 0.6:
        by = ctx.cur.groupby(seg["column"])[f0].apply(lambda s: s.isna().mean()).sort_values(ascending=False)
        chart = bar_chart(f"{f0} missing rate by {seg['column']} (incident window)", by.index.astype(str).tolist(),
                          {"missing rate": by.to_numpy()}, "pct")
        ev = ctx.add_evidence(
            "data_quality",
            f"{seg['coverage']:.0%} of null {f0} rows come from {seg['column']} = {seg['value']}",
            f"Null rate {pct(seg['rate_in'])} inside that segment vs {pct(seg['rate_out'])} everywhere else — "
            "the gap is tied to one data path, not random loss.",
            chart,
        )
        ctx.signal("missing_segment_concentration", seg["coverage"] * min(1.0, (seg["rate_in"] - seg["rate_out"]) / 0.5), [ev])
        ctx.facts["missing_segment"] = seg

    dr_missing = float(ctx.cur.loc[miss_mask, "decision"].mean())
    dr_complete = float(ctx.cur.loc[~miss_mask, "decision"].mean())
    if abs(dr_missing - dr_complete) > 0.04:
        word = _decision_word(ctx)
        ev = ctx.add_evidence(
            "data_quality",
            f"Rows with null {f0}: {word} rate {pct(dr_missing)} vs {pct(dr_complete)} for complete rows",
            "The pipeline imputes the training median for missing values, so these requests are scored as an "
            f"'average' profile — this is what moves the {word} rate.",
        )
        ctx.signal("missing_drives_decision", min(1.0, abs(dr_missing - dr_complete) / 0.12), [ev])
        ctx.facts.update(decision_missing=dr_missing, decision_complete=dr_complete)

    ctx.add_check("data_quality", "Data quality", "flagged",
                  f"{', '.join(names)} missing on {pct(c0)} of requests (was {pct(b0)}).")


# --------------------------------------------------------------------------------------
# 2. Input drift
# --------------------------------------------------------------------------------------


def check_drift(ctx: Ctx) -> None:
    skip = set(ctx.facts.get("missing_features", []))
    scores = {f: psi_between(ctx.base[f], ctx.cur[f]) for f in ctx.feature_names}
    ranked = sorted(((f, s) for f, s in scores.items() if f not in skip), key=lambda t: -t[1])
    drifting = [(f, s) for f, s in ranked if s > 0.1]
    cats = [f for f, _ in ranked]
    chart = bar_chart("Input drift vs pre-incident week (PSI)", cats, {"PSI": [scores[f] for f in cats]}, "num",
                      0.1, "watch 0.1")

    if not drifting:
        ctx.add_check("drift", "Input drift", "clear", f"All inputs stable (max PSI {ranked[0][1]:.2f}).")
        ev = ctx.add_evidence("drift", "Input distributions are stable",
                              f"Max PSI vs the pre-incident week is {ranked[0][1]:.2f} ({ranked[0][0]}); "
                              "0.1 is the usual watch level.", chart, kind="clear")
        ctx.signal("no_feature_drift", 1.0, [ev])
        return

    top_f, top_psi = drifting[0]
    ev = ctx.add_evidence(
        "drift", f"{len(drifting)} input(s) drifted — top: {top_f} (PSI {top_psi:.2f})",
        "PSI vs the 7 days before onset: " + ", ".join(f"{f} {s:.2f}" for f, s in drifting[:5]) + ".",
        chart,
    )
    ctx.signal("feature_drift", min(1.0, top_psi / 0.2) * (0.7 + 0.3 * min(1.0, len(drifting) / 3)), [ev])
    ctx.facts["drift_features"] = drifting[:5]

    # Which segment grew or appeared? (share of traffic, baseline vs incident)
    best = None
    for col in ctx.model["segments"]:
        if col == "model_version" or col not in ctx.cur.columns:
            continue
        sb = ctx.base[col].astype(str).value_counts(normalize=True)
        sc = ctx.cur[col].astype(str).value_counts(normalize=True)
        for val in sc.index:
            delta = float(sc[val] - sb.get(val, 0.0))
            if best is None or delta > best["delta"]:
                best = {"column": col, "value": val, "share_before": float(sb.get(val, 0.0)),
                        "share_after": float(sc[val]), "delta": delta}
    if best and best["delta"] > 0.04:
        rest = ctx.cur[ctx.cur[best["column"]].astype(str) != best["value"]]
        psi_rest = psi_between(ctx.base[top_f], rest[top_f])
        explained = psi_rest < 0.5 * top_psi
        detail = (f"Share of traffic {pct(best['share_before'])} → {pct(best['share_after'])}. "
                  + (f"Excluding this segment, {top_f} PSI falls from {top_psi:.2f} to {psi_rest:.2f} — "
                     "the drift is this segment." if explained else
                     f"Excluding it, {top_f} PSI is still {psi_rest:.2f}, so it explains only part of the drift."))
        sb = ctx.base[best["column"]].astype(str).value_counts(normalize=True)
        sc = ctx.cur[best["column"]].astype(str).value_counts(normalize=True)
        labels = sorted(set(sb.index) | set(sc.index), key=lambda v: -sc.get(v, 0))
        chart = bar_chart(f"Traffic mix by {best['column']}", labels,
                          {"baseline": [sb.get(v, 0) for v in labels], "incident": [sc.get(v, 0) for v in labels]}, "pct")
        ev = ctx.add_evidence(
            "drift",
            f"{'New' if best['share_before'] < 0.005 else 'Growing'} segment: {best['column']} = {best['value']}",
            detail, chart,
        )
        ctx.signal("drift_segment_concentration", min(1.0, best["delta"] / 0.12) * (1.0 if explained else 0.6), [ev])
        best["psi_without"] = psi_rest
        ctx.facts["drift_segment"] = best

    ctx.add_check("drift", "Input drift", "flagged",
                  f"{len(drifting)} input(s) drifted; top {top_f} PSI {top_psi:.2f}.")


# --------------------------------------------------------------------------------------
# 3. Predictions & decisions
# --------------------------------------------------------------------------------------


def check_predictions(ctx: Ctx) -> None:
    word = _decision_word(ctx)
    score_psi = psi_between(ctx.base["score"], ctx.cur["score"])
    dr_b, dr_c = float(ctx.base["decision"].mean()), float(ctx.cur["decision"].mean())
    delta = dr_c - dr_b
    z = delta / math.sqrt(max(dr_b * (1 - dr_b), 1e-6) / max(len(ctx.cur), 1))
    ctx.facts.update(decision_before=dr_b, decision_after=dr_c, score_psi=score_psi)

    x, y = ctx.hourly(lambda d: d["decision"].mean(), smooth=6)
    chart = line_chart(f"{ctx.decision_name} (6h rolling)", x, {ctx.decision_name: y}, "pct", ctx.onset,
                       dr_b, "baseline")
    shifted = (abs(delta) >= 0.02 and abs(z) >= 3) or score_psi > 0.1
    if not shifted:
        ctx.add_check("predictions", "Predictions & decisions", "clear",
                      f"{ctx.decision_name} {pct(dr_b)} → {pct(dr_c)}; score PSI {score_psi:.2f}.")
        ctx.add_evidence("predictions", f"{ctx.decision_name} steady ({pct(dr_b)} → {pct(dr_c)})",
                         f"Score distribution PSI {score_psi:.2f}.", chart, kind="clear")
        return

    ev = ctx.add_evidence(
        "predictions", f"{ctx.decision_name} {pct(dr_b)} → {pct(dr_c)} ({pp(delta)}, z = {z:.1f})",
        f"Score distribution PSI vs baseline: {score_psi:.2f}.", chart,
    )
    ctx.signal("prediction_shift", max(min(1.0, score_psi / 0.2), min(1.0, abs(delta) / 0.05)), [ev])

    # Where did decisions move most?
    best = None
    for col in ctx.model["segments"]:
        if col == "model_version":
            continue
        rb = ctx.base.groupby(col)["decision"].mean()
        rc = ctx.cur.groupby(col)["decision"].mean()
        nc = ctx.cur[col].value_counts()
        for val in rc.index:
            if nc.get(val, 0) < 30 or val not in rb.index:
                continue
            d = float(rc[val] - rb[val])
            weight = abs(d) * math.sqrt(nc[val])
            if best is None or weight > best["weight"]:
                best = {"column": col, "value": str(val), "before": float(rb[val]), "after": float(rc[val]),
                        "weight": weight}
    if best and abs(best["after"] - best["before"]) > 1.5 * abs(delta):
        ctx.add_evidence("predictions",
                         f"Largest shift: {best['column']} = {best['value']} {word} rate "
                         f"{pct(best['before'])} → {pct(best['after'])}",
                         "Localises the customer-visible symptom.", kind="context")
        ctx.facts["decision_segment"] = best

    # Decisions that disagree with the registered threshold point at a config change.
    scored = ctx.cur[ctx.cur["score"].notna()]
    expected = (scored["score"] < ctx.model["threshold"]) if ctx.model_type == "credit" else (scored["score"] >= ctx.model["threshold"])
    mismatch = float((expected.astype(int) != scored["decision"]).mean()) if len(scored) else 0.0
    if abs(delta) > 0.02 and score_psi < 0.05:
        ev = ctx.add_evidence("predictions", "Decision rate moved while scores did not",
                              f"Score PSI {score_psi:.2f}; {pct(mismatch)} of decisions disagree with the registered "
                              f"threshold {ctx.model['threshold']:.3f}.")
        ctx.signal("decision_shift_without_score_shift", 0.5 + 0.5 * min(1.0, mismatch / 0.02), [ev])
    ctx.add_check("predictions", "Predictions & decisions", "flagged",
                  f"{ctx.decision_name} {pct(dr_b)} → {pct(dr_c)} ({pp(delta)}).")


# --------------------------------------------------------------------------------------
# 4. Performance on matured labels
# --------------------------------------------------------------------------------------


def _perf(df: pd.DataFrame, model_type: str) -> dict | None:
    lab = df[df["label"].notna() & (df["status"] == "ok")]
    if len(lab) < 300 or lab["label"].sum() < 10:
        return None
    y = lab["label"].astype(int)
    fl = flagged(model_type, lab["decision"])
    tp = int(((fl == 1) & (y == 1)).sum())
    return {"n": len(lab), "auc": auc(y.to_numpy(), lab["score"].to_numpy()),
            "precision": tp / max(int(fl.sum()), 1), "recall": tp / max(int(y.sum()), 1)}


def check_performance(ctx: Ctx) -> None:
    lag = ctx.model["label_lag_hours"]
    pb, pc = _perf(ctx.base, ctx.model_type), _perf(ctx.cur, ctx.model_type)
    labeled = int(ctx.cur["label"].notna().sum())
    if pb is None or pc is None:
        ctx.add_check("performance", "Labelled performance", "n/a",
                      f"Labels not matured for the incident window (label lag ≈{lag // 24}d; {labeled} labelled rows).")
        ctx.add_evidence("performance", "Ground-truth labels not yet available",
                         f"Outcome labels arrive ≈{lag // 24} days after scoring, so accuracy cannot be measured yet. "
                         "Diagnosis relies on data, decision and log signals instead.", kind="context")
        ctx.facts["labels_available"] = False
        return
    ctx.facts["labels_available"] = True
    ctx.facts["performance"] = {"before": pb, "after": pc}
    d_prec = pb["precision"] - pc["precision"]
    d_auc = (pb["auc"] or 0) - (pc["auc"] or 0)
    x, y = ctx.hourly(lambda d: _hourly_precision(d, ctx.model_type), smooth=24)
    chart = line_chart("Precision on matured labels (24h rolling)", x, {"precision": y}, "pct", ctx.onset,
                       pb["precision"], "baseline")
    if d_prec < 0.05 and d_auc < 0.02:
        ctx.add_check("performance", "Labelled performance", "clear",
                      f"Precision {pct(pb['precision'])} → {pct(pc['precision'])}, AUC {pb['auc']:.3f} → {pc['auc']:.3f}.")
        ev = ctx.add_evidence("performance", "Model quality holds on matured labels",
                              f"Precision {pct(pb['precision'])} → {pct(pc['precision'])}; AUC {pb['auc']:.3f} → {pc['auc']:.3f}.",
                              chart, kind="clear")
        ctx.signal("performance_stable", 1.0, [ev])
        return
    ev = ctx.add_evidence(
        "performance", f"Precision {pct(pb['precision'])} → {pct(pc['precision'])} on matured labels",
        f"AUC {pb['auc']:.3f} → {pc['auc']:.3f}; recall {pct(pb['recall'])} → {pct(pc['recall'])} "
        f"({pc['n']:,} labelled rows in the incident window).", chart,
    )
    ctx.signal("performance_drop", max(min(1.0, d_prec / 0.12), min(1.0, d_auc / 0.04)), [ev])

    lab = ctx.cur[ctx.cur["label"].notna() & (ctx.cur["status"] == "ok")]
    fp = (flagged(ctx.model_type, lab["decision"]) == 1) & (lab["label"] == 0)
    seg = localise(lab, fp, ctx.model["segments"])
    if seg and seg["coverage"] > 0.4:
        action = "declines" if ctx.model_type == "credit" else "blocks"
        ev = ctx.add_evidence(
            "performance", f"{seg['coverage']:.0%} of false-positive {action} are {seg['column']} = {seg['value']}",
            f"False-positive rate {pct(seg['rate_in'])} inside the segment vs {pct(seg['rate_out'])} elsewhere.",
        )
        ctx.signal("performance_segment_concentration", seg["coverage"], [ev])
        ctx.facts["fp_segment"] = seg
    ctx.add_check("performance", "Labelled performance", "flagged",
                  f"Precision {pct(pb['precision'])} → {pct(pc['precision'])}.")


def _hourly_precision(d: pd.DataFrame, model_type: str) -> float:
    lab = d[d["label"].notna() & (d["status"] == "ok")]
    fl = flagged(model_type, lab["decision"])
    if fl.sum() == 0:
        return np.nan
    return float(((fl == 1) & (lab["label"] == 1)).sum() / fl.sum())


# --------------------------------------------------------------------------------------
# 5. Serving
# --------------------------------------------------------------------------------------


def check_serving(ctx: Ctx) -> None:
    slo = ctx.model["latency_slo_ms"]
    p95_b, p95_c = float(ctx.base["latency_ms"].quantile(0.95)), float(ctx.cur["latency_ms"].quantile(0.95))
    to_b = float((ctx.base["status"] != "ok").mean())
    to_c = float((ctx.cur["status"] != "ok").mean())
    ctx.facts.update(p95_before=p95_b, p95_after=p95_c, timeout_before=to_b, timeout_after=to_c,
                     timeouts=int((ctx.cur["status"] != "ok").sum()))
    ratio = p95_c / max(p95_b, 1e-6)
    flagged_any = False

    if ratio > 1.5 and p95_c > 0.7 * slo:
        x, y = ctx.hourly(lambda d: d["latency_ms"].quantile(0.95))
        chart = line_chart("p95 latency (hourly)", x, {"p95 latency": y}, "ms", ctx.onset, slo, f"SLO {slo:.0f}ms")
        ev = ctx.add_evidence("serving", f"p95 latency {p95_b:.0f}ms → {p95_c:.0f}ms ({ratio:.1f}×)",
                              f"SLO is {slo:.0f}ms.", chart)
        ctx.signal("latency_spike", min(1.0, (ratio - 1) / 3), [ev])
        flagged_any = True
    if to_c > 0.005 and to_c > 3 * to_b:
        scale = 1.0 / float(ctx.model.get("log_sample_rate") or 1.0)
        n = ctx.facts["timeouts"] * scale
        amount = ctx.cur.loc[ctx.cur["status"] != "ok"].get("amount")
        money = f" ({_fmt_money(float(amount.sum()) * scale)} in transaction value)" if amount is not None else ""
        est = " (est. from the prediction-log sample)" if scale != 1 else ""
        x, y = ctx.hourly(lambda d: (d["status"] != "ok").mean())
        chart = line_chart("Requests answered by fallback (hourly)", x, {"fallback rate": y}, "pct", ctx.onset,
                           0.01, "alert 1%")
        ev = ctx.add_evidence("serving", f"{pct(to_c)} of requests timed out and got the fallback decision",
                              f"{n:,.0f} requests were decided without a model score{money}{est}.", chart)
        ctx.signal("serving_errors", min(1.0, to_c / 0.04), [ev])
        flagged_any = True

    # Is the degradation tied to a specific model version that appeared near onset?
    both = pd.concat([ctx.base, ctx.cur])
    firsts = both.groupby("model_version")["ts"].min()
    new = [v for v, t in firsts.items() if ctx.onset - timedelta(hours=6) <= t.to_pydatetime() <= ctx.onset + timedelta(hours=2)]
    if new and flagged_any:
        stats = both.groupby("model_version").agg(p95=("latency_ms", lambda s: s.quantile(0.95)),
                                                  fallback=("status", lambda s: (s != "ok").mean()),
                                                  n=("ts", "size"))
        v_new = new[0]
        others = stats.drop(index=v_new)
        if len(others):
            v_old = others["n"].idxmax()
            worse = stats.loc[v_new, "p95"] > 1.8 * others.loc[v_old, "p95"] or \
                stats.loc[v_new, "fallback"] > 3 * others.loc[v_old, "fallback"] + 0.005
            if worse:
                chart = bar_chart("p95 latency by model version", stats.index.tolist(), {"p95 (ms)": stats["p95"].to_numpy()},
                                  "ms", slo, f"SLO {slo:.0f}ms")
                ev = ctx.add_evidence(
                    "serving", f"Degradation is confined to version {v_new}",
                    f"{v_new}: p95 {stats.loc[v_new, 'p95']:.0f}ms, fallback {pct(stats.loc[v_new, 'fallback'])}; "
                    f"{v_old}: p95 {stats.loc[v_old, 'p95']:.0f}ms, fallback {pct(stats.loc[v_old, 'fallback'])}. "
                    f"{v_new} first served {iso(firsts[v_new])}.", chart,
                )
                ctx.signal("version_correlated", 1.0, [ev])
                ctx.facts.update(new_version=v_new, old_version=str(v_old))
    if not flagged_any:
        ctx.add_check("serving", "Serving health", "clear",
                      f"p95 {p95_b:.0f}ms → {p95_c:.0f}ms; fallback rate {pct(to_c)}.")
        ev = ctx.add_evidence("serving", "Serving latency and error rate normal",
                              f"p95 {p95_b:.0f}ms → {p95_c:.0f}ms (SLO {slo:.0f}ms); fallback {pct(to_b)} → {pct(to_c)}.",
                              kind="clear")
        ctx.signal("serving_healthy", 1.0, [ev])
    else:
        ctx.add_check("serving", "Serving health", "flagged",
                      f"p95 {p95_b:.0f}ms → {p95_c:.0f}ms; fallback {pct(to_b)} → {pct(to_c)}.")


# --------------------------------------------------------------------------------------
# 6. Change events
# --------------------------------------------------------------------------------------


def _change_kind(row: dict) -> str:
    msg, svc = row["message"].lower(), row["service"]
    if row["event_type"] == "business":
        return "business"
    if svc == "deploy-bot" or re.search(r"\b(rolled out|promoted|redeploy|deployed|release)\b", msg):
        return "deploy"
    if svc == "config-service" or re.search(r"threshold|cut-off|cutoff|decision config", msg):
        return "config"
    return "data_source"


def _link_terms(ctx: Ctx) -> list[str]:
    terms: list[str] = list(ctx.facts.get("missing_features", []))
    for key in ("missing_segment", "drift_segment", "fp_segment"):
        seg = ctx.facts.get(key)
        if seg:
            terms.append(seg["value"])
            terms.extend(t for t in re.split(r"[_\-\s]", seg["value"]) if len(t) > 3)
    if ctx.facts.get("new_version"):
        terms.append(ctx.facts["new_version"])
    return [t.lower() for t in terms]


def check_changes(ctx: Ctx) -> None:
    signal_names = {"deploy": "deploy_near_onset", "config": "config_change_near_onset",
                    "data_source": "data_source_change_near_onset", "business": "business_event_near_onset"}
    tau = {"deploy": 6.0, "config": 6.0, "data_source": 6.0, "business": 24.0}
    terms = _link_terms(ctx)
    near, far = [], []
    for row in ctx.logs:
        if row["event_type"] not in ("change", "business"):
            continue
        dt_h = (ctx.onset - row["ts"]).total_seconds() / 3600
        if dt_h < -2 or dt_h > 24 * 14:
            continue
        kind = _change_kind(row)
        weight = math.exp(-max(dt_h, 0) / tau[kind]) if dt_h <= 48 else 0.0
        linked = [t for t in terms if t in row["message"].lower()]
        if linked and weight > 0.05:
            weight = min(1.0, weight + 0.3)
        (near if weight >= 0.15 else far).append((row, kind, dt_h, weight, linked))

    for row, kind, dt_h, weight, linked in sorted(near, key=lambda t: -t[3]):
        when = f"{abs(dt_h) * 60:.0f} min" if abs(dt_h) < 2 else f"{abs(dt_h):.1f} h"
        rel = "before" if dt_h >= 0 else "after"
        detail = f"[{row['service']}] {row['message']}"
        if linked:
            detail += f" — mentions {', '.join(sorted(set(linked)))}, which matches the anomaly."
        ev = ctx.add_evidence("changes", f"{kind.replace('_', ' ').capitalize()} change {when} {rel} onset", detail)
        ctx.signal(signal_names[kind], weight, [ev])
        ctx.facts.setdefault("changes", []).append({"kind": kind, "service": row["service"], "message": row["message"],
                                                    "ts": iso(row["ts"]), "hours_before_onset": round(dt_h, 2)})
    for row, kind, dt_h, weight, _ in sorted(far, key=lambda t: t[2])[:2]:
        ctx.add_evidence("changes", f"Ruled out: {kind.replace('_', ' ')} change {dt_h / 24:.1f} days before onset",
                         f"[{row['service']}] {row['message']} — too far from onset to explain a step change.",
                         kind="clear")
    if not near:
        ev = ctx.add_evidence("changes", "No deploy, config, data-source or business change within 48h of onset",
                              "Nothing in the change feed lines up with the anomaly.", kind="clear")
        ctx.signal("no_change_events", 1.0, [ev])
        ctx.add_check("changes", "Change events", "clear", "No changes within 48h before onset.")
    else:
        kinds = sorted({k for _, k, *_ in near})
        ctx.add_check("changes", "Change events", "flagged",
                      f"{len(near)} change(s) near onset: {', '.join(k.replace('_', ' ') for k in kinds)}.")


# --------------------------------------------------------------------------------------
# 7. Error logs
# --------------------------------------------------------------------------------------


def _template(msg: str) -> str:
    return re.sub(r"\d+(?:[.,]\d+)*", "#", msg)


def check_logs(ctx: Ctx) -> None:
    errs = [r for r in ctx.logs if r["level"] in ("WARN", "ERROR")]
    cur = [r for r in errs if ctx.onset - timedelta(hours=1) <= r["ts"] < ctx.window_end]
    base = [r for r in errs if ctx.baseline_start <= r["ts"] < ctx.onset - timedelta(hours=1)]
    cur_h = max((ctx.window_end - ctx.onset).total_seconds() / 3600, 1)
    base_h = max((ctx.onset - ctx.baseline_start).total_seconds() / 3600, 1)
    base_rate: dict[tuple, float] = {}
    for r in base:
        k = (r["service"], _template(r["message"]))
        base_rate[k] = base_rate.get(k, 0) + 1 / base_h
    groups: dict[tuple, list[dict]] = {}
    for r in cur:
        groups.setdefault((r["service"], _template(r["message"])), []).append(r)
    bursts = []
    for (svc, tpl), rows in groups.items():
        rate = len(rows) / cur_h
        if len(rows) >= 2 and rate > 3 * base_rate.get((svc, tpl), 0) + 0.01:
            bursts.append((svc, tpl, rows))
    bursts.sort(key=lambda b: -len(b[2]))
    features = [f.lower() for f in ctx.facts.get("missing_features", [])]
    data_n = infra_n = 0
    mentions = False
    for svc, tpl, rows in bursts[:4]:
        first = min(r["ts"] for r in rows)
        offset = (first - ctx.onset).total_seconds() / 60
        when = f"{offset:+.0f} min vs onset" if abs(offset) < 180 else f"{offset / 60:+.1f} h vs onset"
        mention = any(f in rows[0]["message"].lower() for f in features)
        mentions |= mention
        ev = ctx.add_evidence("logs", f"{svc} {rows[0]['level']} ×{len(rows)} (first {when})", rows[0]["message"])
        if svc in DATA_SERVICES:
            data_n += len(rows)
            ctx.signal("data_errors", min(1.0, len(rows) / 10) * (1.0 if mention else 0.7), [ev])
        elif svc in INFRA_SERVICES:
            infra_n += len(rows)
            ctx.signal("infra_errors", min(1.0, len(rows) / 10), [ev])
    ctx.facts["log_bursts"] = [{"service": s, "count": len(r), "example": r[0]["message"]} for s, _, r in bursts[:4]]
    if bursts:
        ctx.add_check("logs", "Error logs", "flagged",
                      f"{len(bursts)} new error pattern(s): " + ", ".join(f"{s} ×{len(r)}" for s, _, r in bursts[:3]) + ".")
    else:
        ctx.add_evidence("logs", "No new warning/error patterns after onset",
                         "Warning volume matches the baseline week.", kind="clear")
        ctx.add_check("logs", "Error logs", "clear", "No new error patterns after onset.")


ALL_CHECKS = [check_data_quality, check_drift, check_predictions, check_performance, check_serving,
              check_changes, check_logs]
