"""Monitoring job: prediction logs -> hourly metrics, feature health, performance, alerts.

Everything is computed from the raw prediction log, the way a scheduled monitoring job
would, so the dashboard and the diagnosis engine always agree with the underlying data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..stats import EPS, auc, bin_categorical, bin_numeric, numeric_edges, props

WINDOW_H = 24

# Alert thresholds — deliberately simple and explainable.
MISSING_RATE_LIMIT = 0.05
PSI_LIMIT = 0.15  # 0.10 = watch, 0.25 = significant, by common convention
PSI_HIGH = 0.25
SCORE_PSI_LIMIT = 0.15
DECISION_MIN_DELTA = 0.03
DECISION_MIN_Z = 3.0
TIMEOUT_LIMIT = 0.01
PRECISION_MIN_DROP = 0.10


# --------------------------------------------------------------------------------------
# Reference profile (computed once on the training/reference sample)
# --------------------------------------------------------------------------------------


def build_reference_profile(features: list[dict], ref: pd.DataFrame, ref_scores: np.ndarray) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for f in features:
        name = f["name"]
        if f["kind"] == "num":
            edges = numeric_edges(ref[name].to_numpy(float))
            p = props(bin_numeric(ref[name].to_numpy(float), edges), len(edges) - 1)
            profile[name] = {"kind": "num", "cuts": edges[1:-1], "props": p.tolist(),
                             "median": float(np.nanmedian(ref[name]))}
        else:
            cats = sorted(ref[name].dropna().astype(str).unique().tolist())
            p = props(bin_categorical(ref[name].astype(str), cats), len(cats) + 1)
            profile[name] = {"kind": "cat", "categories": cats, "props": p.tolist(),
                             "mode": str(ref[name].mode().iloc[0])}
    edges = numeric_edges(ref_scores)
    profile["__score__"] = {"kind": "num", "cuts": edges[1:-1],
                            "props": props(bin_numeric(ref_scores, edges), len(edges) - 1).tolist()}
    return profile


def _bin_with_profile(values: pd.Series, prof: dict) -> tuple[np.ndarray, int]:
    if prof["kind"] == "num":
        edges = [-np.inf, *prof["cuts"], np.inf]
        return bin_numeric(values.to_numpy(float), edges), len(edges) - 1
    cats = prof["categories"]
    return bin_categorical(values, cats), len(cats) + 1


def _rolling_psi(hour_idx: np.ndarray, bins: np.ndarray, n_bins: int, n_hours: int,
                 ref_props: list[float], window: int = WINDOW_H, min_count: int = 50) -> np.ndarray:
    valid = bins >= 0
    counts = np.zeros((n_hours, n_bins))
    np.add.at(counts, (hour_idx[valid], bins[valid]), 1)
    cs = np.cumsum(counts, axis=0)
    win = cs.copy()
    win[window:] -= cs[:-window]
    totals = win.sum(axis=1, keepdims=True)
    p = np.clip(win / np.where(totals == 0, 1, totals), EPS, None)
    e = np.clip(np.asarray(ref_props + [0.0] * (n_bins - len(ref_props))), EPS, None)
    e = e / e.sum()
    p = p / p.sum(axis=1, keepdims=True)
    out = ((p - e) * np.log(p / e)).sum(axis=1)
    out[totals[:, 0] < min_count] = np.nan
    out[: window - 1] = np.nan  # incomplete first window
    return out


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def compute_monitoring(model: dict, preds: pd.DataFrame, start: datetime, end: datetime):
    """Return (metrics_hourly, feature_stats_hourly, performance_daily) DataFrames."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    n_hours = len(hours)
    hour_idx = ((preds["ts"] - pd.Timestamp(start)) // pd.Timedelta(hours=1)).to_numpy().astype(int)
    keep = (hour_idx >= 0) & (hour_idx < n_hours)
    preds = preds.loc[keep]
    hour_idx = hour_idx[keep]
    hour_key = pd.Series(hour_idx, index=preds.index)

    g = preds.groupby(hour_key)
    volume = g.size().reindex(range(n_hours), fill_value=0).to_numpy()
    decisions = g["decision"].sum().reindex(range(n_hours), fill_value=0).to_numpy()
    lat = g["latency_ms"].quantile([0.5, 0.95, 0.99]).unstack().reindex(range(n_hours))
    timeouts = (preds["status"] != "ok").groupby(hour_key).sum().reindex(range(n_hours), fill_value=0).to_numpy()

    cs_v, cs_d = np.cumsum(volume), np.cumsum(decisions)
    win_v, win_d = cs_v.copy(), cs_d.copy()
    win_v[WINDOW_H:] -= cs_v[:-WINDOW_H]
    win_d[WINDOW_H:] -= cs_d[:-WINDOW_H]

    profile = model["reference_profile"]
    feat_rows = []
    missing_mat, psi_mat = [], []
    for f in model["features"]:
        name = f["name"]
        miss = preds[name].isna().groupby(hour_key).mean().reindex(range(n_hours)).to_numpy()
        bins, n_bins = _bin_with_profile(preds[name], profile[name])
        psi = _rolling_psi(hour_idx, bins, n_bins, n_hours, profile[name]["props"])
        missing_mat.append(np.nan_to_num(miss))
        psi_mat.append(np.nan_to_num(psi))
        for i in range(n_hours):
            feat_rows.append({"ts": hours[i].to_pydatetime(), "feature": name,
                              "missing_rate": float(np.nan_to_num(miss[i])), "psi_24h": float(np.nan_to_num(psi[i]))})
    sbins, n_sbins = _bin_with_profile(preds["score"], profile["__score__"])
    score_psi = _rolling_psi(hour_idx, sbins, n_sbins, n_hours, profile["__score__"]["props"])

    metrics = pd.DataFrame({
        "ts": hours,
        "volume": volume,
        "mean_score": g["score"].mean().reindex(range(n_hours)).to_numpy(),
        "decision_rate": np.where(volume > 0, decisions / np.maximum(volume, 1), np.nan),
        "decision_rate_24h": win_d / np.maximum(win_v, 1),
        "score_psi_24h": score_psi,
        "latency_p50": lat[0.5].to_numpy(),
        "latency_p95": lat[0.95].to_numpy(),
        "latency_p99": lat[0.99].to_numpy(),
        "timeout_rate": timeouts / np.maximum(volume, 1),
        "max_missing_rate": np.max(missing_mat, axis=0),
        "max_psi_24h": np.max(psi_mat, axis=0),
    })
    feats = pd.DataFrame(feat_rows)
    perf = compute_performance(model, preds)
    return metrics, feats, perf


def flagged(model_type: str, decision: pd.Series) -> pd.Series:
    """1 when the model called the positive (risky) class: decline for credit, block for fraud."""
    return (1 - decision) if model_type == "credit" else decision


def compute_performance(model: dict, preds: pd.DataFrame) -> pd.DataFrame:
    lab = preds[preds["label"].notna() & (preds["status"] == "ok")]
    rows = []
    for day, d in lab.groupby(lab["ts"].dt.floor("D")):
        y = d["label"].astype(int)
        fl = flagged(model["model_type"], d["decision"])
        tp = int(((fl == 1) & (y == 1)).sum())
        rows.append({
            "day": day.to_pydatetime(), "labeled": len(d), "positives": int(y.sum()),
            "auc": auc(y.to_numpy(), d["score"].to_numpy()),
            "precision": tp / int(fl.sum()) if fl.sum() else None,
            "recall": tp / int(y.sum()) if y.sum() else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Alert rules
# --------------------------------------------------------------------------------------


def _runs(mask: np.ndarray, min_len: int, max_gap: int = 2) -> list[tuple[int, int]]:
    """Breach episodes as inclusive (start, end) index pairs, merging short gaps."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))
    return [(a, b) for a, b in runs if mask[a:b + 1].sum() >= min_len]


def detect_alerts(model: dict, metrics: pd.DataFrame, feats: pd.DataFrame, perf: pd.DataFrame,
                  end: datetime) -> list[dict]:
    alerts: list[dict] = []
    hours = metrics["ts"].reset_index(drop=True)
    n = len(hours)

    def episode(rule: str, metric: str, a: int, b: int, title: str, severity: str, peak: float,
                threshold: float, baseline: float | None, features: list[str] | None = None,
                open_tail: int = 3, step: timedelta = timedelta(hours=1), ts_index=hours) -> None:
        is_open = b >= len(ts_index) - open_tail
        alerts.append({
            "model_id": model["id"], "rule": rule, "metric": metric, "title": title,
            "features": features or [], "severity": severity,
            "status": "open" if is_open else "resolved",
            "started_at": ts_index[a].to_pydatetime(),
            "ended_at": None if is_open else (ts_index[b] + step).to_pydatetime(),
            "peak_value": float(peak), "threshold": float(threshold),
            "baseline_value": None if baseline is None or np.isnan(baseline) else float(baseline),
        })

    def pre_baseline(series: np.ndarray, a: int) -> float:
        lo = max(0, a - 24 * 7)
        return float(np.nanmedian(series[lo:a])) if a > lo else float("nan")

    # 1) Missing values on any input feature.
    miss = feats.pivot(index="ts", columns="feature", values="missing_rate").reindex(hours).to_numpy()
    names = feats.pivot(index="ts", columns="feature", values="missing_rate").columns.tolist()
    breach = miss > MISSING_RATE_LIMIT
    order = [f["name"] for f in model["features"]]
    for a, b in _runs(breach.any(axis=1), min_len=2):
        cols = sorted((names[j] for j in np.flatnonzero(breach[a:b + 1].any(axis=0))), key=order.index)
        peak = float(np.nanmax(miss[a:b + 1][:, [names.index(c) for c in cols]]))
        episode("missing_values", "missing_rate", a, b,
                f"Missing values in {', '.join(cols)} (peak {peak:.1%})",
                "critical" if peak > 0.2 else "high", peak, MISSING_RATE_LIMIT,
                pre_baseline(miss[:, names.index(cols[0])], a), cols)

    # 2) Input drift vs the training reference (24h rolling PSI).
    psi = feats.pivot(index="ts", columns="feature", values="psi_24h").reindex(hours).to_numpy()
    breach = psi > PSI_LIMIT
    for a, b in _runs(breach.any(axis=1), min_len=3, max_gap=12):
        seg = psi[a:b + 1]
        order = np.argsort(-np.nanmax(seg, axis=0))
        cols = [names[j] for j in order if np.nanmax(seg[:, j]) > PSI_LIMIT]
        peak = float(np.nanmax(seg))
        episode("feature_drift", "psi_24h", a, b,
                f"Input drift on {len(cols)} feature{'s' if len(cols) > 1 else ''}: {', '.join(cols[:3])}"
                f"{'…' if len(cols) > 3 else ''} (PSI peak {peak:.2f})",
                "high" if peak > PSI_HIGH else "medium", peak, PSI_LIMIT, 0.0, cols)

    # 3) Score distribution shift.
    sp = metrics["score_psi_24h"].to_numpy()
    for a, b in _runs(np.nan_to_num(sp) > SCORE_PSI_LIMIT, min_len=3, max_gap=12):
        peak = float(np.nanmax(sp[a:b + 1]))
        episode("prediction_drift", "score_psi_24h", a, b, f"Model score distribution shift (PSI peak {peak:.2f})",
                "medium", peak, SCORE_PSI_LIMIT, 0.0)

    # 4) Decision-rate shift: rolling 24h vs the rate the model was calibrated for on the reference
    #    sample (same idea as drift vs the training reference: an ongoing shift never becomes "normal").
    dr = metrics["decision_rate_24h"].to_numpy()
    vol = metrics["volume"].to_numpy()
    base = np.full(n, float(model["reference_profile"]["__decision_rate__"]))
    base[:WINDOW_H - 1] = np.nan  # incomplete first window
    win_vol = pd.Series(vol).rolling(WINDOW_H, min_periods=1).sum().to_numpy()
    sigma = np.sqrt(np.clip(base * (1 - base), 1e-6, None) / np.maximum(win_vol, 1))
    delta = dr - base
    breach = (np.abs(delta) >= DECISION_MIN_DELTA) & (np.abs(delta) / sigma >= DECISION_MIN_Z)
    for a, b in _runs(np.nan_to_num(breach).astype(bool), min_len=3):
        k = a + int(np.nanargmax(np.abs(delta[a:b + 1])))
        episode("decision_rate_shift", "decision_rate_24h", a, b,
                f"{model['decision_name']} shift: {dr[k]:.1%} vs {base[a]:.1%} expected (24h rolling)",
                "high", dr[k], DECISION_MIN_DELTA, base[a])

    # 5) Latency SLO.
    p95 = metrics["latency_p95"].to_numpy()
    slo = model["latency_slo_ms"]
    for a, b in _runs(np.nan_to_num(p95) > slo, min_len=2):
        peak = float(np.nanmax(p95[a:b + 1]))
        episode("latency_slo", "latency_p95", a, b, f"p95 latency above SLO: {peak:.0f}ms (SLO {slo:.0f}ms)",
                "critical" if peak > 1.25 * slo else "high", peak, slo, pre_baseline(p95, a))

    # 6) Serving errors / fallback decisions.
    tr = metrics["timeout_rate"].to_numpy()
    for a, b in _runs(tr > TIMEOUT_LIMIT, min_len=2):
        peak = float(np.nanmax(tr[a:b + 1]))
        episode("serving_errors", "timeout_rate", a, b,
                f"Gateway timeouts → fallback decisions on {peak:.1%} of requests",
                "critical" if peak > 0.03 else "high", peak, TIMEOUT_LIMIT, pre_baseline(tr, a))

    # 7) Precision drop on matured labels (daily).
    if len(perf) >= 10:
        days = pd.Series(pd.to_datetime(perf["day"]))
        prec = perf["precision"].astype(float).to_numpy()
        base_p = np.full(len(prec), np.nan)
        for i in range(7, len(prec)):
            base_p[i] = np.nanmean(prec[max(0, i - 14):i])
        drop = base_p - prec
        for a, b in _runs(np.nan_to_num(drop) > PRECISION_MIN_DROP, min_len=2, max_gap=1):
            k = a + int(np.nanargmax(drop[a:b + 1]))
            episode("performance_drop", "precision", a, b,
                    f"Precision drop on matured labels: {base_p[a]:.0%} → {prec[k]:.0%}",
                    "high", prec[k], PRECISION_MIN_DROP, base_p[a], open_tail=2,
                    step=timedelta(days=1), ts_index=days)
    return alerts
