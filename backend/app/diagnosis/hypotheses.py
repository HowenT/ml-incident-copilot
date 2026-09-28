"""Root-cause hypotheses as transparent weighted evidence rules.

score(h) = Σ w·signal (supports) − Σ w·signal (contradicts) + PRIOR_WEIGHT · historical prior
confidence = softmax(score / T) across hypotheses.

The weights are hand-set and readable on purpose: an engineer can see exactly why a
hypothesis ranked where it did, and the feedback loop (historical prior) nudges ranking
toward what past incidents with the same signature actually turned out to be.
"""
from __future__ import annotations

import math
from typing import Any

from ..stats import pct
from ..taxonomy import label
from .context import Ctx

PRIOR_WEIGHT = 0.15
TEMPERATURE = 0.25

HYPOTHESES: list[dict[str, Any]] = [
    {
        "category": "upstream_data_change",
        "supports": {"missing_spike": 0.30, "missing_segment_concentration": 0.15, "data_source_change_near_onset": 0.20,
                     "data_errors": 0.15, "missing_drives_decision": 0.10, "prediction_shift": 0.05},
        "contradicts": {"version_correlated": 0.15, "inputs_complete": 0.30},
        "key": "missing_spike",
    },
    {
        "category": "population_drift",
        "supports": {"feature_drift": 0.25, "drift_segment_concentration": 0.20, "business_event_near_onset": 0.15,
                     "prediction_shift": 0.10, "performance_drop": 0.10, "performance_segment_concentration": 0.10},
        "contradicts": {"missing_spike": 0.20, "version_correlated": 0.15, "no_feature_drift": 0.25},
        "key": "feature_drift",
    },
    {
        "category": "model_release_regression",
        "supports": {"deploy_near_onset": 0.30, "version_correlated": 0.30, "latency_spike": 0.10, "serving_errors": 0.10,
                     "infra_errors": 0.05, "prediction_shift": 0.05},
        "contradicts": {"missing_spike": 0.15, "drift_segment_concentration": 0.10},
        "key": "deploy_near_onset",
    },
    {
        "category": "serving_infrastructure",
        "supports": {"latency_spike": 0.30, "serving_errors": 0.25, "infra_errors": 0.25},
        "contradicts": {"version_correlated": 0.30, "missing_spike": 0.10, "serving_healthy": 0.30},
        "key": "latency_spike",
    },
    {
        "category": "config_threshold_change",
        "supports": {"config_change_near_onset": 0.45, "decision_shift_without_score_shift": 0.45},
        "contradicts": {"missing_spike": 0.15, "feature_drift": 0.10},
        "key": "decision_shift_without_score_shift",
    },
    {
        "category": "concept_drift",
        "supports": {"performance_drop": 0.40, "no_feature_drift": 0.20, "no_change_events": 0.20},
        "contradicts": {"missing_spike": 0.20, "drift_segment_concentration": 0.15, "latency_spike": 0.10,
                        "deploy_near_onset": 0.10, "performance_stable": 0.30},
        "key": "performance_drop",
    },
]

ABSENT: dict[str, str] = {
    "missing_spike": "No input lost data — completeness is normal.",
    "feature_drift": "Input distributions are stable vs the pre-incident week.",
    "deploy_near_onset": "No model deployment within 48h before onset.",
    "latency_spike": "Serving latency is normal.",
    "decision_shift_without_score_shift": "Decisions track the registered threshold; scores moved with them.",
    "performance_drop": "No measurable quality drop on matured labels (or labels not yet available).",
}

VERIFY: dict[str, str] = {
    "upstream_data_change": "Pull 20 raw request payloads from {segment} and confirm the field is absent or renamed at "
                            "source; diff against a payload from before onset.",
    "population_drift": "Profile the new segment ({segment}) with the client: expected volume, customer profile, and "
                        "whether these outcomes are genuinely lower-risk.",
    "model_release_regression": "Replay 1% of traffic against the previous version ({old_version}) in shadow and compare "
                                "latency and scores.",
    "serving_infrastructure": "Check provider status and per-node latency for the serving and feature-store fleet.",
    "config_threshold_change": "Diff the live decision config against the last approved version.",
    "concept_drift": "Backtest the model on the most recent matured vintage vs the training period.",
}


def _summary(category: str, ctx: Ctx) -> str:
    f = ctx.facts
    if category == "upstream_data_change" and f.get("missing_feature"):
        seg = f.get("missing_segment")
        where = f" from {seg['column']} = {seg['value']}" if seg else ""
        change = next((c for c in f.get("changes", []) if c["kind"] == "data_source"), None)
        tail = f", starting {c_when(change)} “{change['message'][:70]}…”" if change else ""
        return (f"{f['missing_feature']} arrives empty on {pct(f['missing_after'])} of requests{where}{tail}; "
                "median imputation turns those into ‘average’ profiles.")
    if category == "population_drift" and f.get("drift_features"):
        seg = f.get("drift_segment")
        feats = ", ".join(x for x, _ in f["drift_features"][:3])
        if seg:
            return (f"A {'new' if seg['share_before'] < 0.005 else 'growing'} segment ({seg['column']} = {seg['value']}, "
                    f"{pct(seg['share_after'])} of traffic) shifts {feats} into ranges the model links to risk.")
        return f"Inputs ({feats}) shifted away from the training population."
    if category == "model_release_regression":
        if f.get("new_version"):
            return (f"Release {f['new_version']} degraded serving: p95 {f['p95_before']:.0f}ms → {f['p95_after']:.0f}ms, "
                    f"{pct(f['timeout_after'])} fallback decisions; {f['old_version']} was healthy.")
        return "A deployment near onset may have changed model behaviour."
    if category == "serving_infrastructure":
        return f"Serving layer slowed: p95 {f.get('p95_before', 0):.0f}ms → {f.get('p95_after', 0):.0f}ms."
    if category == "config_threshold_change":
        return "Decisions shifted without a matching change in scores — consistent with a threshold/config change."
    if category == "concept_drift":
        if ctx.signals.get("feature_drift", 0) >= 0.15:
            return "Quality dropped on matured labels, but inputs shifted too — input drift explains the drop more directly."
        return "Quality dropped on matured labels while inputs stayed stable — the input→outcome relationship may have moved."
    return label(category)


def c_when(change: dict) -> str:
    h = change["hours_before_onset"]
    amount = f"{abs(h) * 60:.0f} min" if abs(h) < 2 else f"{abs(h):.1f} h"
    return f"{amount} after" if h >= 0 else f"{amount} before"


def score_hypotheses(ctx: Ctx, prior: dict[str, float]) -> list[dict[str, Any]]:
    ev_by_id = {e["id"]: e for e in ctx.evidence}
    out = []
    for h in HYPOTHESES:
        cat = h["category"]
        supporting, contradicting = [], []
        s = 0.0
        for sig, w in h["supports"].items():
            v = ctx.signals.get(sig, 0.0)
            s += w * v
            if v >= 0.15:
                supporting.append({"signal": sig, "weight": w, "strength": v, "contribution": round(w * v, 3),
                                   "evidence": [ev_by_id[e] for e in ctx.signal_evidence.get(sig, [])]})
        for sig, w in h["contradicts"].items():
            v = ctx.signals.get(sig, 0.0)
            s -= w * v
            if v >= 0.15:
                contradicting.append({"signal": sig, "weight": w, "strength": v, "contribution": round(-w * v, 3),
                                      "evidence": [ev_by_id[e] for e in ctx.signal_evidence.get(sig, [])]})
        p = prior.get(cat, 0.0)
        s += PRIOR_WEIGHT * p
        missing = [ABSENT[h["key"]]] if ctx.signals.get(h["key"], 0.0) < 0.15 and h["key"] in ABSENT else []
        if s < 0.15:
            summary = "Little supporting evidence. " + (missing[0] if missing else "")
        else:
            summary = _summary(cat, ctx)
        seg = ctx.facts.get("missing_segment") or ctx.facts.get("drift_segment") or {}
        verify = VERIFY[cat].format(segment=f"{seg.get('column', 'the affected segment')} = {seg.get('value', '…')}"
                                    if seg else "the affected segment",
                                    old_version=ctx.facts.get("old_version", "the previous version"))
        out.append({
            "category": cat, "label": label(cat), "score": round(max(s, 0.0), 3), "prior": round(p, 3),
            "supporting": sorted(supporting, key=lambda x: -x["contribution"]),
            "contradicting": sorted(contradicting, key=lambda x: x["contribution"]),
            "missing": missing, "verify": verify, "summary": summary.strip(),
        })
    exps = [math.exp(h["score"] / TEMPERATURE) for h in out]
    total = sum(exps)
    for h, e in zip(out, exps):
        h["confidence"] = round(e / total, 3)
    out.sort(key=lambda h: -h["confidence"])
    for i, h in enumerate(out):
        h["rank"] = i + 1
    return out
