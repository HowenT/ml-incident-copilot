"""Blast radius: translate the anomaly into customer-facing numbers (requests, decisions, dollars).

Counts and dollar values are extrapolated by the model's prediction-log sample rate (the fraud
model logs 10% of authorisations), and the extrapolation is stated in the output.
"""
from __future__ import annotations

from typing import Any

from ..stats import pct
from .checks import _fmt_money
from .context import Ctx


def _m(label: str, value: str, detail: str = "") -> dict[str, str]:
    return {"label": label, "value": value, "detail": detail}


def assess_impact(ctx: Ctx, category: str) -> dict[str, Any]:
    full = ctx.cur_full
    rate = float(ctx.model.get("log_sample_rate") or 1.0)
    k = 1.0 / rate  # scale sample counts to production volume

    def n(x: float) -> float:
        return x * k

    hours = (ctx.data_end - ctx.onset).total_seconds() / 3600
    metrics = [_m("Duration so far", f"{hours:.0f} h", "from detected onset to latest data"),
               _m("Requests since onset", f"{n(len(full)):,.0f}",
                  "" if rate == 1 else f"extrapolated from a {rate:.0%} prediction-log sample")]
    headline = ""
    numbers: dict[str, Any] = {"duration_h": round(hours, 1), "requests": round(n(len(full))), "sample_rate": rate}

    if ctx.model_type == "credit":
        f = ctx.facts.get("missing_feature")
        if f:
            aff = full[full[f].isna()]
            complete = full[full[f].notna()]
            ar_aff = float(aff["decision"].mean()) if len(aff) else 0.0
            ar_ok = float(complete["decision"].mean()) if len(complete) else 0.0
            approved = aff[aff["decision"] == 1]
            excess = max(n(len(aff)) * (ar_aff - ar_ok), 0)
            avg_loan = float(approved["loan_amount"].mean()) if len(approved) else 0.0
            approved_value = n(float(approved["loan_amount"].sum()))
            metrics += [
                _m("Applications scored without " + f, f"{n(len(aff)):,.0f}", f"{pct(len(aff) / max(len(full), 1))} of volume"),
                _m("Approved on incomplete data", f"{n(len(approved)):,.0f}", f"{_fmt_money(approved_value)} in loans"),
                _m("Estimated excess approvals", f"≈{excess:,.0f}",
                   f"approval {pct(ar_aff)} vs {pct(ar_ok)} for complete profiles · ≈{_fmt_money(excess * avg_loan)}"),
            ]
            headline = (f"{n(len(aff)):,.0f} applications were decided without {f}; about {excess:,.0f} more were approved "
                        f"than complete profiles would have been (≈{_fmt_money(excess * avg_loan)} in loans).")
            numbers.update(affected=round(n(len(aff))), approved_affected=round(n(len(approved))),
                           excess_approvals=round(excess), excess_loan_value=round(excess * avg_loan),
                           approved_value=round(approved_value))
        else:
            dr_b, dr_c = ctx.facts.get("decision_before", 0), float(full["decision"].mean())
            excess = (dr_c - dr_b) * n(len(full))
            metrics.append(_m("Decision change vs baseline", f"{excess:+,.0f} approvals", f"{pct(dr_b)} → {pct(dr_c)}"))
            headline = f"Approval rate moved {pct(dr_b)} → {pct(dr_c)} ({excess:+,.0f} approvals vs baseline)."
            numbers.update(excess_approvals=round(excess))
        return {"headline": headline, "metrics": metrics, "numbers": numbers}

    # fraud
    base_lab = ctx.base[ctx.base["label"].notna()]
    fraud_rate = float(base_lab["label"].mean()) if len(base_lab) else 0.012
    avg_fraud_amt = float(base_lab.loc[base_lab["label"] == 1, "amount"].mean()) if base_lab["label"].sum() else 250.0
    fallback = full[full["status"] != "ok"]
    if category in ("model_release_regression", "serving_infrastructure") or len(fallback) > 0.01 * len(full):
        count, value = n(len(fallback)), n(float(fallback["amount"].sum()))
        exp_fraud = count * fraud_rate
        metrics += [
            _m("Approved without a fraud score", f"{count:,.0f}",
               f"{pct(len(fallback) / max(len(full), 1))} of transactions · {_fmt_money(value)}"),
            _m("Expected fraud let through", f"≈{exp_fraud:,.0f} txns",
               f"≈{_fmt_money(exp_fraud * avg_fraud_amt)} at the baseline fraud rate {pct(fraud_rate, 2)}"),
        ]
        headline = (f"{count:,.0f} transactions ({_fmt_money(value)}) were approved without a fraud check; at the usual "
                    f"fraud rate that is ≈{exp_fraud:,.0f} fraudulent transactions (≈{_fmt_money(exp_fraud * avg_fraud_amt)}).")
        numbers.update(fallback=round(count), fallback_value=round(value), expected_fraud=round(exp_fraud),
                       expected_fraud_value=round(exp_fraud * avg_fraud_amt))
    if category in ("population_drift", "concept_drift", "config_threshold_change") or not headline:
        dr_b = ctx.facts.get("decision_before", float(ctx.base["decision"].mean()))
        dr_c = float(full["decision"].mean())
        excess = max((dr_c - dr_b) * n(len(full)), 0)
        lab = full[full["label"].notna() & (full["decision"] == 1)]
        fp = lab[lab["label"] == 0]
        fp_count, fp_value = n(len(fp)), n(float(fp["amount"].sum()))
        metrics += [
            _m("Extra blocked transactions vs baseline", f"≈{excess:,.0f}", f"block rate {pct(dr_b)} → {pct(dr_c)}"),
            _m("Confirmed false-positive blocks", f"{fp_count:,.0f}",
               f"{_fmt_money(fp_value)} of legitimate spend declined (labelled so far)"),
        ]
        seg = ctx.facts.get("drift_segment")
        if seg:
            seg_rows = full[full[seg["column"]].astype(str) == seg["value"]]
            other = full[full[seg["column"]].astype(str) != seg["value"]]
            metrics.append(_m(f"Block rate for {seg['value']}", pct(float(seg_rows["decision"].mean())),
                              f"vs {pct(float(other['decision'].mean()))} for everyone else · {n(len(seg_rows)):,.0f} txns"))
            numbers.update(segment_block_rate=float(seg_rows["decision"].mean()),
                           other_block_rate=float(other["decision"].mean()), segment_txns=round(n(len(seg_rows))))
        headline = (f"About {excess:,.0f} more transactions were blocked than usual; {fp_count:,.0f} confirmed legitimate "
                    f"purchases ({_fmt_money(fp_value)}) were declined so far.")
        numbers.update(excess_blocks=round(excess), false_positive_blocks=round(fp_count), false_positive_value=round(fp_value))
    return {"headline": headline, "metrics": metrics, "numbers": numbers}
