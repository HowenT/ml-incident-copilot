"""Turn a diagnosis into an owner-assigned action plan: verify -> mitigate -> fix -> prevent -> communicate.

Every action carries a rationale that points back at the evidence, and a plain-language line
used in the customer update. Client-owned actions become "what we need from you".
"""
from __future__ import annotations

import re
from typing import Any

from ..stats import pct
from .glossary import plain

FDE = "FDE (incident owner)"


def _money(x: float | None) -> str:
    if not x:
        return "$0"
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}k" if x >= 1e3 else f"${x:,.0f}"


def _log_hint(facts: dict, pattern: str) -> str | None:
    for b in facts.get("log_bursts", []):
        m = re.search(pattern, b["example"])
        if m:
            return m.group(1)
    return None


def generate_actions(diag: dict[str, Any], model: dict[str, Any], client: str) -> list[dict[str, Any]]:
    top = diag["hypotheses"][0]
    cat = top["category"]
    f = diag["facts"]
    n = diag["impact"]["numbers"]
    acts: list[dict[str, Any]] = []

    def add(priority: str, kind: str, title: str, detail: str, owner: str, rationale: str, customer_text: str = "") -> None:
        acts.append({"priority": priority, "kind": kind, "title": title, "detail": detail, "owner": owner,
                     "rationale": rationale, "customer_text": customer_text,
                     "ours": 0 if owner.startswith(client.split()[0]) else 1})

    add("P0", "verify", "Confirm the root cause", top["verify"], FDE,
        f"Top hypothesis “{top['label']}” at {top['confidence']:.0%}. A 20-minute confirmation de-risks every action below.",
        "We are confirming the exact cause with a direct check of the affected data.")

    c0 = client.split()[0]
    if cat == "upstream_data_change":
        feat = f.get("missing_feature", "the missing field")
        seg = f.get("missing_segment") or {}
        seg_txt = f"{seg.get('column')} = {seg.get('value')}" if seg else "the affected source"
        found = _log_hint(f, r"found '([^']+)'")
        change = next((c for c in f.get("changes", []) if c["kind"] == "data_source"), None)
        ticket = re.search(r"(CHG-\d+)", change["message"]).group(1) if change and re.search(r"CHG-\d+", change["message"]) else "the routing change"
        add("P0", "mitigate", f"Hold auto-approvals for applications missing {feat}",
            f"Route applications where {feat} is null ({pct(f.get('missing_after'))} of volume, concentrated in {seg_txt}) "
            "to manual review until the feed is fixed.",
            f"{c0} Credit Risk Ops",
            f"≈{n.get('excess_approvals', 0):,} excess approvals so far; approval {pct(f.get('decision_missing'))} on incomplete "
            f"profiles vs {pct(f.get('decision_complete'))} on complete ones.",
            f"Pause automatic approval for applications that arrive without a {plain(feat)} and review them manually.")
        add("P0", "mitigate", "Enable the critical-feature guardrail",
            f"Configure scoring to return REFER instead of a score when {', '.join(f.get('missing_features', [feat]))} "
            "are null, rather than silently imputing the training median.",
            f"{FDE} + ML Platform",
            "Median imputation is why the outage looked healthy from the serving side — no errors, just wrong decisions.",
            "Add a safeguard so the model refuses to score applications with missing credit data instead of guessing.")
        add("P1", "fix", f"Fix the field mapping for {seg.get('value', 'the new source')}",
            (f"The v2 payload renamed the field ('{found}' instead of '{feat}'). " if found else "")
            + f"Map it in the bureau connector and validate on 20 live payloads before re-enabling auto-approval.",
            f"{c0} Data Engineering",
            f"Error logs and the {ticket} change line up with onset to the minute.",
            f"Update {plain(seg.get('value'))} so the {plain(feat)} is read correctly.")
        add("P1", "fix", "Re-score affected applications and review approvals",
            f"After the fix, re-score the {n.get('affected', 0):,} affected applications and send the "
            f"{n.get('approved_affected', 0):,} approvals made on incomplete data ({_money(n.get('approved_value'))}) to credit review.",
            f"{FDE} + {c0} Credit Risk",
            "Decisions already made are the real exposure; re-scoring shows which approvals would have changed.",
            f"Re-assess the {n.get('approved_affected', 0):,} applications approved without complete data.")
        add("P2", "prevent", "Schema contract test + per-source completeness alert",
            "Contract test on bureau payloads in CI; alert when any critical field exceeds 5% nulls for one data source.",
            "ML Platform", "Would have caught this within the first hour.",
            "Add automatic checks that flag missing credit data within the hour.")
        add("P2", "prevent", "Loop the model owner into client integration changes",
            f"Changes like {ticket} should notify the model owner before go-live so payloads can be validated.",
            f"Account team + {c0} IT", "The change was made client-side without a model-impact review.",
            "Agree a short heads-up process for integration changes that feed the model.")

    elif cat == "population_drift":
        seg = f.get("drift_segment") or {}
        val = seg.get("value", "the new segment")
        rate, other = n.get("segment_block_rate"), n.get("other_block_rate")
        add("P0", "mitigate", f"Switch {val} from hard declines to step-up authentication",
            f"For {seg.get('column', 'segment')} = {val}, send transactions above the block threshold to step-up (OTP/3DS) "
            "instead of declining, until the model is recalibrated.",
            f"{c0} Card Fraud Ops",
            f"Block rate {pct(rate)} for {val} vs {pct(other)} for other cards; {n.get('false_positive_blocks', 0):,} confirmed "
            f"false positives ({_money(n.get('false_positive_value'))}) so far.",
            f"Ask {plain(val)} customers for a quick verification instead of declining their purchases.")
        add("P0", "mitigate", f"Interim threshold for {val} (if step-up is unavailable)",
            f"Raise the block threshold for {val} until its block rate is ≤{pct((other or 0.02) * 1.5)}; review fraud caught daily.",
            f"{FDE} + {c0} Card Fraud Ops",
            "Keeps protection on while cutting false declines; reversible within minutes.",
            f"Temporarily adjust the fraud settings for {plain(val)} transactions to reduce false declines.")
        feats = ", ".join(x for x, _ in f.get("drift_features", [])[:3])
        add("P1", "fix", "Recalibrate / retrain with post-launch data",
            f"Retrain with labelled post-launch traffic and give the model {seg.get('column', 'segment')} as a feature "
            "(or add a segment-specific calibration layer).",
            "ML Team", f"Drifted inputs: {feats}. The model has never seen legitimate customers with this profile.",
            f"Retrain the fraud model so it recognises typical {plain(val)} spending.")
        add("P1", "fix", f"Get the {val} customer profile from {c0}",
            "Expected volume, cardholder profile, travel patterns, and whether cardholders are pre-verified at onboarding.",
            f"{c0} Product / Partnerships", "Business context the model cannot infer from data alone.",
            f"Share the expected profile and volume of {plain(val)} customers with us.")
        add("P2", "prevent", "Launch checklist: shadow-score new programmes before go-live",
            "New card programmes and partner launches are announced to the model owner two weeks ahead and shadow-scored.",
            f"Account team + {c0} Product", "The launch reached the model with no warning.",
            "Agree a launch checklist so new programmes are tested against the model before go-live.")

    elif cat == "model_release_regression":
        new, old = f.get("new_version", "the new release"), f.get("old_version", "the previous version")
        ns = _log_hint(f, r"namespace (\S+?)[\s;:)]") or "the new feature namespace"
        add("P0", "mitigate", f"Roll back {model['id']} {new} → {old}",
            f"Latency and fallbacks are confined to {new}; {old} was healthy. Roll back first, debug second.",
            "ML Platform", f"p95 {f.get('p95_before', 0):.0f}ms → {f.get('p95_after', 0):.0f}ms; "
                           f"{pct(f.get('timeout_after'))} of transactions answered by fallback.",
            "Roll back to the previous, stable version of the fraud model.")
        add("P0", "mitigate", "Review transactions approved without a fraud score",
            f"{n.get('fallback', 0):,} transactions ({_money(n.get('fallback_value'))}) were approved by fallback. Score them "
            "offline with the previous version and send high scores to fraud ops for review.",
            f"{c0} Card Fraud Ops + {FDE}",
            f"≈{n.get('expected_fraud', 0):,} fraudulent transactions (≈{_money(n.get('expected_fraud_value'))}) expected at the "
            "baseline fraud rate.",
            "Review the transactions that were approved without a fraud check.")
        add("P1", "fix", f"Fix feature-store caching for {ns}",
            "Logs show the cache hit ratio collapsing with ttl 0s. Set a TTL, pre-warm the cache, and load-test at peak QPS "
            "before re-releasing.", "ML Platform",
            "Feature-store warnings start within minutes of the rollout.",
            "Fix the configuration that slowed the new version down, and load-test before re-release.")
        add("P2", "prevent", "Canary gate on latency and fallback rate",
            "Block promotion beyond a 5% canary if p95 > SLO or fallback rate > 0.5%.", "ML Platform",
            "A canary would have limited the blast radius to 5% of traffic for minutes.",
            "Release future versions gradually with automatic checks on response time.")
        add("P2", "prevent", "Revisit the ALLOW fallback for high-risk merchants",
            "Consider CHALLENGE instead of ALLOW when the fraud check times out on high-risk merchant categories.",
            f"{c0} Card Fraud Ops + {FDE}", "Fallback ALLOW turned a latency problem into a fraud-loss exposure.",
            "Review what happens to a payment when the fraud check is slow.")

    elif cat == "serving_infrastructure":
        add("P0", "mitigate", "Add capacity / fail over the serving fleet",
            "Scale out model-serving and feature-store replicas; fail over to the secondary zone if latency persists.",
            "ML Platform", f"p95 {f.get('p95_before', 0):.0f}ms → {f.get('p95_after', 0):.0f}ms without a model change.",
            "Add capacity to the systems that run the model.")
        add("P0", "mitigate", "Review transactions approved without a score",
            f"{n.get('fallback', 0):,} transactions approved by fallback; score offline and review the riskiest.",
            f"{c0} Card Fraud Ops + {FDE}", "Fallback approvals are unscored exposure.",
            "Review the transactions approved without a fraud check.")
        add("P2", "prevent", "Synthetic latency probes + fallback policy review",
            "Probe p99 per node every minute; route high-risk categories to CHALLENGE on timeout.", "ML Platform",
            "Detect infra slowdowns before customers do.", "Add early-warning checks on response time.")

    elif cat == "config_threshold_change":
        add("P0", "mitigate", "Revert the decision config to the last approved version",
            "Diff live vs approved config; revert and confirm the decision rate returns to baseline within an hour.",
            "ML Platform", "Decisions moved while scores did not.", "Restore the approved decision settings.")
        add("P1", "fix", "Re-decision affected requests", "Replay decisions made under the unapproved config.",
            f"{FDE} + {c0} Risk", "Customers were decided under settings nobody approved.",
            "Re-check decisions made under the wrong settings.")
        add("P2", "prevent", "Four-eyes approval and effective dates for decision configs",
            "Require approval and an explicit effective date for threshold changes.", "ML Platform",
            "Stops early or accidental config pushes.", "Add an approval step for decision-setting changes.")

    else:  # concept_drift
        add("P1", "fix", "Backtest on the latest matured vintage and plan a retrain",
            "Compare feature importance and calibration on recent outcomes vs the training period.", "ML Team",
            "Quality fell while inputs stayed stable.", "Retrain the model on recent outcomes.")
        add("P1", "mitigate", "Interim threshold adjustment with risk sign-off",
            "Adjust the cut-off to hold the expected loss rate until the retrain ships.", f"{FDE} + {c0} Risk",
            "Limits losses while the retrain is prepared.", "Adjust decision settings temporarily, with your sign-off.")
        add("P2", "prevent", "Scheduled challenger retrains", "Quarterly challenger model with macro-scenario backtests.",
            "ML Team", "Concept drift is gradual; cadence beats heroics.", "Set up regular model refreshes.")

    # Reuse what worked last time.
    reuse = next((s for s in diag.get("similar", []) if s["root_cause_category"] == cat and s["similarity"] >= 0.55), None)
    if reuse:
        for a in acts:
            if a["kind"] == "fix":
                a["rationale"] += f" Reused from {reuse['code']}: {reuse['fix_applied']}"
                break

    add("P0", "communicate", f"Send the customer update to {c0}",
        "Use the customer update (Updates view). Agree the next update time and who approves mitigations on their side.",
        FDE, "The client raised this; a clear update within the hour protects trust more than a perfect one later.")
    add("P2", "prevent", "Record the confirmed root cause (feedback loop)",
        "Close the incident with the actual root cause and fix so future diagnoses and similar-incident search improve.",
        FDE, "Feeds the historical prior used to rank hypotheses.")
    order = {"P0": 0, "P1": 1, "P2": 2}
    acts.sort(key=lambda a: order[a["priority"]])
    return acts
