"""Two audiences, one set of facts.

* Engineer summary — technical diagnosis: evidence, alternatives, blast radius, checks, plan.
* Customer summary — plain language for the client's business owner: what happened, what it
  means for them, why, what we are doing, what we need from them, when they hear next.

Both are deterministic templates over the diagnosis, so every number is traceable. An LLM
can optionally polish the wording (see llm.py) but is instructed not to add facts.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..stats import pct
from ..taxonomy import ROOT_CAUSES
from .glossary import plain


def _dt(iso_str: str | None) -> str:
    if not iso_str:
        return "n/a"
    d = datetime.fromisoformat(iso_str)
    return d.strftime("%a %d %b, %H:%M UTC")


def _money(x: float | None) -> str:
    if not x:
        return "$0"
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}k" if x >= 1e3 else f"${x:,.0f}"


def _a(action: Any, key: str) -> Any:
    return action[key] if isinstance(action, dict) else getattr(action, key)


# --------------------------------------------------------------------------------------
# Engineer
# --------------------------------------------------------------------------------------


def engineer_summary(inc: Any, model: dict, diag: dict, actions: list[Any]) -> str:
    top = diag["hypotheses"][0]
    lines = [
        f"### {inc.code} · Technical diagnosis",
        f"**Model** `{model['id']}` v{model['current_version']} · **Client** {inc.customer} · "
        f"**Severity** {inc.severity} · **Status** {inc.status}",
        f"**Onset** {_dt(diag['onset'])} — {diag['onset_method']}  ",
        f"**Analysed** {diag['window']['start']} → {diag['window']['end']} vs baseline "
        f"{diag['baseline']['start']} → {diag['baseline']['end']} "
        f"({diag['rows']['incident']:,} vs {diag['rows']['baseline']:,} requests)",
        "",
        f"#### Most likely root cause: {top['label']} ({top['confidence']:.0%})",
        top["summary"],
        "",
        "**Evidence**",
    ]
    for s in top["supporting"]:
        for e in s["evidence"][:2]:
            lines.append(f"- {e['title']}. {e['detail']}".rstrip())
    if top["contradicting"]:
        lines += ["", "**Counter-evidence**"]
        for s in top["contradicting"]:
            for e in s["evidence"][:1]:
                lines.append(f"- {e['title']}")
    lines += ["", "#### Alternatives considered", "| Hypothesis | Confidence | Why it ranks lower |", "|---|---|---|"]
    for h in diag["hypotheses"][1:]:
        why = (h["missing"][0] if h["missing"] else
               h["contradicting"][0]["evidence"][0]["title"] if h["contradicting"] and h["contradicting"][0]["evidence"] else
               "Weaker supporting evidence.")
        lines.append(f"| {h['label']} | {h['confidence']:.0%} | {why} |")
    lines += ["", "#### Blast radius", diag["impact"]["headline"]]
    for m in diag["impact"]["metrics"]:
        lines.append(f"- **{m['label']}:** {m['value']}" + (f" ({m['detail']})" if m["detail"] else ""))
    lines += ["", "#### Checks run", "| Layer | Result | Detail |", "|---|---|---|"]
    icon = {"flagged": "⚠️ flagged", "clear": "✅ clear", "n/a": "➖ n/a"}
    for c in diag["checks"]:
        lines.append(f"| {c['label']} | {icon.get(c['status'], c['status'])} | {c['headline']} |")
    lines += ["", "#### Action plan"]
    for i, a in enumerate(actions, 1):
        done = " ✔" if _a(a, "status") == "done" else ""
        lines.append(f"{i}. **[{_a(a, 'priority')} · {_a(a, 'kind')}]** {_a(a, 'title')} — _{_a(a, 'owner')}_{done}")
    lines += ["", "#### Open questions", f"- {top['verify']}"]
    if not diag["facts"].get("labels_available", True):
        lines.append(f"- Ground-truth labels lag ≈{model['label_lag_hours'] // 24} days: re-check model quality once "
                     "the incident window matures.")
    if diag.get("similar"):
        lines += ["", "_Similar past incidents: " + "; ".join(
            f"{s['code']} ({s['similarity']:.0%}, {s['root_cause_label']})" for s in diag["similar"]) + "_"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Customer
# --------------------------------------------------------------------------------------


def _what_happened(cat: str, diag: dict, model: dict) -> str:
    f, n = diag["facts"], diag["impact"]["numbers"]
    onset = _dt(diag["onset"])
    if cat == "upstream_data_change" and f.get("missing_feature"):
        seg = f.get("missing_segment") or {}
        dseg = f.get("decision_segment") or {}
        where = f", all of them coming through {plain(seg.get('value'))}" if seg else ""
        region = (f" This is why the approval rate in the {plain(dseg['value'])} rose from {pct(dseg['before'], 0)} to "
                  f"{pct(dseg['after'], 0)}.") if dseg else ""
        return (f"Since {onset}, about {pct(f['missing_after'], 0)} of loan applications "
                f"({n.get('affected', 0):,} so far) reached the {plain(model['id'])} without the applicant's "
                f"{plain(f['missing_feature'])}{where}. When that information is missing, the model fills the gap with a "
                f"typical value, so riskier applicants look average. Applications missing the score were approved "
                f"{pct(f.get('decision_missing'), 0)} of the time, compared with {pct(f.get('decision_complete'), 0)} for "
                f"complete applications.{region}")
    if cat == "population_drift":
        seg = f.get("drift_segment") or {}
        return (f"Since around {onset}, a new group of customers ({plain(seg.get('value'))} holders) has been using their cards in "
                f"ways the {plain(model['id'])} rarely saw when it was trained: for example, larger purchases far from home on "
                f"accounts that are only weeks old. The model reads that pattern as risky, so it has been declining "
                f"{pct(n.get('segment_block_rate'), 0)} of {plain(seg.get('value'))} transactions, compared with "
                f"{pct(n.get('other_block_rate'), 1)} for other cards. Most of these declines were genuine customers.")
    if cat == "model_release_regression":
        return (f"Since {onset}, shortly after we released a new version of the {plain(model['id'])}, the fraud check has "
                f"been responding too slowly for about {pct(f.get('timeout_after'), 0)} of card transactions. When the check "
                f"takes too long, the payment gateway approves the transaction without a fraud score (its standard "
                f"fallback). {n.get('fallback', 0):,} transactions ({_money(n.get('fallback_value'))}) were approved this way.")
    if cat == "serving_infrastructure":
        return (f"Since {onset}, the systems that run the {plain(model['id'])} have been responding slowly, and some "
                f"requests received the fallback decision instead of a model decision.")
    if cat == "config_threshold_change":
        return f"Since {onset}, decisions changed because the decision settings applied to model scores changed."
    return (f"Since {onset}, the {plain(model['id'])} has become less accurate, even though the data it receives looks "
            f"normal. Customer behaviour appears to have shifted relative to the period the model learned from.")


def _meaning(cat: str, diag: dict) -> list[str]:
    n = diag["impact"]["numbers"]
    if cat == "upstream_data_change":
        return [f"{n.get('approved_affected', 0):,} applications were approved using incomplete information "
                f"({_money(n.get('approved_value'))} in loans).",
                f"We estimate roughly {n.get('excess_approvals', 0):,} of those would not have been approved with complete "
                f"data (≈{_money(n.get('excess_loan_value'))}).",
                "No customer data was lost. The information exists at the bureau; it just didn't reach the model."]
    if cat == "population_drift":
        return [f"About {n.get('excess_blocks', 0):,} more transactions were declined than usual.",
                f"{n.get('false_positive_blocks', 0):,} of the declines are already confirmed as genuine purchases "
                f"({_money(n.get('false_positive_value'))}), and more will be confirmed as outcomes come in.",
                "Fraud protection for your other cards is working normally."]
    if cat in ("model_release_regression", "serving_infrastructure"):
        return ["No genuine customers were declined because of this. The risk is fraud getting through without a check.",
                f"At normal fraud rates we would expect about {n.get('expected_fraud', 0):,} fraudulent transactions "
                f"(≈{_money(n.get('expected_fraud_value'))}) among those approved without a score.",
                "Some customers experienced slower checkout."]
    return [diag["impact"]["headline"]]


def customer_summary(inc: Any, model: dict, diag: dict, actions: list[Any], now: datetime) -> str:
    top = diag["hypotheses"][0]
    cat = top["category"]
    contact = (inc.customer_contact or "team").split(",")[0].split()[0]
    sender = (inc.reported_by or "Your FDE").split("(")[0].strip()
    ours = [a for a in actions if _a(a, "ours") and _a(a, "customer_text")]
    theirs = [a for a in actions if not _a(a, "ours") and _a(a, "customer_text")]
    certainty = ("We are confident about the cause" if top["confidence"] >= 0.75 else
                 "This is our leading explanation, and we are confirming it now")
    lines = [
        f"**Subject:** {inc.title}: what happened and next steps ({inc.code})",
        "",
        f"Hi {contact},",
        "",
        "**What happened**  ",
        _what_happened(cat, diag, model),
        "",
        "**What this means for you**",
        *[f"- {m}" for m in _meaning(cat, diag)],
        "",
        "**Why it happened**  ",
        f"{certainty}: {ROOT_CAUSES[cat]['plain']}. " + _why_detail(cat, diag),
        "",
        "**What we are doing**",
    ]
    for a in ours:
        prefix = "Done: " if _a(a, "status") == "done" else ""
        lines.append(f"- {prefix}{_a(a, 'customer_text')}")
    if theirs:
        lines += ["", "**What we need from you**"]
        for a in theirs:
            prefix = "Done, thank you: " if _a(a, "status") == "done" else ""
            lines.append(f"- {prefix}{_a(a, 'customer_text')}")
    lines += ["", f"**Next update:** by {(now + timedelta(hours=4)).strftime('%a %d %b, %H:%M UTC')}, or sooner if "
                  "anything changes.", "", "Best,  ", sender]
    return "\n".join(lines)


def _why_detail(cat: str, diag: dict) -> str:
    f = diag["facts"]
    change = next((c for c in f.get("changes", []) if c["kind"] in ("data_source", "deploy", "business")), None)
    if cat == "upstream_data_change" and change:
        return (f"The problem started {_ago(change['hours_before_onset'])} after an integration change on your side "
                f"(“{change['message'].split(':', 1)[-1].strip()}”). The new connection sends the credit score under a "
                "different field name, so our side received it as empty.")
    if cat == "population_drift" and change:
        return ("It lines up with the launch of the new card programme. Nothing is broken; the model simply hasn't "
                "learned what normal looks like for these customers yet.")
    if cat == "model_release_regression" and f.get("new_version"):
        return (f"The slowdown affects only the new version ({f['new_version']}); the previous version was fast. It needs "
                "data from a new store that was not configured to cache results, which added delay to every request.")
    return ""


def _ago(hours: float) -> str:
    return f"{hours * 60:.0f} minutes" if hours < 2 else f"{hours:.0f} hours"


def build_summaries(inc: Any, model: dict, diag: dict, actions: list[Any], now: datetime) -> dict[str, str]:
    return {"engineer": engineer_summary(inc, model, diag, actions),
            "customer": customer_summary(inc, model, diag, actions, now)}

