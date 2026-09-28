"""Optional: polish a template summary with Claude, without letting it invent facts.

The deterministic template is always generated first and is the source of truth. Claude only
rewrites wording for the audience; the prompt pins every number to the supplied draft/facts.
If no API credentials are configured, or the call fails, callers keep the template.
"""
from __future__ import annotations

import json
from typing import Any

from ..config import ANTHROPIC_MODEL, llm_available

SYSTEM = (
    "You edit incident communications for a Forward Deployed Engineering team that supports ML models "
    "in production at a financial-services client. You receive a draft and the structured facts behind it. "
    "Rewrite the draft for the stated audience. Rules: use only facts present in the draft or the facts JSON; "
    "keep every number, date, version and identifier exactly as given; do not add causes, commitments or "
    "timelines that are not in the input; keep the same section structure and Markdown formatting. "
    "Return only the rewritten Markdown."
)

AUDIENCE = {
    "engineer": "On-call ML/platform engineers. Be precise and terse; keep tables; lead with the most decision-relevant evidence.",
    "customer": "The client's business owner (credit/fraud risk lead). No jargon (no PSI, p95, imputation, "
                "feature names); warm, accountable, concise; make the ask of them explicit.",
}


class LLMUnavailable(RuntimeError):
    pass


def polish(audience: str, draft: str, facts: dict[str, Any]) -> tuple[str, str]:
    """Return (markdown, generator_label)."""
    if not llm_available():
        raise LLMUnavailable("No Anthropic credentials configured (set ANTHROPIC_API_KEY).")
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Audience: {AUDIENCE[audience]}\n\n<facts>\n{json.dumps(facts, default=str)}\n</facts>\n\n"
                           f"<draft>\n{draft}\n</draft>",
            }],
        )
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable(f"Could not reach the Anthropic API: {e}") from e
    except anthropic.RateLimitError as e:
        raise LLMUnavailable("Anthropic API rate limit reached; try again shortly.") from e
    except anthropic.APIStatusError as e:
        raise LLMUnavailable(f"Anthropic API error {e.status_code}: {e.message}") from e

    if response.stop_reason == "refusal":
        raise LLMUnavailable("The model declined this request; keeping the template summary.")
    if response.stop_reason == "max_tokens":
        raise LLMUnavailable("Response was cut off; keeping the template summary.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise LLMUnavailable("Empty response; keeping the template summary.")
    return text, f"claude:{response.model}"
