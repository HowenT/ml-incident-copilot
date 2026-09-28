"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'copilot.db').as_posix()}")
MODEL_ARTIFACT_DIR = DATA_DIR / "models"

# Optional LLM polish for summaries. The deterministic template path always works;
# Claude is only used when an API key is configured and the user asks for it.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
LLM_DISABLED = os.getenv("LLM_DISABLED", "").lower() in {"1", "true", "yes"}

CLIENT_NAME = os.getenv("CLIENT_NAME", "Harborline Financial")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "7"))


def llm_available() -> bool:
    if LLM_DISABLED:
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def utcnow():
    """Naive UTC now (the database stores naive UTC timestamps)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(tzinfo=None)
