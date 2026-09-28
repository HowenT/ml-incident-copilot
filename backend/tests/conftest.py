import os
import tempfile

# Must run before any backend import: the engine is created from these at import time.
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="mlic-test-")
os.environ.pop("DATABASE_URL", None)
os.environ["AUTO_SEED"] = "0"
os.environ["LLM_DISABLED"] = "1"

import json  # noqa: E402
from datetime import datetime  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import select  # noqa: E402

from backend.app.db import SessionLocal  # noqa: E402
from backend.app.models import Alert  # noqa: E402
from backend.app.seed import seed  # noqa: E402
from backend.app.services.knowledge_seed import SAMPLE_REPORTS  # noqa: E402
from backend.app.services.store import meta  # noqa: E402
from backend.app.services.workflow import create_incident, diagnose  # noqa: E402

ANCHOR = datetime(2026, 9, 28, 12, 0)


@pytest.fixture(scope="session")
def seeded():
    seed(ANCHOR, verbose=False)
    with SessionLocal() as db:
        truth = json.loads(meta(db, "ground_truth"))
        truth["alerts_at_seed"] = {(a.model_id, a.rule, a.status != "resolved") for a in db.scalars(select(Alert))}
        return truth


@pytest.fixture(scope="session")
def scenario_incidents(seeded):
    """Open + diagnose the two live-demo incidents exactly as a user would."""
    out = {}
    with SessionLocal() as db:
        for key, model, rules in [("credit-bureau", "credit-risk", {"missing_values", "decision_rate_shift"}),
                                  ("fraud-latency", "fraud-detect", {"latency_slo", "serving_errors"})]:
            alerts = db.scalars(select(Alert).where(Alert.model_id == model, Alert.status == "open",
                                                    Alert.incident_id.is_(None))).all()
            ids = [a.id for a in alerts if a.rule in rules]
            inc = create_incident(db, {**SAMPLE_REPORTS[key], "alert_ids": ids})
            out[key] = {"id": inc.id, "diagnosis": diagnose(db, inc)}
    return out
