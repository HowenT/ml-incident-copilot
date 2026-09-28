"""Synthetic client environment: two production models, 30 days of traffic, three incidents.

Client: Harborline Financial (fictional) — consumer lender + card issuer.

Models are *really* trained (scikit-learn) on a reference population, then score simulated
production traffic. Incidents are injected into the traffic / serving layer only, so the
downstream effects (approval-rate jump, false-positive blocks, timeouts) emerge from the
model's actual behaviour instead of being hard-coded.

Scenarios (relative to the data end "now"):
  A  credit-risk   T-2d 02:00  Client routes West + 30% of Midwest bureau pulls to a new
                               bureau API version whose payload renames `bureau_score`.
                               The connector emits nulls, the pipeline silently imputes the
                               median, risky applicants look average -> approvals jump.
  B  fraud-detect  T-1d 09:00  Release 7.3.0 adds real-time velocity features; the new
                               feature-store namespace ships with cache TTL=0 -> p95 latency
                               5x, ~6% gateway timeouts, fallback policy ALLOWs unscored txns.
  C  fraud-detect  T-9d 00:00  Client launches the "TravelPlus" co-brand card. New, legitimate
                               cardholders transact abroad with large amounts on brand-new
                               accounts -> covariate drift, block rate doubles, precision falls.
Noise: a 2-hour network blip (auto-resolved alerts) and benign deploys far from any onset.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import utcnow

# --------------------------------------------------------------------------------------
# Model specs
# --------------------------------------------------------------------------------------

CREDIT_FEATURES = [
    {"name": "age", "kind": "num", "critical": False, "description": "Applicant age"},
    {"name": "annual_income", "kind": "num", "critical": True, "description": "Stated annual income (USD)"},
    {"name": "employment_years", "kind": "num", "critical": False, "description": "Years with current employer"},
    {"name": "debt_to_income", "kind": "num", "critical": True, "description": "Debt-to-income ratio"},
    {"name": "credit_utilization", "kind": "num", "critical": True, "description": "Revolving credit utilization"},
    {"name": "bureau_score", "kind": "num", "critical": True, "description": "Credit bureau score (300-850)"},
    {"name": "bureau_inquiries_6m", "kind": "num", "critical": False, "description": "Hard inquiries, last 6 months"},
    {"name": "loan_amount", "kind": "num", "critical": True, "description": "Requested loan amount (USD)"},
    {"name": "loan_term_months", "kind": "num", "critical": False, "description": "Loan term"},
    {"name": "region", "kind": "cat", "critical": False, "description": "Applicant region"},
    {"name": "channel", "kind": "cat", "critical": False, "description": "Application channel"},
]
CREDIT_SEGMENTS = ["region", "channel", "bureau_source", "model_version"]

FRAUD_FEATURES = [
    {"name": "amount", "kind": "num", "critical": True, "description": "Transaction amount (USD)"},
    {"name": "merchant_category", "kind": "cat", "critical": True, "description": "Merchant category"},
    {"name": "hour_of_day", "kind": "num", "critical": False, "description": "Local hour of transaction"},
    {"name": "distance_from_home_km", "kind": "num", "critical": False, "description": "Distance from home address"},
    {"name": "is_international", "kind": "num", "critical": False, "description": "Cross-border transaction flag"},
    {"name": "card_present", "kind": "num", "critical": False, "description": "Card-present flag"},
    {"name": "txn_velocity_1h", "kind": "num", "critical": False, "description": "Transactions on card, last hour"},
    {"name": "account_age_days", "kind": "num", "critical": False, "description": "Days since account opened"},
    {"name": "device_risk_score", "kind": "num", "critical": False, "description": "Device fingerprint risk (0-1)"},
]
FRAUD_SEGMENTS = ["merchant_category", "card_program", "channel", "model_version"]

REGIONS = ["Northeast", "Southeast", "Midwest", "West"]
REGION_P = [0.24, 0.26, 0.22, 0.28]
CHANNELS = ["web", "mobile", "branch", "broker"]
CHANNEL_P = [0.45, 0.35, 0.12, 0.08]

MCC = ["grocery", "fuel", "restaurants", "online_retail", "electronics", "travel", "digital_goods", "atm"]
MCC_P = [0.22, 0.11, 0.16, 0.20, 0.06, 0.05, 0.10, 0.10]
MCC_AMOUNT = {"grocery": 45, "fuel": 48, "restaurants": 34, "online_retail": 72, "electronics": 240,
              "travel": 380, "digital_goods": 22, "atm": 120}
MCC_CARD_PRESENT = {"grocery": 0.93, "fuel": 0.95, "restaurants": 0.9, "online_retail": 0.0, "electronics": 0.5,
                    "travel": 0.25, "digital_goods": 0.0, "atm": 1.0}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Timeline:
    start: datetime
    end: datetime
    onset_bureau: datetime
    onset_deploy: datetime
    onset_travelplus: datetime
    blip_start: datetime
    credit_redeploy: datetime

    @property
    def travelplus_ramp_hours(self) -> int:
        return 36


def build_timeline(anchor: datetime | None = None, days: int = 30) -> Timeline:
    anchor = anchor or utcnow()
    end = anchor.replace(minute=0, second=0, microsecond=0)
    day0 = end.replace(hour=0)
    return Timeline(
        start=end - timedelta(days=days),
        end=end,
        onset_bureau=day0 - timedelta(days=2) + timedelta(hours=2),
        onset_deploy=day0 - timedelta(days=1) + timedelta(hours=9),
        onset_travelplus=day0 - timedelta(days=9),
        blip_start=day0 - timedelta(days=12) + timedelta(hours=14),
        credit_redeploy=day0 - timedelta(days=10) + timedelta(hours=11),
    )


def _arrivals(rng: np.random.Generator, tl: Timeline, per_hour: float, weekend: float, peak_hour: int,
              multiplier=None) -> np.ndarray:
    hours = pd.date_range(tl.start, tl.end, freq="h", inclusive="left")
    h = hours.hour.to_numpy()
    lam = per_hour * (1 + 0.6 * np.cos(2 * np.pi * (h - peak_hour) / 24))
    lam = lam * np.where(hours.dayofweek.to_numpy() >= 5, weekend, 1.0)
    if multiplier is not None:
        lam = lam * multiplier(hours)
    counts = rng.poisson(lam)
    starts = np.repeat(hours.to_numpy().astype("datetime64[s]"), counts)
    ts = starts + rng.integers(0, 3600, counts.sum()).astype("timedelta64[s]")
    return np.sort(ts)


# --------------------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------------------


def credit_population(rng: np.random.Generator, n: int) -> pd.DataFrame:
    region = rng.choice(REGIONS, n, p=REGION_P)
    channel = rng.choice(CHANNELS, n, p=CHANNEL_P)
    age = np.clip(rng.normal(41, 12, n), 21, 80).round()
    income = np.clip(rng.lognormal(np.log(62000), 0.45, n), 15000, 400000).round(-2)
    emp = np.clip(rng.gamma(2.0, 3.2, n), 0, 40).round(1)
    dti = np.clip(rng.beta(2.2, 6, n) * 0.9 + 0.02, 0.01, 0.85).round(3)
    util = np.clip(rng.beta(2, 3, n), 0, 1).round(3)
    score = np.clip(715 - 110 * (util - 0.4) - 60 * (dti - 0.26) + rng.normal(0, 55, n), 300, 850).round()
    inq = rng.poisson(np.clip(1.2 + (700 - score) / 110, 0.2, 6)).astype(float)
    loan = np.clip(rng.lognormal(np.log(15000), 0.6, n), 1000, 80000).round(-2)
    term = rng.choice([36, 60], n, p=[0.6, 0.4]).astype(float)
    logit = (
        -2.55
        - 0.014 * (score - 700)
        + 3.0 * (dti - 0.26)
        + 1.2 * (util - 0.4)
        + 0.2 * (inq - 1.2)
        - 0.35 * (np.log(income) - np.log(62000))
        - 0.03 * (emp - 6)
        + 0.2 * (np.log(loan) - np.log(15000))
        + 0.15 * (term == 60)
        - 0.01 * (age - 41)
    )
    label = (rng.random(n) < sigmoid(logit)).astype(int)
    return pd.DataFrame({
        "age": age, "annual_income": income, "employment_years": emp, "debt_to_income": dti,
        "credit_utilization": util, "bureau_score": score, "bureau_inquiries_6m": inq,
        "loan_amount": loan, "loan_term_months": term, "region": region, "channel": channel,
        "bureau_source": "kestrel_v1", "label": label,
    })


def _fraud_rows(rng: np.random.Generator, n: int, hour: np.ndarray, program: str) -> pd.DataFrame:
    if program == "travelplus":
        mcc = rng.choice(["travel", "restaurants", "online_retail"], n, p=[0.7, 0.15, 0.15])
    else:
        mcc = rng.choice(MCC, n, p=MCC_P)
    med = np.array([MCC_AMOUNT[m] for m in mcc])
    amount = np.clip(rng.lognormal(np.log(med), 0.8), 1, 20000).round(2)
    card_present = (rng.random(n) < np.array([MCC_CARD_PRESENT[m] for m in mcc])).astype(float)
    distance = rng.lognormal(np.log(6), 1.0, n) * np.where(mcc == "travel", 15, 1)
    intl = (rng.random(n) < np.where(mcc == "travel", 0.3, 0.02)).astype(float)
    velocity = rng.poisson(0.5, n).astype(float)
    acct_age = np.clip(rng.lognormal(np.log(1100), 0.9, n), 5, 9000).round()
    device = rng.beta(1.4, 9, n).round(3)
    card_program = rng.choice(["standard", "premium"], n, p=[0.85, 0.15])
    night = ((hour <= 5) | (hour >= 23)).astype(float)

    if program == "travelplus":
        # Legitimate travellers on brand-new co-brand cards: large, far-away, cross-border.
        amount = np.clip(rng.lognormal(np.log(330), 0.7, n), 20, 20000).round(2)
        distance = rng.lognormal(np.log(900), 0.9, n)
        intl = (rng.random(n) < 0.4).astype(float)
        acct_age = rng.integers(20, 150, n).astype(float)
        card_present = (rng.random(n) < 0.55).astype(float)
        card_program = np.full(n, "travelplus")
        logit = -6.4 + 0.2 * np.log(amount / 60) + 3.0 * (device - 0.13)
    else:
        logit = (
            -6.6
            + 0.8 * np.log(amount / 60)
            + 1.6 * intl
            + 0.85 * velocity
            + 10.0 * (device - 0.13)
            + 0.9 * night
            - 0.5 * np.log(acct_age / 1100)
            + 1.2 * np.isin(mcc, ["electronics", "digital_goods"])
            + 0.3 * np.log(distance / 6)
            - 1.0 * card_present
        )
    label = (rng.random(n) < sigmoid(logit)).astype(int)
    channel = np.where(card_present == 1, "pos", rng.choice(["ecommerce", "wallet"], n, p=[0.7, 0.3]))
    return pd.DataFrame({
        "amount": amount, "merchant_category": mcc, "hour_of_day": hour.astype(float),
        "distance_from_home_km": distance.round(1), "is_international": intl, "card_present": card_present,
        "txn_velocity_1h": velocity, "account_age_days": acct_age, "device_risk_score": device,
        "card_program": card_program, "channel": channel, "label": label,
    })


def fraud_reference(rng: np.random.Generator, n: int) -> pd.DataFrame:
    diurnal = 1 + 0.6 * np.cos(2 * np.pi * (np.arange(24) - 17) / 24)
    hour = rng.choice(24, n, p=diurnal / diurnal.sum())
    return _fraud_rows(rng, n, hour, "standard")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def _pipeline(features: list[dict]) -> Pipeline:
    num = [f["name"] for f in features if f["kind"] == "num"]
    cat = [f["name"] for f in features if f["kind"] == "cat"]
    pre = ColumnTransformer([
        # Median imputation is the silent default in many production pipelines — and part of
        # why the bureau-score outage (scenario A) looks healthy from the serving side.
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat),
    ])
    clf = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.07, max_leaf_nodes=24,
                                         l2_regularization=1.0, random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def train(features: list[dict], ref: pd.DataFrame) -> Pipeline:
    cols = [f["name"] for f in features]
    model = _pipeline(features)
    model.fit(ref[cols], ref["label"])
    return model


def score(model: Pipeline, features: list[dict], df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[[f["name"] for f in features]])[:, 1]


# --------------------------------------------------------------------------------------
# Production traffic
# --------------------------------------------------------------------------------------


def simulate_credit(rng: np.random.Generator, tl: Timeline, model: Pipeline, threshold: float) -> pd.DataFrame:
    ts = _arrivals(rng, tl, per_hour=80, weekend=0.7, peak_hour=15)
    df = credit_population(rng, len(ts))
    df.insert(0, "ts", pd.to_datetime(ts))
    df.insert(0, "request_id", [f"APP-{i:07d}" for i in rng.permutation(len(df)) + 1_000_000])

    # Scenario A — client-side bureau routing change; the v2 payload renames the score field.
    after = (df["ts"] >= tl.onset_bureau).to_numpy()
    migrated = after & ((df["region"] == "West").to_numpy()
                        | ((df["region"] == "Midwest").to_numpy() & (rng.random(len(df)) < 0.3)))
    df.loc[migrated, "bureau_source"] = "kestrel_v2"
    df.loc[migrated, ["bureau_score", "bureau_inquiries_6m"]] = np.nan

    df["score"] = score(model, CREDIT_FEATURES, df).round(4)
    df["decision"] = (df["score"] < threshold).astype(int)  # 1 = approved
    df["latency_ms"] = rng.lognormal(np.log(55), 0.3, len(df)).round(1)
    df["status"] = "ok"
    df["model_version"] = "3.4.1"
    _mask_immature_labels(df, tl, lag=timedelta(days=14), jitter_hours=0, rng=rng)
    return df


def simulate_fraud(rng: np.random.Generator, tl: Timeline, model: Pipeline, threshold: float) -> pd.DataFrame:
    base_ts = _arrivals(rng, tl, per_hour=130, weekend=1.1, peak_hour=17)
    base = _fraud_rows(rng, len(base_ts), pd.DatetimeIndex(base_ts).hour.to_numpy(), "standard")
    base.insert(0, "ts", pd.to_datetime(base_ts))

    # Scenario C — TravelPlus co-brand launch ramps up to ~26% extra traffic over 36h.
    onset = np.datetime64(tl.onset_travelplus)

    def ramp(hours: pd.DatetimeIndex) -> np.ndarray:
        elapsed = (hours.to_numpy() - onset) / np.timedelta64(1, "h")
        return np.clip(elapsed / tl.travelplus_ramp_hours, 0, 1) * 0.26

    tp_ts = _arrivals(rng, tl, per_hour=130, weekend=1.1, peak_hour=17, multiplier=ramp)
    tp = _fraud_rows(rng, len(tp_ts), pd.DatetimeIndex(tp_ts).hour.to_numpy(), "travelplus")
    tp.insert(0, "ts", pd.to_datetime(tp_ts))

    df = pd.concat([base, tp], ignore_index=True).sort_values("ts", ignore_index=True)
    df.insert(0, "request_id", [f"TX-{i:08d}" for i in rng.permutation(len(df)) + 10_000_000])
    df["score"] = score(model, FRAUD_FEATURES, df).round(4)

    # Scenario B — release 7.3.0 with a cache-less feature-store namespace.
    ts = df["ts"].to_numpy()
    after_deploy = ts >= np.datetime64(tl.onset_deploy)
    blip = (ts >= np.datetime64(tl.blip_start)) & (ts < np.datetime64(tl.blip_start + timedelta(hours=2)))
    n = len(df)
    latency = rng.lognormal(np.log(38), 0.35, n)
    latency[after_deploy] = rng.lognormal(np.log(120), 0.6, after_deploy.sum())
    latency[blip] = rng.lognormal(np.log(150), 0.45, blip.sum())
    df["latency_ms"] = latency.round(1)
    df["model_version"] = np.where(after_deploy, "7.3.0", "7.2.0")
    timed_out = latency > 300  # API gateway budget; fallback policy = ALLOW (unscored)
    df["status"] = np.where(timed_out, "timeout_fallback", "ok")
    df.loc[timed_out, "score"] = np.nan
    df["decision"] = (df["score"] >= threshold).astype(int)  # 1 = blocked; NaN >= t is False
    _mask_immature_labels(df, tl, lag=timedelta(hours=48), jitter_hours=24, rng=rng)
    return df


def _mask_immature_labels(df: pd.DataFrame, tl: Timeline, lag: timedelta, jitter_hours: int,
                          rng: np.random.Generator) -> None:
    jitter = rng.integers(0, jitter_hours * 3600 + 1, len(df)).astype("timedelta64[s]")
    available = df["ts"].to_numpy() + np.timedelta64(int(lag.total_seconds()), "s") + jitter
    df["label_available_at"] = pd.to_datetime(available)
    matured = available <= np.datetime64(tl.end)
    df["label"] = np.where(matured, df["label"].astype(float), np.nan)


# --------------------------------------------------------------------------------------
# Logs & change events
# --------------------------------------------------------------------------------------


def generate_logs(rng: np.random.Generator, tl: Timeline, credit: pd.DataFrame, fraud: pd.DataFrame) -> list[dict]:
    logs: list[dict] = []

    def add(ts: datetime, model_id: str | None, service: str, level: str, event_type: str, message: str) -> None:
        if tl.start <= ts < tl.end:
            logs.append({"ts": ts, "model_id": model_id, "service": service, "level": level,
                         "event_type": event_type, "message": message})

    # --- background noise ------------------------------------------------------------
    day = tl.start.replace(hour=0)
    while day < tl.end:
        add(day + timedelta(hours=1, minutes=30), "credit-risk", "batch-scoring", "INFO", "info",
            f"Nightly portfolio re-score completed (n={rng.integers(38000, 42000):,})")
        add(day + timedelta(hours=3, minutes=10), "fraud-detect", "label-ingest", "INFO", "info",
            f"Chargeback/confirmed-fraud labels ingested (n={rng.integers(90, 160)})")
        for h in (0, 6, 12, 18):
            add(day + timedelta(hours=h, minutes=int(rng.integers(0, 10))), "credit-risk", "feature-pipeline",
                "INFO", "info", f"Feature freshness check passed (max lag {rng.integers(1, 6)}m)")
            add(day + timedelta(hours=h, minutes=int(rng.integers(0, 10))), "fraud-detect", "model-serving",
                "INFO", "info", f"Health check OK (replicas={rng.integers(5, 7)}, p50={rng.integers(34, 42)}ms)")
        if rng.random() < 0.6:
            add(day + timedelta(hours=int(rng.integers(8, 20)), minutes=int(rng.integers(0, 60))), None,
                "reporting-db", "WARN", "error", f"Slow query on reporting replica ({rng.uniform(1.5, 4):.1f}s)")
        if rng.random() < 0.3:
            add(day + timedelta(hours=int(rng.integers(0, 24))), "fraud-detect", "model-serving", "INFO", "info",
                "Autoscaler adjusted replicas 6 -> 5 (cpu 31%)")
        day += timedelta(days=1)

    # Benign change events, far from any incident onset.
    add(tl.start + timedelta(days=2, hours=10), "fraud-detect", "deploy-bot", "INFO", "change",
        "fraud-detect 7.2.0 promoted to 100% (quarterly retrain, approved by model risk)")
    add(tl.credit_redeploy, "credit-risk", "deploy-bot", "INFO", "change",
        "credit-risk 3.4.1 redeployed (base image security patch, no model or config change)")
    add(tl.start + timedelta(days=15, hours=16), None, "config-service", "INFO", "change",
        "Alert routing updated: #harborline-ml-oncall now receives SEV2+")

    # --- Scenario A: bureau routing change -------------------------------------------
    a = tl.onset_bureau
    add(a - timedelta(minutes=14), "credit-risk", "client-change-feed", "INFO", "change",
        "CHG-8812 (Harborline IT): route West + 30% of Midwest bureau pulls to Kestrel Bureau API v2")
    add(a - timedelta(minutes=2), "credit-risk", "bureau-connector", "INFO", "change",
        "Kestrel API v2 endpoint enabled for routing group WEST_MIDWEST_V2 (response schema 2.0)")
    miss = credit[credit["bureau_score"].isna()].groupby(credit["ts"].dt.floor("h")).size()
    total = credit.groupby(credit["ts"].dt.floor("h")).size()
    for hour, n in miss.items():
        add(hour.to_pydatetime() + timedelta(minutes=1), "credit-risk", "bureau-connector", "WARN", "error",
            f"Field 'bureau_score' not found in Kestrel v2 response (found 'riskScoreV2'); emitted null for {n} requests")
        rate = n / total.get(hour, n)
        if rate > 0.05:
            add(hour.to_pydatetime() + timedelta(minutes=5), "credit-risk", "feature-pipeline", "WARN", "error",
                f"Null rate for bureau_score at {rate:.1%} (soft limit 5%); imputing training median")

    # --- Scenario B: release 7.3.0 ---------------------------------------------------
    b = tl.onset_deploy
    add(b - timedelta(minutes=6), "fraud-detect", "deploy-bot", "INFO", "change",
        "fraud-detect 7.3.0 rolled out to 100% (adds 12 real-time velocity features from feature-store namespace fraud_velocity_v2)")
    add(b + timedelta(minutes=3), "fraud-detect", "feature-store", "WARN", "error",
        "Cache hit ratio for namespace fraud_velocity_v2 dropped to 31% (baseline 97%); ttl_seconds=0")
    add(b + timedelta(minutes=40), "fraud-detect", "model-serving", "INFO", "info",
        "Autoscaler adjusted replicas 6 -> 10 (cpu 81%)")
    to = fraud[fraud["status"] != "ok"]
    to_hour = to.groupby(to["ts"].dt.floor("h")).size()
    for hour, n in to_hour.items():
        if hour >= pd.Timestamp(b):
            add(hour.to_pydatetime() + timedelta(minutes=int(rng.integers(0, 50))), "fraud-detect", "api-gateway",
                "ERROR", "error", f"fraud-detect upstream timeout after 300ms; fallback policy ALLOW applied (n={n})")
            if hour.hour % 2 == 0:
                add(hour.to_pydatetime() + timedelta(minutes=17), "fraud-detect", "feature-store", "WARN", "error",
                    f"p99 read latency {rng.integers(340, 460)}ms on namespace fraud_velocity_v2 (cache miss)")

    # --- Scenario C: TravelPlus launch -----------------------------------------------
    c = tl.onset_travelplus
    add(c - timedelta(hours=3), "fraud-detect", "client-change-feed", "INFO", "business",
        "Harborline marketing: TravelPlus co-brand travel card launched — 40k new cardholders activated, travel booking promo live")

    # --- Noise: network blip ---------------------------------------------------------
    add(tl.blip_start - timedelta(minutes=5), None, "cloud-status", "WARN", "error",
        "Provider maintenance in zone us-east-1b: degraded network throughput expected for ~2h")
    for hour, n in to_hour.items():
        if tl.blip_start <= hour.to_pydatetime() < tl.blip_start + timedelta(hours=2):
            add(hour.to_pydatetime() + timedelta(minutes=20), "fraud-detect", "api-gateway", "ERROR", "error",
                f"fraud-detect upstream timeout after 300ms; fallback policy ALLOW applied (n={n})")
    add(tl.blip_start + timedelta(hours=2, minutes=10), None, "cloud-status", "INFO", "info",
        "Provider maintenance in zone us-east-1b completed")

    logs.sort(key=lambda r: r["ts"])
    return logs
