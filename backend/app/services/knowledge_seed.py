"""Past incidents that bootstrap the knowledge base (resolved before the simulated month).

They carry a diagnosis signature, the diagnosed category and the human-confirmed root
cause, so similarity search and diagnosis priors work from day one. Two of them were
mis- or partially-diagnosed — the feedback loop has something to learn from.
"""
from datetime import datetime

HISTORICAL_INCIDENTS = [
    {
        "code": "INC-2026-0107", "model_id": "credit-risk", "severity": "SEV2",
        "title": "Missing employer income after payroll-verification vendor API upgrade",
        "created_at": datetime(2026, 4, 14, 9, 20),
        "customer_impact": "Approval rate for salaried applicants up 9pp in two days; risk team paused auto-approvals for 3 hours.",
        "business_context": "Vendor VeriPay announced API v3 two weeks earlier; Harborline IT switched over on Monday morning.",
        "affected_segments": "Applications with payroll verification (≈22% of volume)",
        "diagnosed_category": "upstream_data_change", "diagnosed_confidence": 0.84,
        "signature": {"missing_spike": 0.9, "missing_segment_concentration": 0.8, "data_source_change_near_onset": 0.7,
                      "data_errors": 0.8, "prediction_shift": 0.4, "missing_drives_decision": 0.7, "model_credit": 1.0},
        "resolution": {
            "root_cause_category": "upstream_data_change",
            "root_cause_detail": "VeriPay API v3 moved annual income to `income.gross_annual`; the connector mapping still read `annual_income`, emitting nulls that the pipeline imputed with the training median.",
            "fix_applied": "Patched connector field mapping; re-scored 3,140 affected applications; 212 approvals routed to manual review.",
            "prevention": "Contract test on vendor payload schema in CI; per-source missing-rate alert at 5%.",
            "diagnosis_feedback": "correct", "time_to_mitigate_min": 190,
            "tags": ["schema-change", "vendor-api", "silent-imputation"], "resolved_by": "Maya Chen (FDE)",
        },
    },
    {
        "code": "INC-2025-0931", "model_id": "fraud-detect", "severity": "SEV2",
        "title": "Block-rate spike on e-commerce electronics during Black Friday weekend",
        "created_at": datetime(2025, 11, 28, 14, 5),
        "customer_impact": "False declines on high-value online electronics purchases; ~1,900 cardholder complaints over the weekend.",
        "business_context": "Black Friday promotions; basket sizes 2-3x normal for electronics merchants.",
        "affected_segments": "Online electronics merchants, high-ticket baskets",
        "diagnosed_category": "population_drift", "diagnosed_confidence": 0.78,
        "signature": {"feature_drift": 0.8, "drift_segment_concentration": 0.7, "business_event_near_onset": 0.6,
                      "prediction_shift": 0.7, "performance_drop": 0.5, "performance_segment_concentration": 0.6,
                      "no_change_events": 0.6, "model_fraud": 1.0},
        "resolution": {
            "root_cause_category": "population_drift",
            "root_cause_detail": "Seasonal basket-size shift moved legitimate electronics purchases into the high-amount region the model associates with fraud.",
            "fix_applied": "Temporary +0.05 block threshold for online electronics during the promo window; step-up authentication instead of hard declines.",
            "prevention": "Seasonal calendar feeds a pre-emptive threshold review; retrained with holiday data in January.",
            "diagnosis_feedback": "correct", "time_to_mitigate_min": 240,
            "tags": ["seasonality", "promotion", "false-declines"], "resolved_by": "Sam Ortiz (FDE)",
        },
    },
    {
        "code": "INC-2026-0152", "model_id": "fraud-detect", "severity": "SEV2",
        "title": "p99 latency breach after feature-store node pool migration",
        "created_at": datetime(2026, 5, 20, 22, 40),
        "customer_impact": "Checkout latency complaints from two large merchants; no fallback decisions.",
        "business_context": "Platform team migrated feature-store nodes to a new instance family overnight.",
        "affected_segments": "All traffic",
        "diagnosed_category": "serving_infrastructure", "diagnosed_confidence": 0.81,
        "signature": {"latency_spike": 0.9, "serving_errors": 0.4, "infra_errors": 0.8, "model_fraud": 1.0},
        "resolution": {
            "root_cause_category": "serving_infrastructure",
            "root_cause_detail": "New node pool had a smaller connection limit; feature lookups queued under peak load.",
            "fix_applied": "Rolled back node pool; raised connection pool 32 → 128.",
            "prevention": "Load test gate for infra migrations; synthetic p99 probe on the feature store.",
            "diagnosis_feedback": "correct", "time_to_mitigate_min": 75,
            "tags": ["feature-store", "latency", "infra-migration"], "resolved_by": "Priya Nair (ML Platform)",
        },
    },
    {
        "code": "INC-2026-0164", "model_id": "fraud-detect", "severity": "SEV1",
        "title": "fraud-detect 7.1.0 release raised timeouts on card-not-present traffic",
        "created_at": datetime(2026, 6, 9, 10, 15),
        "customer_impact": "4% of online transactions approved by fallback without scoring for ~40 minutes.",
        "business_context": "Release 7.1.0 added device-graph features.",
        "affected_segments": "Card-not-present transactions",
        "diagnosed_category": "model_release_regression", "diagnosed_confidence": 0.88,
        "signature": {"deploy_near_onset": 0.9, "version_correlated": 0.9, "latency_spike": 0.8, "serving_errors": 0.7,
                      "infra_errors": 0.5, "model_fraud": 1.0},
        "resolution": {
            "root_cause_category": "model_release_regression",
            "root_cause_detail": "N+1 device-graph lookups per request in 7.1.0 feature code.",
            "fix_applied": "Rolled back to 7.0.4 within 40 minutes; batched lookups; re-released behind a canary latency gate.",
            "prevention": "Canary gate on p95 latency and timeout rate before 100% rollout.",
            "diagnosis_feedback": "correct", "time_to_mitigate_min": 40,
            "tags": ["release", "rollback", "fallback-allow"], "resolved_by": "Sam Ortiz (FDE)",
        },
    },
    {
        "code": "INC-2026-0178", "model_id": "credit-risk", "severity": "SEV2",
        "title": "Approval rate drop after risk-appetite threshold pushed to production early",
        "created_at": datetime(2026, 7, 2, 8, 50),
        "customer_impact": "Approval rate fell from 62% to 51% overnight; broker partners escalated.",
        "business_context": "Risk committee approved a tighter cut-off effective next quarter; config was merged early.",
        "affected_segments": "All applications",
        "diagnosed_category": "config_threshold_change", "diagnosed_confidence": 0.9,
        "signature": {"config_change_near_onset": 0.9, "decision_shift_without_score_shift": 0.9, "prediction_shift": 0.2,
                      "model_credit": 1.0},
        "resolution": {
            "root_cause_category": "config_threshold_change",
            "root_cause_detail": "Decision cut-off changed 0.083 → 0.065 via config merge without a release window.",
            "fix_applied": "Reverted config; re-decisioned 1,020 declined applications.",
            "prevention": "Four-eyes approval and effective-date field on decision configs.",
            "diagnosis_feedback": "correct", "time_to_mitigate_min": 55,
            "tags": ["config", "threshold", "change-management"], "resolved_by": "Maya Chen (FDE)",
        },
    },
    {
        "code": "INC-2026-0190", "model_id": "credit-risk", "severity": "SEV3",
        "title": "Nulls in bureau_inquiries_6m after nightly ETL partial failure",
        "created_at": datetime(2026, 7, 21, 6, 30),
        "customer_impact": "Minor score shift; no customer-visible impact reported.",
        "business_context": "Nightly ETL job hit a warehouse quota and published a partial partition.",
        "affected_segments": "Applications scored between 01:00 and 07:00",
        "diagnosed_category": "upstream_data_change", "diagnosed_confidence": 0.7,
        "signature": {"missing_spike": 0.7, "data_errors": 0.6, "missing_drives_decision": 0.2, "model_credit": 1.0},
        "resolution": {
            "root_cause_category": "upstream_data_change",
            "root_cause_detail": "Partial ETL partition published with nulls for bureau_inquiries_6m.",
            "fix_applied": "Re-ran ETL partition and re-scored the 6-hour window.",
            "prevention": "Completeness check blocks publishing partitions with >2% nulls on critical fields.",
            "diagnosis_feedback": "correct", "time_to_mitigate_min": 130,
            "tags": ["etl", "completeness", "silent-imputation"], "resolved_by": "Priya Nair (ML Platform)",
        },
    },
    {
        "code": "INC-2026-0203", "model_id": "credit-risk", "severity": "SEV3",
        "title": "Gradual AUC decay on credit model after consecutive rate hikes",
        "created_at": datetime(2026, 8, 4, 11, 0),
        "customer_impact": "Early-delinquency rate on new vintages 1.3pp above plan.",
        "business_context": "Two central-bank rate hikes in Q2; payment burden rose for variable-rate borrowers.",
        "affected_segments": "Variable-rate borrowers, high DTI",
        "diagnosed_category": "population_drift", "diagnosed_confidence": 0.46,
        "signature": {"performance_drop": 0.7, "feature_drift": 0.3, "no_change_events": 1.0, "gradual_onset": 0.8,
                      "model_credit": 1.0},
        "resolution": {
            "root_cause_category": "concept_drift",
            "root_cause_detail": "Input distributions barely moved; the relationship between DTI and default strengthened under higher rates.",
            "fix_applied": "Retrained on 2025-H2/2026-H1 vintages with rate-sensitivity features.",
            "prevention": "Quarterly challenger retrain; macro-scenario backtests.",
            "diagnosis_feedback": "incorrect", "time_to_mitigate_min": None,
            "tags": ["concept-drift", "macro", "retrain"], "resolved_by": "Maya Chen (FDE)",
        },
    },
    {
        "code": "INC-2026-0215", "model_id": "fraud-detect", "severity": "SEV2",
        "title": "Gateway timeouts during API gateway upgrade caused fallback approvals",
        "created_at": datetime(2026, 8, 27, 19, 45),
        "customer_impact": "2.1% of transactions approved without scoring for 70 minutes.",
        "business_context": "Platform upgraded the API gateway 2.13 → 2.14 during peak hours.",
        "affected_segments": "All traffic",
        "diagnosed_category": "model_release_regression", "diagnosed_confidence": 0.55,
        "signature": {"latency_spike": 0.7, "serving_errors": 0.9, "deploy_near_onset": 0.6, "infra_errors": 0.8,
                      "model_fraud": 1.0},
        "resolution": {
            "root_cause_category": "serving_infrastructure",
            "root_cause_detail": "Gateway 2.14 lowered the upstream keep-alive pool; the model itself was unchanged.",
            "fix_applied": "Rolled back gateway to 2.13; fallback for high-risk merchant categories changed from ALLOW to CHALLENGE.",
            "prevention": "Gateway changes join the model change calendar; fallback policy reviewed quarterly.",
            "diagnosis_feedback": "partially_correct", "time_to_mitigate_min": 65,
            "tags": ["gateway", "fallback-allow", "change-calendar"], "resolved_by": "Priya Nair (ML Platform)",
        },
    },
]

# Pre-filled customer reports for the live demo ("Use sample customer report" button).
SAMPLE_REPORTS = {
    "credit-bureau": {
        "title": "Unexpected approval-rate jump in West region",
        "severity": "SEV2",
        "customer_impact": "Harborline's credit risk team sees West-region auto-approvals up ~15pp since early Thursday. "
                           "They are worried riskier applicants are being approved and are considering pausing auto-approvals.",
        "business_context": "No intentional policy change on Harborline's side. Their IT team has been migrating some "
                            "integrations this week. Month-end lending volume is ramping up.",
        "affected_segments": "West region applications (possibly part of Midwest)",
        "reported_by": "Alex Rivera (FDE on-call)",
        "customer_contact": "Dana Park, Head of Credit Risk, Harborline",
    },
    "fraud-latency": {
        "title": "Checkout latency and unscored approvals on card transactions",
        "severity": "SEV1",
        "customer_impact": "Two large merchants report slow card authorisations since yesterday morning. Harborline "
                           "fraud ops noticed transactions being approved with no fraud score attached.",
        "business_context": "Our ML platform team shipped fraud-detect 7.3.0 yesterday. Harborline is in peak season for "
                            "e-commerce and any unscored approvals are a direct fraud-loss exposure.",
        "affected_segments": "All card traffic; merchants with tight authorisation timeouts",
        "reported_by": "Alex Rivera (FDE on-call)",
        "customer_contact": "Jordan Lee, Head of Card Fraud, Harborline",
    },
    "fraud-travelplus": {
        "title": "Travel purchases declined for new TravelPlus cardholders",
        "severity": "SEV2",
        "customer_impact": "Harborline support logged ~180 complaints in 24h from new TravelPlus cardholders whose "
                           "flight and hotel purchases abroad were declined. The co-brand partner has escalated.",
        "business_context": "TravelPlus co-brand travel card launched last week with a travel-booking promotion. "
                            "The ML team was not consulted before launch.",
        "affected_segments": "TravelPlus cardholders; travel merchants; cross-border transactions",
        "reported_by": "Maya Chen (FDE)",
        "customer_contact": "Jordan Lee, Head of Card Fraud, Harborline",
    },
}


def sample_report_key(model_id: str, rules: set[str]) -> str | None:
    if model_id == "credit-risk":
        return "credit-bureau"
    if model_id == "fraud-detect":
        if rules & {"latency_slo", "serving_errors"}:
            return "fraud-latency"
        return "fraud-travelplus"
    return None
