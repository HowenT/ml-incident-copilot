"""Root-cause taxonomy shared by diagnosis, actions, summaries and the knowledge base."""

ROOT_CAUSES: dict[str, dict[str, str]] = {
    "upstream_data_change": {
        "label": "Upstream data / schema change",
        "plain": "a change in the data your systems send to the model",
    },
    "population_drift": {
        "label": "Input population drift",
        "plain": "a new kind of customer or transaction that the model has not seen before",
    },
    "model_release_regression": {
        "label": "Model release regression",
        "plain": "a problem introduced by the most recent model release",
    },
    "serving_infrastructure": {
        "label": "Serving infrastructure",
        "plain": "slowness in the systems that run the model",
    },
    "config_threshold_change": {
        "label": "Decision threshold / config change",
        "plain": "a change to the decision settings applied to model scores",
    },
    "concept_drift": {
        "label": "Concept drift",
        "plain": "a shift in how customer behaviour relates to outcomes",
    },
    "other": {"label": "Other", "plain": "another cause"},
}

SEVERITIES = ["SEV1", "SEV2", "SEV3"]
STATUSES = ["open", "investigating", "mitigated", "resolved"]


def label(category: str | None) -> str:
    return ROOT_CAUSES.get(category or "other", ROOT_CAUSES["other"])["label"]
