"""Client glossary: how to say technical identifiers in the client's own words.

FDEs keep one of these per account; it keeps customer updates free of column names.
"""
PLAIN = {
    # features
    "bureau_score": "credit bureau score",
    "bureau_inquiries_6m": "number of recent credit checks",
    "annual_income": "stated income",
    "debt_to_income": "debt-to-income ratio",
    "credit_utilization": "credit card utilisation",
    "loan_amount": "loan amount",
    "amount": "transaction amount",
    "merchant_category": "merchant type",
    "distance_from_home_km": "distance from home",
    "is_international": "cross-border flag",
    "account_age_days": "account age",
    "device_risk_score": "device risk",
    "txn_velocity_1h": "recent transaction count",
    # segments / values
    "bureau_source": "bureau connection",
    "kestrel_v2": "the new Kestrel Bureau API (v2) connection",
    "kestrel_v1": "the existing Kestrel Bureau connection",
    "card_program": "card programme",
    "travelplus": "TravelPlus card",
    "region": "region",
    "West": "West region",
    "Midwest": "Midwest region",
    # models
    "credit-risk": "credit decisioning model",
    "fraud-detect": "card fraud model",
}


def plain(term: str | None) -> str:
    if term is None:
        return ""
    return PLAIN.get(term, term.replace("_", " "))
