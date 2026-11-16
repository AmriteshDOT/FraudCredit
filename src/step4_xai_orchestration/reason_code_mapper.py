STATUTORY_REASON_CATALOG: dict[str, dict[str, str]] = {
    "vel_amt_1h": {
        "code": "RC_01_VELOCITY_SPIKE",
        "legal_text": "Abnormal 1-hour cumulative transaction velocity exceeding historical baseline.",
    },
    "vel_zscore_1h": {
        "code": "RC_02_BURST_ANOMALY",
        "legal_text": "Transaction amount deviates significantly from account temporal spend distribution.",
    },
    "ppr_risk": {
        "code": "RC_03_GRAPH_CONTAGION",
        "legal_text": "Direct or 2-hop transactional proximity to verified high-risk entities.",
    },
    "shared_device_cnt": {
        "code": "RC_04_SHARED_INFRASTRUCTURE",
        "legal_text": "Multiple distinct accounts operating concurrently from identical device/IP identifiers.",
    },
    "gnn_emb_0": {
        "code": "RC_05_TOPOLOGICAL_LAYERING",
        "legal_text": "Multi-hop fund routing topology consistent with rapid layering patterns.",
    },
}


def map_shap_to_statutory_codes(
    top_drivers: list[tuple[str, float]],
    min_positive_phi: float = 0.05,
) -> list[dict[str, object]]:
    """
    Deterministically maps positive risk-increasing SHAP attributions (phi_i > 0)
    to audited RBI statutory adverse-action codes—zero LLM involvement.
    """
    mapped = []
    for feat_name, phi_val in top_drivers:
        if phi_val <= min_positive_phi:
            continue
        entry = STATUTORY_REASON_CATALOG.get(
            feat_name,
            {
                "code": "RC_99_COMPOSITE_RISK",
                "legal_text": f"Elevated composite risk indicator on feature '{feat_name}'.",
            },
        )
        mapped.append(
            {
                "reason_code": entry["code"],
                "feature": feat_name,
                "shap_contribution": round(float(phi_val), 4),
                "statutory_description": entry["legal_text"],
            }
        )
    return mapped
