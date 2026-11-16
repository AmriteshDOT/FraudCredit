from typing import Literal
from pydantic import BaseModel, Field


class StatutoryReasonItem(BaseModel):
    reason_code: Literal[
        "RC_01_VELOCITY_SPIKE",
        "RC_02_BURST_ANOMALY",
        "RC_03_GRAPH_CONTAGION",
        "RC_04_SHARED_INFRASTRUCTURE",
        "RC_05_TOPOLOGICAL_LAYERING",
        "RC_99_COMPOSITE_RISK",
    ]
    feature: str
    shap_contribution: float = Field(gt=0.0)
    statutory_description: str


class AdverseActionNotice(BaseModel):
    """Rigid FSM-enforced JSON schema preventing hallucinated denial/freeze explanations."""

    account_id: str
    decision: Literal["BLOCK_AND_FREEZE", "STEP_UP_2FA", "ALLOW"]
    calibrated_fraud_prob: float = Field(ge=0.0, le=1.0)
    shap_base_value: float
    statutory_reasons: list[StatutoryReasonItem] = Field(min_length=1)
    cited_policy_ids: list[str] = Field(min_length=1)
    ai_disclosure_banner: Literal[
        "Automated risk assessment subject to human review under RBI Model Risk Governance."
    ] = "Automated risk assessment subject to human review under RBI Model Risk Governance."
    human_escalation_available: Literal[True] = True


class CollectionActionPayload(BaseModel):
    """Rigid FSM-enforced JSON schema for compliant debt collection orchestration."""

    account_id: str
    selected_action: Literal[
        "NO_ACTION_WAIT",
        "WHATSAPP_SOFT_NUDGE",
        "IVR_PAYMENT_REMINDER",
        "RESTRUCTURE_EMI_OFFER",
        "HUMAN_AGENT_ESCALATE",
    ]
    survival_prob_30d: float = Field(ge=0.0, le=1.0)
    contacts_used_this_week: int = Field(ge=0, le=3)
    within_rbi_calling_window: bool
    approved_script_template: str
    cited_policy_ids: list[str] = Field(min_length=1)
