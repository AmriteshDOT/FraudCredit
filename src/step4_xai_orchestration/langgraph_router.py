from typing import TypedDict
from langgraph.graph import END, START, StateGraph

from .collections_cmdp import ModelBasedCollectionsMDP
from .compliance_schemas import AdverseActionNotice, CollectionActionPayload
from .reason_code_mapper import map_shap_to_statutory_codes
from .vector_policy_store import RegulatoryPolicyStore


class ComplianceState(TypedDict, total=False):
    workflow_type: str  # "FRAUD_ADVERSE_ACTION" | "CREDIT_COLLECTION"
    account_id: str
    fraud_prob: float
    phi_0: float
    top_drivers: list[tuple[str, float]]
    dpd_bucket: int
    survival_prob_30d: float
    contacts_used: int
    in_legal_hours: bool
    mapped_reasons: list[dict]
    retrieved_policies: list[dict[str, str]]
    cmdp_action: str
    validated_json: str


def build_compliance_orchestrator(
    policy_store: RegulatoryPolicyStore,
    cmdp_solver: ModelBasedCollectionsMDP,
):
    """
    Compiles a deterministic LangGraph StateGraph where AI/LLM has zero decision power
    and every output passes rigid Pydantic FSM schema validation.
    """

    def route_branch(state: ComplianceState) -> str:
        return (
            "fraud_shap_node"
            if state["workflow_type"] == "FRAUD_ADVERSE_ACTION"
            else "collection_cmdp_node"
        )

    def fraud_shap_node(state: ComplianceState) -> ComplianceState:
        reasons = map_shap_to_statutory_codes(state["top_drivers"])
        if not reasons:
            reasons = [
                {
                    "reason_code": "RC_99_COMPOSITE_RISK",
                    "feature": "composite_score",
                    "shap_contribution": 0.01,
                    "statutory_description": "Elevated composite risk threshold exceeded.",
                }
            ]
        query = " ".join(r["statutory_description"] for r in reasons)
        policies = policy_store.retrieve_clauses(query, n_results=2)
        return {"mapped_reasons": reasons, "retrieved_policies": policies}

    def collection_cmdp_node(state: ComplianceState) -> ComplianceState:
        action = cmdp_solver.best_compliant_action(
            dpd_bucket=state["dpd_bucket"],
            contacts_used=state["contacts_used"],
            in_legal_hours=state["in_legal_hours"],
        )
        policies = policy_store.retrieve_clauses(
            f"Collection outreach {action} RBI hours", n_results=2
        )
        return {"cmdp_action": action, "retrieved_policies": policies}

    def schema_formatter_node(state: ComplianceState) -> ComplianceState:
        policy_ids = [p["policy_id"] for p in state["retrieved_policies"]]
        if state["workflow_type"] == "FRAUD_ADVERSE_ACTION":
            decision = (
                "BLOCK_AND_FREEZE" if state["fraud_prob"] >= 0.80 else "STEP_UP_2FA"
            )
            payload = AdverseActionNotice(
                account_id=state["account_id"],
                decision=decision,
                calibrated_fraud_prob=state["fraud_prob"],
                shap_base_value=state["phi_0"],
                statutory_reasons=state["mapped_reasons"],
                cited_policy_ids=policy_ids,
            )
            return {"validated_json": payload.model_dump_json(indent=2)}

        act = state["cmdp_action"]
        scripts = {
            "NO_ACTION_WAIT": "Suppressed outreach: outside RBI calling hours or weekly cap reached.",
            "WHATSAPP_SOFT_NUDGE": "Reminder: Your Kotak EMI is due. Reply HELP to connect with support.",
            "IVR_PAYMENT_REMINDER": "Automated reminder regarding your upcoming EMI installment.",
            "RESTRUCTURE_EMI_OFFER": "You are eligible for a 3-month EMI tenure restructuring option.",
            "HUMAN_AGENT_ESCALATE": "Routing case to licensed relationship officer for assisted resolution.",
        }
        payload = CollectionActionPayload(
            account_id=state["account_id"],
            selected_action=act,
            survival_prob_30d=state["survival_prob_30d"],
            contacts_used_this_week=state["contacts_used"],
            within_rbi_calling_window=state["in_legal_hours"],
            approved_script_template=scripts[act],
            cited_policy_ids=policy_ids,
        )
        return {"validated_json": payload.model_dump_json(indent=2)}

    graph = StateGraph(ComplianceState)
    graph.add_node("fraud_shap_node", fraud_shap_node)
    graph.add_node("collection_cmdp_node", collection_cmdp_node)
    graph.add_node("schema_formatter_node", schema_formatter_node)

    graph.add_conditional_edges(START, route_branch)
    graph.add_edge("fraud_shap_node", "schema_formatter_node")
    graph.add_edge("collection_cmdp_node", "schema_formatter_node")
    graph.add_edge("schema_formatter_node", END)

    return graph.compile()
