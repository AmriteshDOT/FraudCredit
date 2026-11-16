import numpy as np
import xgboost as xgb

from .collections_cmdp import ModelBasedCollectionsMDP, ModelFreeCollectionsQLearning
from .langgraph_router import build_compliance_orchestrator
from .shap_explainer import HybridSHAPExplainer
from .vector_policy_store import RegulatoryPolicyStore


def run_step4_pipeline() -> None:
    rng = np.random.default_rng(42)
    feature_names = [
        "vel_amt_1h",
        "vel_zscore_1h",
        "ppr_risk",
        "shared_device_cnt",
        "gnn_emb_0",
        "gnn_emb_1",
    ]
    x_train = rng.normal(size=(200, len(feature_names))).astype(np.float32)
    y_train = (x_train[:, 0] + 1.5 * x_train[:, 2] + x_train[:, 3] > 2.2).astype(int)

    clf = xgb.XGBClassifier(n_estimators=40, max_depth=3, random_state=42)
    clf.fit(x_train, y_train)

    # 1. Exact TreeSHAP local attribution
    xai = HybridSHAPExplainer(clf, feature_names)
    flagged_row = np.array([3.1, 2.5, 2.8, 3.0, 1.4, 0.2], dtype=np.float32)
    shap_out = xai.explain_instance(flagged_row)
    raw_margin = float(clf.predict(flagged_row.reshape(1, -1), output_margin=True)[0])
    assert abs(shap_out["local_accuracy_sum"] - raw_margin) < 1e-4
    print(
        f"[TreeSHAP] Local Accuracy Verified: sum(phi)+phi_0 = {shap_out['local_accuracy_sum']:.4f} == logit {raw_margin:.4f}"
    )

    # 2. Solve Model-Based CMDP (S,A,P,R,V) & Update Model-Free Q-Learner (s,a,r,s',Q)
    cmdp = ModelBasedCollectionsMDP()
    cmdp.solve_value_iteration()
    q_coll = ModelFreeCollectionsQLearning()
    act_q = q_coll.select_action(dpd_bucket=1, contacts_used=1, in_legal_hours=True)
    q_coll.update((1, 1), act_q, reward=115.0, s_next_raw=(0, 2))

    # 3. Execute Deterministic LangGraph + ChromaDB + Pydantic Compliance Router
    policy_store = RegulatoryPolicyStore()
    orchestrator = build_compliance_orchestrator(policy_store, cmdp)

    fraud_res = orchestrator.invoke(
        {
            "workflow_type": "FRAUD_ADVERSE_ACTION",
            "account_id": "KOTAK_ACCT_8821",
            "fraud_prob": 0.91,
            "phi_0": shap_out["phi_0"],
            "top_drivers": shap_out["top_drivers"],
        }
    )
    print(f"[Adverse Action JSON]\n{fraud_res['validated_json']}")

    coll_res = orchestrator.invoke(
        {
            "workflow_type": "CREDIT_COLLECTION",
            "account_id": "KOTAK_LOAN_4410",
            "dpd_bucket": 1,
            "survival_prob_30d": 0.58,
            "contacts_used": 1,
            "in_legal_hours": True,
        }
    )
    print(f"[Collections Strategy JSON]\n{coll_res['validated_json']}")


if __name__ == "__main__":
    run_step4_pipeline()
