import json
import numpy as np
import xgboost as xgb

from src.step4_xai_orchestration.collections_cmdp import ModelBasedCollectionsMDP
from src.step4_xai_orchestration.langgraph_router import build_compliance_orchestrator
from src.step4_xai_orchestration.shap_explainer import HybridSHAPExplainer
from src.step4_xai_orchestration.vector_policy_store import RegulatoryPolicyStore


def test_treeshap_local_accuracy_invariant():
    rng = np.random.default_rng(0)
    names = ["vel_amt_1h", "ppr_risk", "shared_device_cnt"]
    x = rng.normal(size=(80, 3)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 1.0).astype(int)

    clf = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=0).fit(x, y)
    explainer = HybridSHAPExplainer(clf, names)
    res = explainer.explain_instance(x[0])
    logit = float(clf.predict(x[0:1], output_margin=True)[0])
    assert abs(res["local_accuracy_sum"] - logit) < 1e-4


def test_cmdp_enforces_rbi_calling_hours_and_contact_caps():
    cmdp = ModelBasedCollectionsMDP()
    cmdp.solve_value_iteration()
    # Outside 08:00-19:00 IST or >= 3 contacts/week must force NO_ACTION_WAIT
    assert (
        cmdp.best_compliant_action(dpd_bucket=2, contacts_used=0, in_legal_hours=False)
        == "NO_ACTION_WAIT"
    )
    assert (
        cmdp.best_compliant_action(dpd_bucket=2, contacts_used=3, in_legal_hours=True)
        == "NO_ACTION_WAIT"
    )


def test_langgraph_router_produces_schema_valid_json():
    cmdp = ModelBasedCollectionsMDP()
    cmdp.solve_value_iteration()
    app = build_compliance_orchestrator(RegulatoryPolicyStore(), cmdp)

    out = app.invoke(
        {
            "workflow_type": "FRAUD_ADVERSE_ACTION",
            "account_id": "ACCT_99",
            "fraud_prob": 0.87,
            "phi_0": -2.1,
            "top_drivers": [("vel_amt_1h", 1.85), ("ppr_risk", 1.12)],
        }
    )
    parsed = json.loads(out["validated_json"])
    assert parsed["decision"] == "BLOCK_AND_FREEZE"
    assert parsed["human_escalation_available"] is True
    assert parsed["statutory_reasons"][0]["reason_code"] == "RC_01_VELOCITY_SPIKE"
