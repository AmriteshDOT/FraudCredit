from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.step5_actuation_guardrails.actuation_fsm import ActuationDFSM, ActuationState
from src.step5_actuation_guardrails.fastapi_gateway import app, service_state
from src.step5_actuation_guardrails.idempotent_actuator import FencedIdempotentLedger
from src.step5_actuation_guardrails.saga_coordinator import ActuationSagaOrchestrator


def test_dfsm_blocks_freeze_without_valid_shap():
    dfsm = ActuationDFSM(tau_2fa=0.40, tau_fraud=0.80)
    assert (
        dfsm.transition(calibrated_prob=0.95, shap_valid=False)
        == ActuationState.STEP_UP_2FA
    )
    assert (
        dfsm.transition(calibrated_prob=0.95, shap_valid=True) == ActuationState.FROZEN
    )


def test_fencing_token_rejects_stale_writer_and_saga_rolls_back():
    ledger = FencedIdempotentLedger()
    t1 = ledger.issue_fencing_token()
    t2 = ledger.issue_fencing_token()

    ledger.apply_mutation("k2", "ACCT_1", "FROZEN", fencing_token=t2)
    with pytest.raises(RuntimeError):
        ledger.apply_mutation("k1", "ACCT_1", "ACTIVE", fencing_token=t1)

    saga = ActuationSagaOrchestrator(ledger)
    out = saga.execute_freeze_saga(
        "TXN_FAIL", "ACCT_2", "{}", simulate_audit_failure=True
    )
    assert out["saga_status"] == "COMPENSATED_ROLLED_BACK"
    assert ledger.get_status("ACCT_2") == "ACTIVE"


def test_fastapi_kubernetes_probes_and_failsafe_degradation():
    with TestClient(app) as client:
        assert client.get("/health/startup").status_code == 200
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200

        # Simulate silent ML container failure -> gateway degrades safely to STEP_UP instead of freezing
        service_state.ml_engine_healthy = False
        try:
            resp = client.post(
                "/v1/actuate",
                json={
                    "txn_id": "TXN_SAFE_1",
                    "account_id": "ACCT_SAFE",
                    "calibrated_prob": 0.96,
                    "shap_valid": True,
                    "compliance_json": "{}",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["action"] == "FAILSAFE_STEP_UP"
        finally:
            service_state.ml_engine_healthy = True
