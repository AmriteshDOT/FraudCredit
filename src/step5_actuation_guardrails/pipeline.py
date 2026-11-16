import numpy as np

from .actuation_fsm import ActuationDFSM
from .cost_threshold import PlattCalibrator, derive_bayes_optimal_thresholds
from .idempotent_actuator import FencedIdempotentLedger
from .resolution_controller import ModelBasedActuationMDP, ModelFreeActuationQLearning
from .saga_coordinator import ActuationSagaOrchestrator


def run_step5_pipeline() -> None:
    rng = np.random.default_rng(42)
    logits = np.concatenate([rng.normal(-3.0, 1.0, 980), rng.normal(2.5, 0.8, 20)])
    y_true = np.array([0] * 980 + [1] * 20)

    # 1. Platt Probability Calibration & Bayes-Optimal Threshold Derivation
    cal = PlattCalibrator()
    cal.fit(logits, y_true)
    probs = cal.predict_proba(logits)
    tau_2fa, tau_fraud = derive_bayes_optimal_thresholds(
        y_true, probs, avg_fraud_amount=75_000.0
    )
    print(
        f"[Calibration & Thresholds] tau_2fa={tau_2fa:.3f} | tau_fraud*={tau_fraud:.3f}"
    )

    # 2. 5-Tuple DFSM Guarded by SHAP Invariant Valid(phi_i)
    dfsm = ActuationDFSM(tau_2fa=tau_2fa, tau_fraud=tau_fraud)
    valid_phi = dfsm.verify_shap_invariant(
        phi_0=-2.0,
        phi_vec=np.array([2.4, 2.1]),
        raw_logit=2.5,
        statutory_reasons=[{"code": "RC_01"}],
    )
    act_valid = dfsm.transition(
        calibrated_prob=0.93, shap_valid=valid_phi, ml_healthy=True
    )
    act_unexplained = dfsm.transition(
        calibrated_prob=0.93, shap_valid=False, ml_healthy=True
    )
    print(
        f"[DFSM Guard] Valid SHAP -> {act_valid.value} | Invalid SHAP -> {act_unexplained.value}"
    )

    # 3. Idempotent Ledger + Monotonic Fencing Tokens + Compensating Saga
    ledger = FencedIdempotentLedger()
    saga = ActuationSagaOrchestrator(ledger)
    res_commit = saga.execute_freeze_saga("TXN_1001", "ACCT_7788", '{"reason":"RC_01"}')
    res_replay = saga.execute_freeze_saga("TXN_1001", "ACCT_7788", '{"reason":"RC_01"}')
    res_rollback = saga.execute_freeze_saga(
        "TXN_1002", "ACCT_9900", '{"reason":"RC_01"}', simulate_audit_failure=True
    )
    print(
        f"[Idempotent Saga] First={res_commit['saga_status']} | Replay={res_replay['idempotent_replay']} | Failure={res_rollback['saga_status']}"
    )

    # 4. Dual State-Transition Control: Model-Based (S,A,P,R,V) & Model-Free (s,a,r,s',Q)
    act_mdp = ModelBasedActuationMDP()
    act_mdp.solve_value_iteration()
    q_act = ModelFreeActuationQLearning()
    chosen = q_act.select_action(calibrated_prob=0.92, pod_healthy=True)
    q_act.update((0.92, True), chosen, reward=95.0, s_next_raw=(0.10, True))
    print(
        f"[Resolution Control] Model-Based(high_p, healthy)={act_mdp.best_action(2, True)} | Degraded={act_mdp.best_action(2, False)}"
    )


if __name__ == "__main__":
    run_step5_pipeline()
