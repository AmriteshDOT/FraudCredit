from enum import Enum
import numpy as np


class ActuationState(str, Enum):
    INIT = "INIT"
    ALLOWED = "ALLOW"
    STEP_UP_2FA = "STEP_UP_2FA"
    FROZEN = "BLOCK_AND_FREEZE"
    FAILSAFE_DEGRADED = "FAILSAFE_STEP_UP"


class ActuationDFSM:
    """
    5-Tuple Deterministic Finite State Machine M = (Q, Sigma, delta, q0, F):
    Enforces mathematical explainability invariant Valid(phi_i) before allowing
    irreversible BLOCK_AND_FREEZE transitions.
    """

    def __init__(self, tau_2fa: float = 0.40, tau_fraud: float = 0.80):
        self.tau_2fa = tau_2fa
        self.tau_fraud = tau_fraud

    @staticmethod
    def verify_shap_invariant(
        phi_0: float,
        phi_vec: np.ndarray,
        raw_logit: float,
        statutory_reasons: list[dict],
        tol: float = 1e-4,
    ) -> bool:
        """Valid(phi_i) <=> |phi_0 + sum(phi_i) - z_i| < tol AND len(reasons) >= 1."""
        local_accuracy_ok = abs((phi_0 + float(np.sum(phi_vec))) - raw_logit) < tol
        has_statutory_code = len(statutory_reasons) >= 1
        return bool(local_accuracy_ok and has_statutory_code)

    def transition(
        self,
        calibrated_prob: float,
        shap_valid: bool,
        ml_healthy: bool = True,
    ) -> ActuationState:
        if not ml_healthy:
            # Silent model failure guardrail: never silently block on degraded ML
            return (
                ActuationState.FAILSAFE_DEGRADED
                if calibrated_prob > self.tau_2fa
                else ActuationState.ALLOWED
            )

        if calibrated_prob > self.tau_fraud and shap_valid:
            return ActuationState.FROZEN
        if calibrated_prob > self.tau_2fa:
            return ActuationState.STEP_UP_2FA
        return ActuationState.ALLOWED
