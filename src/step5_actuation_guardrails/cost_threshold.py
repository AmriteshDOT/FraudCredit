import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import precision_recall_curve


class PlattCalibrator:
    """
    Fits sigmoid p_hat = 1 / (1 + exp(-(w * z + b))) via maximum likelihood
    to restore calibrated posterior probabilities from Focal-Loss logits.
    """

    def __init__(self):
        self.w = 1.0
        self.b = 0.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> None:
        z = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)

        def nll(params: np.ndarray) -> float:
            w, b = params
            m = np.clip(w * z + b, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-m))
            p = np.clip(p, 1e-9, 1.0 - 1e-9)
            return float(
                -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)) + 1e-4 * (w**2)
            )

        res = minimize(nll, x0=np.array([1.0, 0.0]), method="BFGS")
        self.w, self.b = float(res.x[0]), float(res.x[1])

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        m = np.clip(self.w * np.asarray(logits, dtype=np.float64) + self.b, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-m))


def derive_bayes_optimal_thresholds(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    avg_fraud_amount: float = 50_000.0,
    c_fn_ratio: float = 1.0,
    c_fp_freeze: float = 1_500.0,
) -> tuple[float, float]:
    """
    Finds tau_fraud* minimizing expected economic loss across the PR curve:
    L(tau) = C_FN * FN(tau) * E[A_fraud] + C_FP * FP(tau)
    Returns (tau_2fa, tau_fraud).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, calibrated_probs)
    if len(thresholds) == 0:
        return 0.40, 0.80

    best_tau = 0.80
    min_loss = float("inf")
    for tau in thresholds:
        preds = (calibrated_probs >= tau).astype(int)
        fn = float(np.sum((y_true == 1) & (preds == 0)))
        fp = float(np.sum((y_true == 0) & (preds == 1)))
        expected_loss = (c_fn_ratio * avg_fraud_amount * fn) + (c_fp_freeze * fp)
        if expected_loss < min_loss:
            min_loss = expected_loss
            best_tau = float(tau)

    tau_fraud = float(np.clip(best_tau, 0.50, 0.98))
    tau_2fa = float(np.clip(0.5 * tau_fraud, 0.20, tau_fraud - 0.10))
    return tau_2fa, tau_fraud
