import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    L_focal = -(1/N) * sum_i alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy majority-class transactions so gradients focus on hard fraud cases.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        return torch.mean(alpha_t * modulating_factor * bce)


def xgboost_focal_objective(alpha: float = 0.75, gamma: float = 2.0):
    """
    Returns custom XGBoost objective yielding exact analytical 1st-order gradient g_i
    and 2nd-order Hessian h_i of Binary Focal Loss with respect to raw margin z_i.
    """

    def _objective(
        y_true: np.ndarray, y_pred_logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        z = np.clip(y_pred_logits, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        y = y_true.astype(np.float64)

        p_t = np.where(y == 1.0, p, 1.0 - p)
        p_t = np.clip(p_t, 1e-7, 1.0 - 1e-7)
        alpha_t = np.where(y == 1.0, alpha, 1.0 - alpha)
        sign = 2.0 * y - 1.0  # +1 for fraud, -1 for legitimate

        # First derivative dL / dz
        one_minus_pt = 1.0 - p_t
        grad = (
            alpha_t
            * (one_minus_pt ** (gamma - 1.0))
            * (gamma * p_t * np.log(p_t) + p_t - 1.0)
            * sign
        )

        # Second derivative d^2L / dz^2 (positive curvature stabilized for Newton tree splits)
        u = gamma * p_t * np.log(p_t) + p_t - 1.0
        du_dpt = gamma * (np.log(p_t) + 1.0) + 1.0
        dL_dpt = alpha_t * (one_minus_pt ** (gamma - 1.0)) * u
        d2L_dpt2 = (
            alpha_t
            * (one_minus_pt ** (gamma - 2.0))
            * (-(gamma - 1.0) * u + one_minus_pt * du_dpt)
        )
        dp_dz = p * (1.0 - p)
        hess = d2L_dpt2 * (dp_dz**2) + dL_dpt * sign * dp_dz * (1.0 - 2.0 * p)
        hess = np.maximum(np.abs(hess), 1e-6)

        return grad, hess

    return _objective
