import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


class CreditSurvivalEngine:
    """
    Time-to-default survival analysis for right-censored loan portfolios:
    h(t | x_i) = h_0(t) * exp(beta^T x_i),  S(t | x_i) = exp(-H_0(t) * exp(beta^T x_i))
    Evaluated strictly via Harrell's Concordance Index (C-index).
    """

    def __init__(self, use_rsf: bool = False, alpha_reg: float = 1e-2, seed: int = 42):
        self.use_rsf = use_rsf
        if use_rsf:
            self.model = RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=seed,
            )
        else:
            self.model = CoxPHSurvivalAnalysis(alpha=alpha_reg)

    @staticmethod
    def make_structured_target(
        events_occurred: np.ndarray, time_to_event: np.ndarray
    ) -> np.ndarray:
        return Surv.from_arrays(
            event=events_occurred.astype(bool), time=time_to_event.astype(np.float64)
        )

    def fit(
        self,
        x_borrower: np.ndarray,
        events_occurred: np.ndarray,
        time_to_event: np.ndarray,
    ) -> None:
        y_surv = self.make_structured_target(events_occurred, time_to_event)
        self.model.fit(x_borrower, y_surv)

    def predict_risk_scores(self, x_borrower: np.ndarray) -> np.ndarray:
        return self.model.predict(x_borrower)

    def predict_survival_at_horizons(
        self,
        x_borrower: np.ndarray,
        horizons_days: tuple[float, ...] = (30.0, 60.0, 90.0),
    ) -> np.ndarray:
        """Evaluates S(t | x_i) = P(T > t | x_i) across collection intervention horizons."""
        surv_fns = self.model.predict_survival_function(x_borrower)
        out = np.zeros((len(surv_fns), len(horizons_days)), dtype=np.float64)
        for i, fn in enumerate(surv_fns):
            clipped_t = np.clip(horizons_days, fn.x[0], fn.x[-1])
            out[i] = fn(clipped_t)
        return out

    def evaluate_c_index(
        self,
        x_borrower: np.ndarray,
        events_occurred: np.ndarray,
        time_to_event: np.ndarray,
    ) -> float:
        risk_scores = self.predict_risk_scores(x_borrower)
        c_idx, _, _, _, _ = concordance_index_censored(
            events_occurred.astype(bool), time_to_event.astype(np.float64), risk_scores
        )
        return float(c_idx)
