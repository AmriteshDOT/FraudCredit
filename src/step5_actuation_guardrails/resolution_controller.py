import numpy as np

RESOLUTION_ACTIONS = (
    "ALLOW",
    "STEP_UP_2FA",
    "BLOCK_AND_FREEZE",
    "FAILSAFE_RULES_BYPASS",
)


class ModelBasedActuationMDP:
    """
    Known (S, A, P, R, V) solved via Value Iteration when cost matrix (C_FN, C_FP)
    and ML pod health laws are known in closed form.
    State s = prob_bucket (0..2) * 2 + pod_healthy (0..1) -> 6 states.
    """

    def __init__(self, gamma: float = 0.95):
        self.num_states = 6
        self.num_actions = len(RESOLUTION_ACTIONS)
        self.gamma = gamma
        self.P = np.zeros(
            (self.num_actions, self.num_states, self.num_states), dtype=np.float64
        )
        self.R = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.V = np.zeros(self.num_states, dtype=np.float64)
        self.policy = np.zeros(self.num_states, dtype=np.int64)
        self._build_tensors()

    def _build_tensors(self) -> None:
        fraud_prior = (0.01, 0.55, 0.94)
        for p_b in range(3):
            for healthy in (0, 1):
                s = p_b * 2 + healthy
                p_f = fraud_prior[p_b]
                for a_idx, act in enumerate(RESOLUTION_ACTIONS):
                    self.P[a_idx, s, s] = (
                        1.0  # Absorbing terminal resolution per transaction
                    )
                    if healthy == 0 and act == "BLOCK_AND_FREEZE":
                        # Forbidden unsafe freeze when ML container is degraded
                        self.R[s, a_idx] = -10_000.0
                    elif act == "ALLOW":
                        self.R[s, a_idx] = -500.0 * p_f
                    elif act == "STEP_UP_2FA":
                        self.R[s, a_idx] = -40.0 * p_f - 5.0 * (1.0 - p_f)
                    elif act == "BLOCK_AND_FREEZE":
                        self.R[s, a_idx] = 100.0 * p_f - 150.0 * (1.0 - p_f)
                    else:  # FAILSAFE_RULES_BYPASS
                        self.R[s, a_idx] = -80.0 * p_f - 10.0 * (1.0 - p_f)

    def solve_value_iteration(self, tol: float = 1e-7) -> None:
        """V*(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V*(s')]."""
        while True:
            q_sa = self.R + self.gamma * np.einsum("ast,t->sa", self.P, self.V)
            v_next = np.max(q_sa, axis=1)
            if np.max(np.abs(v_next - self.V)) < tol:
                self.V = v_next
                self.policy = np.argmax(q_sa, axis=1)
                break
            self.V = v_next

    def best_action(self, prob_bucket: int, pod_healthy: bool) -> str:
        return RESOLUTION_ACTIONS[int(self.policy[prob_bucket * 2 + int(pod_healthy)])]


class ModelFreeActuationQLearning:
    """
    Model-free Q-learning over (s, a, r, s', Q) tuples adapting resolution decisions
    when merchant chargeback rates or customer 2FA drop-off friction shift online.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        gamma: float = 0.95,
        eps: float = 0.05,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((3, 2, len(RESOLUTION_ACTIONS)), dtype=np.float64)

    @staticmethod
    def _idx(calibrated_prob: float, pod_healthy: bool) -> tuple[int, int]:
        p_b = 2 if calibrated_prob > 0.80 else (1 if calibrated_prob > 0.40 else 0)
        return p_b, int(pod_healthy)

    def select_action(self, calibrated_prob: float, pod_healthy: bool) -> str:
        p_b, h_b = self._idx(calibrated_prob, pod_healthy)
        if not pod_healthy:
            return "FAILSAFE_RULES_BYPASS"
        if self.rng.random() < self.eps:
            return str(self.rng.choice(RESOLUTION_ACTIONS))
        return RESOLUTION_ACTIONS[int(np.argmax(self.Q[p_b, h_b]))]

    def update(
        self,
        s_raw: tuple[float, bool],
        action: str,
        reward: float,
        s_next_raw: tuple[float, bool],
    ) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]."""
        p1, h1 = self._idx(*s_raw)
        p2, h2 = self._idx(*s_next_raw)
        a_idx = RESOLUTION_ACTIONS.index(action)
        td_target = reward + self.gamma * np.max(self.Q[p2, h2])
        self.Q[p1, h1, a_idx] += self.alpha * (td_target - self.Q[p1, h1, a_idx])
