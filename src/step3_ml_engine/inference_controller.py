import numpy as np

INFERENCE_ACTIONS = ("TABULAR_XGB_ONLY", "HYBRID_GNN_XGB", "HYBRID_PLUS_SURVIVAL")


class ModelBasedInferenceMDP:
    """
    Known (S, A, P, R, V) solved via Value Iteration when model inference latencies
    and PR-AUC lift per exposure tier are analytically known.
    State s = amount_tier * 4 + latency_budget_bucket (12 states).
    """

    def __init__(self, gamma: float = 0.95):
        self.num_states = 12  # 3 amount tiers x 4 remaining latency buckets
        self.num_actions = len(INFERENCE_ACTIONS)
        self.gamma = gamma
        self.P = np.zeros(
            (self.num_actions, self.num_states, self.num_states), dtype=np.float64
        )
        self.R = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.V = np.zeros(self.num_states, dtype=np.float64)
        self.policy = np.zeros(self.num_states, dtype=np.int64)
        self._build_tensors()

    def _build_tensors(self) -> None:
        for amt_t in range(3):
            for bud_b in range(4):
                s = amt_t * 4 + bud_b
                for a_idx, act in enumerate(INFERENCE_ACTIONS):
                    lat_cost = (
                        0
                        if act == "TABULAR_XGB_ONLY"
                        else (1 if act == "HYBRID_GNN_XGB" else 2)
                    )
                    b_next = max(0, bud_b - lat_cost)
                    self.P[a_idx, s, amt_t * 4 + b_next] = 1.0

                    if bud_b < lat_cost:
                        self.R[s, a_idx] = -40.0  # SLA overrun penalty
                    else:
                        exposure_mult = 1.0 + 1.5 * amt_t
                        pr_gain = (
                            6.0
                            if act == "TABULAR_XGB_ONLY"
                            else (12.0 if act == "HYBRID_GNN_XGB" else 14.5)
                        )
                        self.R[s, a_idx] = exposure_mult * pr_gain - 2.0 * lat_cost

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

    def best_action(self, amount_tier: int, budget_bucket: int) -> str:
        return INFERENCE_ACTIONS[int(self.policy[amount_tier * 4 + budget_bucket])]


class ModelFreeEnsembleQLearning:
    """
    Model-free Q-learning over (s, a, r, s', Q) sampling tuples adapting model selection
    under non-stationary concept drift and tail latency shocks.
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
        self.Q = np.zeros((3, 4, len(INFERENCE_ACTIONS)), dtype=np.float64)

    @staticmethod
    def _idx(amount_inr: float, remaining_ms: float) -> tuple[int, int]:
        amt_t = 2 if amount_inr > 100_000 else (1 if amount_inr > 10_000 else 0)
        bud_b = int(np.clip(remaining_ms // 10.0, 0, 3))
        return amt_t, bud_b

    def select_action(self, amount_inr: float, remaining_ms: float) -> str:
        a_t, b_b = self._idx(amount_inr, remaining_ms)
        if self.rng.random() < self.eps:
            return str(self.rng.choice(INFERENCE_ACTIONS))
        return INFERENCE_ACTIONS[int(np.argmax(self.Q[a_t, b_b]))]

    def update(
        self,
        s_raw: tuple[float, float],
        action: str,
        reward: float,
        s_next_raw: tuple[float, float],
    ) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]."""
        a1, b1 = self._idx(*s_raw)
        a2, b2 = self._idx(*s_next_raw)
        act_idx = INFERENCE_ACTIONS.index(action)
        td_target = reward + self.gamma * np.max(self.Q[a2, b2])
        self.Q[a1, b1, act_idx] += self.alpha * (td_target - self.Q[a1, b1, act_idx])
