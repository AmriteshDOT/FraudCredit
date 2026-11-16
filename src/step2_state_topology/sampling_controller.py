import numpy as np

SAMPLING_ACTIONS = ("SAMPLE_RECENCY", "SAMPLE_AMOUNT", "SAMPLE_PPR_WEIGHTED", "HALT_HOP")


class ModelBasedGraphMDP:
    """
    Vectorized Bellman Value Iteration over known (S, A, P, R, V):
    V*(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V*(s')]
    States s = hop * 5 + budget_bucket (15 discrete states).
    """
    def __init__(self, gamma: float = 0.95):
        self.num_states = 15  # 3 hops (0, 1, 2) x 5 latency budget buckets (0..4)
        self.num_actions = len(SAMPLING_ACTIONS)
        self.gamma = gamma
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states), dtype=np.float64)
        self.R = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.V = np.zeros(self.num_states, dtype=np.float64)
        self.policy = np.zeros(self.num_states, dtype=np.int64)
        self._build_mdp_tensors()

    def _build_mdp_tensors(self) -> None:
        for h in range(3):
            for b in range(5):
                s = h * 5 + b
                for a_idx, act in enumerate(SAMPLING_ACTIONS):
                    if act == "HALT_HOP" or h == 2 or b == 0:
                        s_halt = 2 * 5 + b
                        self.P[a_idx, s, s_halt] = 1.0
                        self.R[s, a_idx] = 0.0
                        continue

                    cost = 1 if act in ("SAMPLE_RECENCY", "SAMPLE_AMOUNT") else 2
                    b_nom = max(0, b - cost)
                    b_spike = max(0, b - cost - 1)
                    self.P[a_idx, s, (h + 1) * 5 + b_nom] = 0.85
                    self.P[a_idx, s, (h + 1) * 5 + b_spike] += 0.15

                    if b <= 1:
                        self.R[s, a_idx] = -25.0  # SLA breach penalty
                    else:
                        gain = 12.0 if (h == 0 and act == "SAMPLE_RECENCY") else (9.0 if act == "SAMPLE_PPR_WEIGHTED" else 6.5)
                        self.R[s, a_idx] = gain - 1.5 * cost

    def solve_value_iteration(self, tol: float = 1e-7) -> None:
        while True:
            # Vectorized Bellman backup across all states and actions simultaneously
            q_sa = self.R + self.gamma * np.einsum("ast,t->sa", self.P, self.V)
            v_next = np.max(q_sa, axis=1)
            if np.max(np.abs(v_next - self.V)) < tol:
                self.V = v_next
                self.policy = np.argmax(q_sa, axis=1)
                break
            self.V = v_next

    def best_action(self, hop: int, budget_bucket: int) -> str:
        return SAMPLING_ACTIONS[int(self.policy[hop * 5 + budget_bucket])]


class ModelFreeSamplerQLearning:
    """
    Tabular Q-learning over empirical (s, a, r, s', Q) sampling tuples when
    super-node fan-out or adversarial camouflage dynamics are non-stationary.
    """
    def __init__(self, alpha: float = 0.15, gamma: float = 0.95, eps: float = 0.05, seed: int = 42):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        # 3 degree buckets x 5 latency buckets x 4 actions
        self.Q = np.zeros((3, 5, len(SAMPLING_ACTIONS)), dtype=np.float64)

    @staticmethod
    def _idx(frontier_max_deg: int, remaining_ms: float) -> tuple[int, int]:
        deg_b = 2 if frontier_max_deg > 10_000 else (1 if frontier_max_deg > 500 else 0)
        lat_b = int(np.clip(remaining_ms // 5.0, 0, 4))
        return deg_b, lat_b

    def select_action(self, frontier_max_deg: int, remaining_ms: float) -> str:
        d_b, l_b = self._idx(frontier_max_deg, remaining_ms)
        if self.rng.random() < self.eps:
            return str(self.rng.choice(SAMPLING_ACTIONS))
        return SAMPLING_ACTIONS[int(np.argmax(self.Q[d_b, l_b]))]

    def update(
        self,
        s_raw: tuple[int, float],
        action: str,
        reward: float,
        s_next_raw: tuple[int, float],
    ) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]."""
        d1, l1 = self._idx(*s_raw)
        d2, l2 = self._idx(*s_next_raw)
        a_idx = SAMPLING_ACTIONS.index(action)
        td_target = reward + self.gamma * np.max(self.Q[d2, l2])
        self.Q[d1, l1, a_idx] += self.alpha * (td_target - self.Q[d1, l1, a_idx])