import numpy as np

COLLECTION_ACTIONS = (
    "NO_ACTION_WAIT",
    "WHATSAPP_SOFT_NUDGE",
    "IVR_PAYMENT_REMINDER",
    "RESTRUCTURE_EMI_OFFER",
    "HUMAN_AGENT_ESCALATE",
)


class ModelBasedCollectionsMDP:
    """
    Known (S, A, P, R, V) solved via Value Iteration when borrower survival hazard
    h(t|x) and statutory contact limits (max 3 contacts/week, 08:00-19:00 IST) are known.
    State s = dpd_bucket (0..3) * 4 + contacts_used (0..3) -> 16 states.
    """

    def __init__(self, gamma: float = 0.95):
        self.num_states = 16
        self.num_actions = len(COLLECTION_ACTIONS)
        self.gamma = gamma
        self.P = np.zeros(
            (self.num_actions, self.num_states, self.num_states), dtype=np.float64
        )
        self.R = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        self.V = np.zeros(self.num_states, dtype=np.float64)
        self.policy = np.zeros(self.num_states, dtype=np.int64)
        self._build_cmdp()

    def _build_cmdp(self) -> None:
        recovery_lift = (0.05, 0.22, 0.30, 0.48, 0.55)
        op_cost = (0.0, 2.0, 8.0, 25.0, 60.0)
        for dpd in range(4):
            for cnt in range(4):
                s = dpd * 4 + cnt
                for a_idx, act in enumerate(COLLECTION_ACTIONS):
                    is_outreach = act != "NO_ACTION_WAIT"
                    # Hard Constrained MDP penalty if weekly contact cap (cnt >= 3) is breached
                    if is_outreach and cnt >= 3:
                        self.P[a_idx, s, s] = 1.0
                        self.R[s, a_idx] = -500.0
                        continue

                    next_cnt = min(3, cnt + (1 if is_outreach else 0))
                    p_cure = min(0.85, recovery_lift[a_idx] * (1.1 - 0.15 * dpd))
                    next_dpd_cured = max(0, dpd - 1)
                    next_dpd_worse = min(3, dpd + 1)

                    self.P[a_idx, s, next_dpd_cured * 4 + next_cnt] += p_cure
                    self.P[a_idx, s, next_dpd_worse * 4 + next_cnt] += 1.0 - p_cure

                    expected_yield = 400.0 * p_cure
                    self.R[s, a_idx] = expected_yield - op_cost[a_idx]

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

    def best_compliant_action(
        self, dpd_bucket: int, contacts_used: int, in_legal_hours: bool
    ) -> str:
        if not in_legal_hours or contacts_used >= 3:
            return "NO_ACTION_WAIT"
        return COLLECTION_ACTIONS[int(self.policy[dpd_bucket * 4 + contacts_used])]


class ModelFreeCollectionsQLearning:
    """
    Model-free Q-learning over (s, a, r, s', Q) tuples adapting collection strategy
    when borrower behavioral elasticity to restructure offers is unknown.
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
        self.Q = np.zeros((4, 4, len(COLLECTION_ACTIONS)), dtype=np.float64)

    def select_action(
        self, dpd_bucket: int, contacts_used: int, in_legal_hours: bool
    ) -> str:
        # Hard pre-action compliance mask: AI cannot pick illegal outreach actions
        if not in_legal_hours or contacts_used >= 3:
            return "NO_ACTION_WAIT"
        if self.rng.random() < self.eps:
            return str(self.rng.choice(COLLECTION_ACTIONS))
        return COLLECTION_ACTIONS[int(np.argmax(self.Q[dpd_bucket, contacts_used]))]

    def update(
        self,
        s_raw: tuple[int, int],
        action: str,
        reward: float,
        s_next_raw: tuple[int, int],
    ) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]."""
        d1, c1 = s_raw
        d2, c2 = s_next_raw
        a_idx = COLLECTION_ACTIONS.index(action)
        td_target = reward + self.gamma * np.max(self.Q[d2, c2])
        self.Q[d1, c1, a_idx] += self.alpha * (td_target - self.Q[d1, c1, a_idx])
