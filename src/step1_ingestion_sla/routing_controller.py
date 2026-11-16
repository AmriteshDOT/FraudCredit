import random
from typing import Dict, List, Tuple

ACTIONS = ("ROUTE_DEEP_ML", "ROUTE_CACHED_RULES", "SHED_LOAD")


class ModelBasedQueueMDP:
    """Known (S, A, P, R, V) solved via exact Value Iteration."""

    def __init__(
        self,
        max_q: int = 20,
        lam: float = 800.0,
        mu_ml: float = 1000.0,
        gamma: float = 0.95,
    ):
        self.states = list(range(max_q + 1))
        self.lam = lam
        self.mu_ml = mu_ml
        self.gamma = gamma
        self.V: Dict[int, float] = {s: 0.0 for s in self.states}
        self.policy: Dict[int, str] = {s: "ROUTE_DEEP_ML" for s in self.states}

    def transition_kernel(self, q: int, a: str) -> List[Tuple[int, float]]:
        dt = 0.001
        p_arr = min(0.95, self.lam * dt)
        mu_eff = (
            self.mu_ml
            if a == "ROUTE_DEEP_ML"
            else (self.mu_ml * 10.0 if a == "ROUTE_CACHED_RULES" else self.mu_ml * 20.0)
        )
        p_srv = min(0.95, mu_eff * dt) if q > 0 else 0.0

        q_up = min(len(self.states) - 1, q + 1)
        q_dn = max(0, q - 1)
        return [
            (q_up, p_arr * (1.0 - p_srv)),
            (q_dn, (1.0 - p_arr) * p_srv),
            (q, 1.0 - p_arr * (1.0 - p_srv) - (1.0 - p_arr) * p_srv),
        ]

    def reward(self, q: int, a: str) -> float:
        est_wait_ms = (q + 1) * (1000.0 / self.mu_ml) if a == "ROUTE_DEEP_ML" else 1.5
        acc_utility = (
            10.0
            if a == "ROUTE_DEEP_ML"
            else (6.5 if a == "ROUTE_CACHED_RULES" else -5.0)
        )
        sla_penalty = 50.0 * max(0.0, est_wait_ms - 40.0)
        return acc_utility - sla_penalty

    def solve_value_iteration(self, tol: float = 1e-6) -> None:
        """V*(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V*(s')]."""
        while True:
            delta = 0.0
            for s in self.states:
                best_val, best_act = -float("inf"), ACTIONS[0]
                for a in ACTIONS:
                    q_sa = sum(
                        p * (self.reward(s, a) + self.gamma * self.V[s_next])
                        for s_next, p in self.transition_kernel(s, a)
                    )
                    if q_sa > best_val:
                        best_val, best_act = q_sa, a
                delta = max(delta, abs(best_val - self.V[s]))
                self.V[s] = best_val
                self.policy[s] = best_act
            if delta < tol:
                break


class ModelFreeRouterQLearning:
    """Unknown shock dynamics learned via (s, a, r, s', Q) sampling tuples."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, eps: float = 0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Q: Dict[Tuple[int, int], Dict[str, float]] = {}

    def _discretize(self, wait_ms: float, cb_open: bool) -> Tuple[int, int]:
        w_bucket = min(5, int(wait_ms // 10.0))
        return (w_bucket, int(cb_open))

    def select_action(self, wait_ms: float, cb_open: bool) -> str:
        s = self._discretize(wait_ms, cb_open)
        q_vals = self.Q.setdefault(s, {a: 0.0 for a in ACTIONS})
        if random.random() < self.eps:
            return random.choice(ACTIONS)
        return max(q_vals, key=q_vals.get)

    def update(
        self,
        s_raw: Tuple[float, bool],
        a: str,
        r: float,
        s_next_raw: Tuple[float, bool],
    ) -> None:
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a'} Q(s',a') - Q(s,a)]."""
        s = self._discretize(*s_raw)
        s_next = self._discretize(*s_next_raw)
        q_s = self.Q.setdefault(s, {act: 0.0 for act in ACTIONS})
        q_next = self.Q.setdefault(s_next, {act: 0.0 for act in ACTIONS})
        td_target = r + self.gamma * max(q_next.values())
        q_s[a] += self.alpha * (td_target - q_s[a])
