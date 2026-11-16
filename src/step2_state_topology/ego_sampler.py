import networkx as nx
import numpy as np


class CausalEgoGraphSampler:
    """
    Bounded 2-hop causal ego-graph sampler defeating super-node explosion O(d_max^L):
    |V_ego(v)| <= 1 + k1 + k1 * k2 = O(1), strictly filtering tau < t_obs.
    """
    def __init__(self, fanout: tuple[int, int] = (25, 10), seed: int = 42):
        self.fanout = fanout
        self.rng = np.random.default_rng(seed)

    def _sample_frontier_edges(
        self,
        edges: list[tuple[str, str, dict]],
        k: int,
        strategy: str,
        ppr_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, str, dict]]:
        if len(edges) <= k:
            return edges
        if strategy == "SAMPLE_RECENCY":
            return sorted(edges, key=lambda e: e[2]["timestamp"], reverse=True)[:k]
        if strategy == "SAMPLE_AMOUNT":
            return sorted(edges, key=lambda e: e[2]["amount"], reverse=True)[:k]
        if strategy == "SAMPLE_PPR_WEIGHTED" and ppr_scores is not None:
            weights = np.array([ppr_scores.get(e[1], 0.0) + 1e-4 for e in edges], dtype=np.float64)
            probs = weights / weights.sum()
            idx = self.rng.choice(len(edges), size=k, replace=False, p=probs)
            return [edges[i] for i in idx]
        idx = self.rng.choice(len(edges), size=k, replace=False)
        return [edges[i] for i in idx]

    def sample_ego_nodes(
        self,
        g: nx.MultiDiGraph,
        root_node: str,
        t_obs: float,
        strategy: str = "SAMPLE_RECENCY",
        ppr_scores: dict[str, float] | None = None,
    ) -> set[str]:
        visited = {root_node}
        frontier = {root_node}

        for k_budget in self.fanout:
            next_frontier: set[str] = set()
            for u in frontier:
                if u not in g:
                    continue
                causal_out = [
                    (u, v, data)
                    for _, v, data in g.out_edges(u, data=True)
                    if data["timestamp"] < t_obs
                ]
                chosen = self._sample_frontier_edges(causal_out, k_budget, strategy, ppr_scores)
                for _, v, _ in chosen:
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
            frontier = next_frontier
            if not frontier:
                break

        return visited