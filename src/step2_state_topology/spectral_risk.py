import networkx as nx
import numpy as np


def compute_fraud_seeded_ppr(
    causal_g: nx.MultiDiGraph,
    fraud_seeds: set[str],
    alpha: float = 0.85,
) -> dict[str, float]:
    """
    Computes Fraud-Seeded Personalized PageRank:
    r = (1 - alpha) * s + alpha * P^T * r
    via NetworkX's sparse power-iteration solver over the causal DiGraph projection.
    """
    if causal_g.number_of_nodes() == 0:
        return {}

    # Collapse parallel MultiDiGraph edges into weighted DiGraph for stochastic transition matrix P
    di_g = nx.DiGraph()
    di_g.add_nodes_from(causal_g.nodes())
    for u, v, data in causal_g.edges(data=True):
        w = data.get("weight", 1.0)
        if di_g.has_edge(u, v):
            di_g[u][v]["weight"] += w
        else:
            di_g.add_edge(u, v, weight=w)

    valid_seeds = [n for n in di_g.nodes() if n in fraud_seeds]
    if not valid_seeds:
        return {n: 0.0 for n in di_g.nodes()}

    personalization = {n: (1.0 / len(valid_seeds) if n in fraud_seeds else 0.0) for n in di_g.nodes()}
    return nx.pagerank(di_g, alpha=alpha, personalization=personalization, weight="weight")


def compute_closed_cycle_traces(causal_g: nx.MultiDiGraph, max_k: int = 3) -> dict[str, np.ndarray]:
    """
    Computes exact diagonal cycle counts C_k(v) = [A^k]_{vv} for k in {2, ..., max_k}
    using SciPy sparse adjacency matrix powers to expose circular layering loops.
    """
    nodes = list(causal_g.nodes())
    if not nodes:
        return {}

    a_sparse = nx.to_scipy_sparse_array(causal_g, nodelist=nodes, weight=None, format="csr")
    a_binary = (a_sparse > 0).astype(np.int64)

    diags = []
    a_power = a_binary @ a_binary  # k = 2
    for k in range(2, max_k + 1):
        if k > 2:
            a_power = a_power @ a_binary
        diags.append(a_power.diagonal())

    stacked = np.vstack(diags).T  # shape: [num_nodes, max_k - 1]
    return {node: stacked[i] for i, node in enumerate(nodes)}