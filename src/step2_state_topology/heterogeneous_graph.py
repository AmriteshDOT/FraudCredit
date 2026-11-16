import networkx as nx
import numpy as np
import pandas as pd
import torch

from .temporal_velocity import bochner_time_encoding


def build_hetero_nx_graph(
    events_df: pd.DataFrame,
    node_features: dict[str, np.ndarray],
) -> nx.MultiDiGraph:
    """
    Constructs Continuous-Time Dynamic Heterogeneous Graph G = (V, E, X_V, X_E)
    using NetworkX MultiDiGraph with typed nodes and timestamped relational edges.
    """
    g = nx.MultiDiGraph()
    for node_id, feat in node_features.items():
        node_type = node_id.split("_")[0]
        g.add_node(node_id, node_type=node_type, x=np.asarray(feat, dtype=np.float32))

    for row in events_df.itertuples(index=False):
        g.add_edge(
            row.src,
            row.dst,
            rel_type=row.rel_type,
            timestamp=float(row.timestamp),
            amount=float(row.amount),
            weight=float(row.amount) + 1.0,
        )
        # Bipartite entity-sharing relations (Device/IP) propagate risk bidirectionally
        if row.rel_type in ("USES_DEVICE", "SHARES_IP"):
            g.add_edge(
                row.dst,
                row.src,
                rel_type=row.rel_type,
                timestamp=float(row.timestamp),
                amount=float(row.amount),
                weight=1.0,
            )
    return g


def extract_causal_subgraph(g: nx.MultiDiGraph, nodes: set[str], t_obs: float) -> nx.MultiDiGraph:
    """Slices strictly historical edges E(t_obs) = {e | tau < t_obs} over target node set."""
    causal_edges = [
        (u, v, k)
        for u, v, k, data in g.edges(nbunch=nodes, keys=True, data=True)
        if v in nodes and data["timestamp"] < t_obs
    ]
    return g.edge_subgraph(causal_edges).copy()


def to_pyg_data_dict(sub_g: nx.MultiDiGraph, t_obs: float, time_dim: int = 8) -> dict[str, torch.Tensor]:
    """Converts causal NetworkX subgraph into contiguous PyTorch CSR/COO tensors for Step 3 GNN."""
    node_list = list(sub_g.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.tensor(np.stack([sub_g.nodes[n]["x"] for n in node_list]), dtype=torch.float32)
    src_idx, dst_idx, edge_attr_list = [], [], []

    for u, v, data in sub_g.edges(data=True):
        src_idx.append(node_to_idx[u])
        dst_idx.append(node_to_idx[v])
        dt_vec = bochner_time_encoding(np.array([t_obs - data["timestamp"]]), dim=time_dim)[0]
        edge_attr_list.append(np.concatenate([[np.log1p(data["amount"])], dt_vec]))

    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    edge_attr = (
        torch.tensor(np.stack(edge_attr_list), dtype=torch.float32)
        if edge_attr_list
        else torch.empty((0, 1 + time_dim), dtype=torch.float32)
    )
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "node_to_idx": node_to_idx}