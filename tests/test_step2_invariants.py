import numpy as np
import pandas as pd

from src.step2_state_topology.ego_sampler import CausalEgoGraphSampler
from src.step2_state_topology.heterogeneous_graph import (
    build_hetero_nx_graph,
    extract_causal_subgraph,
)
from src.step2_state_topology.oot_splitter import strict_oot_maturity_split
from src.step2_state_topology.spectral_risk import (
    compute_closed_cycle_traces,
    compute_fraud_seeded_ppr,
)
from src.step2_state_topology.temporal_velocity import compute_rolling_velocities


def test_rolling_velocity_blocks_current_row_leakage():
    df = pd.DataFrame(
        [
            ("u1", "u2", "TRANSFERS_TO", 100.0, 500.0),
            ("u1", "u3", "TRANSFERS_TO", 200.0, 99_999.0),
        ],
        columns=["src", "dst", "rel_type", "timestamp", "amount"],
    )
    out = compute_rolling_velocities(df, windows=("1h",))
    assert out.loc[0, "vel_amt_1h"] == 0.0
    assert out.loc[1, "vel_amt_1h"] == 500.0


def test_causal_ego_graph_caps_supernodes_and_blocks_future():
    rows = [("hub", f"nbr_{i}", "TRANSFERS_TO", float(i), 10.0) for i in range(400)]
    rows.append(("hub", "future_leak", "TRANSFERS_TO", 999.0, 50_000.0))
    df = pd.DataFrame(rows, columns=["src", "dst", "rel_type", "timestamp", "amount"])
    feats = {
        n: np.ones(2, dtype=np.float32) for n in set(df["src"]).union(set(df["dst"]))
    }

    g = build_hetero_nx_graph(df, feats)
    sampler = CausalEgoGraphSampler(fanout=(25, 10))
    ego_nodes = sampler.sample_ego_nodes(g, root_node="hub", t_obs=500.0)

    assert "future_leak" not in ego_nodes
    assert len(ego_nodes) <= 1 + 25 + 25 * 10


def test_ppr_and_cycle_detection():
    df = pd.DataFrame(
        [
            ("seed", "m1", "TRANSFERS_TO", 10.0, 100.0),
            ("m1", "m2", "TRANSFERS_TO", 20.0, 100.0),
            ("m2", "seed", "TRANSFERS_TO", 30.0, 100.0),
            ("clean", "other", "TRANSFERS_TO", 15.0, 10.0),
        ],
        columns=["src", "dst", "rel_type", "timestamp", "amount"],
    )
    feats = {
        n: np.ones(2, dtype=np.float32) for n in set(df["src"]).union(set(df["dst"]))
    }
    g = build_hetero_nx_graph(df, feats)
    causal_g = extract_causal_subgraph(g, set(feats.keys()), t_obs=50.0)

    ppr = compute_fraud_seeded_ppr(causal_g, fraud_seeds={"seed"})
    assert ppr["m1"] > ppr["clean"] == 0.0

    cycles = compute_closed_cycle_traces(causal_g, max_k=3)
    assert cycles["seed"][1] == 1  # k=3 cycle trace
