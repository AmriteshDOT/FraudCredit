import numpy as np
import pandas as pd

from .ego_sampler import CausalEgoGraphSampler
from .heterogeneous_graph import build_hetero_nx_graph, extract_causal_subgraph, to_pyg_data_dict
from .oot_splitter import strict_oot_maturity_split
from .sampling_controller import ModelBasedGraphMDP, ModelFreeSamplerQLearning
from .spectral_risk import compute_closed_cycle_traces, compute_fraud_seeded_ppr
from .temporal_velocity import compute_rolling_velocities


def run_step2_pipeline() -> None:
    day = 86400.0
    events_df = pd.DataFrame(
        [
            ("acct_seed", "acct_m1", "TRANSFERS_TO", 5 * day, 95_000.0, 1),
            ("acct_m1", "acct_m2", "TRANSFERS_TO", 5 * day + 120.0, 92_000.0, 1),
            ("acct_m2", "acct_seed", "TRANSFERS_TO", 5 * day + 240.0, 90_000.0, 1),
            ("acct_m1", "dev_shared_99", "USES_DEVICE", 5 * day + 10.0, 0.0, 1),
            ("acct_target", "dev_shared_99", "USES_DEVICE", 38 * day, 0.0, 0),
            ("acct_target", "acct_merch_1", "TRANSFERS_TO", 40 * day, 145_000.0, 1),
            ("acct_target", "acct_future", "TRANSFERS_TO", 42 * day, 250_000.0, 1),
        ],
        columns=["src", "dst", "rel_type", "timestamp", "amount", "is_fraud"],
    )

    # 1. Strict OOT split with 30-day chargeback maturity buffer
    d_train, d_val = strict_oot_maturity_split(events_df, t_split=36 * day, t_val_end=45 * day, maturity_lag_sec=30 * day)
    print(f"[OOT Split] Train={len(d_train)} rows | Val={len(d_val)} rows")

    # 2. Vectorized causal rolling velocities (closed='left')
    vel_df = compute_rolling_velocities(events_df, windows=("1h", "24h"))
    print(f"[Tabular Velocity] Target 1h spend={vel_df.loc[5, 'vel_amt_1h']:.1f} | z-score={vel_df.loc[5, 'vel_zscore_1h']:.2f}")

    # 3. Build Heterogeneous Financial Graph & Causal Subgraph at t_obs = Day 40
    all_nodes = set(events_df["src"]).union(set(events_df["dst"]))
    node_feats = {n: np.random.default_rng(hash(n) & 0xFFFF).normal(size=4).astype(np.float32) for n in all_nodes}
    full_g = build_hetero_nx_graph(events_df, node_feats)

    t_obs = 40 * day
    causal_g = extract_causal_subgraph(full_g, all_nodes, t_obs=t_obs)

    # 4. Fraud-Seeded Personalized PageRank & Closed Layering Cycle Traces
    ppr = compute_fraud_seeded_ppr(causal_g, fraud_seeds={"acct_seed"})
    cycles = compute_closed_cycle_traces(causal_g, max_k=3)
    print(f"[Spectral Risk] Target PPR={ppr.get('acct_target', 0.0):.4f} | Seed 3-Cycles={cycles['acct_seed'][1]}")

    # 5. Bounded 2-Hop Ego-Graph Sampling -> PyTorch Tensors for Step 3 GNN
    sampler = CausalEgoGraphSampler(fanout=(25, 10))
    ego_nodes = sampler.sample_ego_nodes(full_g, root_node="acct_target", t_obs=t_obs, strategy="SAMPLE_PPR_WEIGHTED", ppr_scores=ppr)
    assert "acct_future" not in ego_nodes
    ego_sub_g = extract_causal_subgraph(full_g, ego_nodes, t_obs=t_obs)
    pyg_tensors = to_pyg_data_dict(ego_sub_g, t_obs=t_obs)
    print(f"[PyG Tensor Output] x={tuple(pyg_tensors['x'].shape)} | edge_index={tuple(pyg_tensors['edge_index'].shape)}")

    # 6. Dual MDP State-Transition Control: (S,A,P,R,V) & (s,a,r,s',Q)
    mdp = ModelBasedGraphMDP()
    mdp.solve_value_iteration()
    q_ctrl = ModelFreeSamplerQLearning()
    chosen_act = q_ctrl.select_action(frontier_max_deg=12_000, remaining_ms=14.0)
    q_ctrl.update((12_000, 14.0), chosen_act, reward=8.2, s_next_raw=(180, 9.5))
    print(f"[Control Policy] Model-Based(hop=0, b=4)={mdp.best_action(0, 4)} | Model-Free Q-Action={chosen_act}")


if __name__ == "__main__":
    run_step2_pipeline()