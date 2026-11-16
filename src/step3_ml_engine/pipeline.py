import numpy as np
import torch

from .hybrid_fraud_trainer import HybridGraphTabularFraudModel
from .inference_controller import ModelBasedInferenceMDP, ModelFreeEnsembleQLearning
from .survival_collections import CreditSurvivalEngine


def run_step3_pipeline() -> None:
    rng = np.random.default_rng(42)
    num_nodes = 200
    # Extreme class imbalance: < 1% fraud (only 2 fraud rings / 4 nodes out of 200)
    y_nodes = np.zeros(num_nodes, dtype=np.int64)
    y_nodes[[0, 1, 2, 3]] = 1

    x_nodes_np = rng.normal(size=(num_nodes, 8)).astype(np.float32)
    x_nodes_np[y_nodes == 1] += 1.5  # Topological risk signal in fraud clique
    x_nodes = torch.tensor(x_nodes_np)

    # Ring edges connecting mule cluster + sparse background graph
    src = [0, 1, 2, 3, 0, 2] + rng.integers(4, num_nodes, size=400).tolist()
    dst = [1, 2, 3, 0, 2, 0] + rng.integers(4, num_nodes, size=400).tolist()
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 1. Train Hybrid 2-Layer Residual GraphSAGE + Focal-XGBoost Model
    hybrid = HybridGraphTabularFraudModel(
        node_in_dim=8, gnn_hidden_dim=16, focal_alpha=0.80, focal_gamma=2.0
    )
    train_mask = torch.arange(num_nodes) < 140
    hybrid.fit_gnn(x_nodes, edge_index, torch.tensor(y_nodes), train_mask, epochs=40)
    print(
        f"[Dirichlet Energy] Layer 0->1->2: {[round(e, 4) for e in hybrid.last_dirichlet_energies]}"
    )

    x_tab = rng.normal(size=(num_nodes, 6)).astype(np.float32)
    x_tab[y_nodes == 1, 0] += 2.2  # High 1h spend velocity feature
    node_idx = np.arange(num_nodes)

    hybrid.fit_hybrid(x_tab[:140], y_nodes[:140], node_idx[:140], x_nodes, edge_index)
    pr_auc = hybrid.evaluate_pr_auc(x_tab, y_nodes, node_idx, x_nodes, edge_index)
    print(f"[Hybrid Fraud Engine] Imbalance={y_nodes.mean():.2%} | PR-AUC={pr_auc:.4f}")

    # 2. Credit Collection Time-to-Default Survival Analysis (scikit-survival CoxPH)
    n_loans = 300
    x_loans = rng.normal(size=(n_loans, 5))
    risk_latent = x_loans @ np.array([0.8, -0.5, 0.4, 0.0, 0.3])
    true_default_days = rng.exponential(scale=120.0 * np.exp(-risk_latent))
    censor_days = rng.uniform(30.0, 180.0, size=n_loans)
    events_occurred = true_default_days <= censor_days
    observed_days = np.minimum(true_default_days, censor_days)

    surv_engine = CreditSurvivalEngine(use_rsf=False)
    surv_engine.fit(x_loans[:220], events_occurred[:220], observed_days[:220])
    c_index = surv_engine.evaluate_c_index(
        x_loans[220:], events_occurred[220:], observed_days[220:]
    )
    surv_curves = surv_engine.predict_survival_at_horizons(
        x_loans[220:222], horizons_days=(30.0, 60.0, 90.0)
    )
    print(
        f"[Survival Collections] C-index={c_index:.4f} | Sample S(30d,60d,90d)={np.round(surv_curves[0], 3)}"
    )

    # 3. Dual State-Transition Control: Model-Based (S,A,P,R,V) & Model-Free (s,a,r,s',Q)
    inf_mdp = ModelBasedInferenceMDP()
    inf_mdp.solve_value_iteration()
    q_ens = ModelFreeEnsembleQLearning()
    act = q_ens.select_action(amount_inr=150_000.0, remaining_ms=25.0)
    q_ens.update((150_000.0, 25.0), act, reward=18.5, s_next_raw=(5_000.0, 35.0))
    print(
        f"[Inference Control] Model-Based(high_amt, high_budget)={inf_mdp.best_action(2, 3)} | Q-Action={act}"
    )


if __name__ == "__main__":
    run_step3_pipeline()
