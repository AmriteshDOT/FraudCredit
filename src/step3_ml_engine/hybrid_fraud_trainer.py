import numpy as np
from sklearn.metrics import average_precision_score
import torch
import xgboost as xgb

from .focal_loss import BinaryFocalLoss, xgboost_focal_objective
from .graphsage_encoder import ResidualGraphSAGE


class HybridGraphTabularFraudModel:
    """
    Two-stage SOTA fraud classifier:
    1. Trains 2-layer ResidualGraphSAGE with Focal Loss on topological graph.
    2. Extracts unit-normalized node embeddings z_v and concatenates with tabular features:
       x_tilde = [x_tab || z_v].
    3. Trains histogram-based XGBoost with custom Focal (g_i, h_i) optimizing PR-AUC.
    """

    def __init__(
        self,
        node_in_dim: int,
        gnn_hidden_dim: int = 32,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.gnn = ResidualGraphSAGE(
            in_channels=node_in_dim, hidden_channels=gnn_hidden_dim
        )
        self.focal_criterion = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective=xgboost_focal_objective(alpha=focal_alpha, gamma=focal_gamma),
            eval_metric="aucpr",
            random_state=seed,
        )
        self.last_dirichlet_energies: list[float] = []

    def fit_gnn(
        self,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        y_nodes: torch.Tensor,
        train_mask: torch.Tensor,
        epochs: int = 60,
        lr: float = 1e-2,
    ) -> None:
        optimizer = torch.optim.AdamW(self.gnn.parameters(), lr=lr, weight_decay=1e-4)
        self.gnn.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.gnn(x_nodes, edge_index)
            loss = self.focal_criterion(logits[train_mask], y_nodes[train_mask])
            loss.backward()
            optimizer.step()

        self.gnn.eval()
        with torch.no_grad():
            _, self.last_dirichlet_energies = self.gnn.encode(x_nodes, edge_index)

    def _fuse_features(
        self,
        x_tab: np.ndarray,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        node_indices: np.ndarray,
    ) -> np.ndarray:
        self.gnn.eval()
        with torch.no_grad():
            z_all, _ = self.gnn.encode(x_nodes, edge_index)
        z_selected = z_all[node_indices].cpu().numpy()
        return np.hstack([x_tab, z_selected])

    def fit_hybrid(
        self,
        x_tab_train: np.ndarray,
        y_train: np.ndarray,
        train_node_idx: np.ndarray,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        x_fused = self._fuse_features(x_tab_train, x_nodes, edge_index, train_node_idx)
        self.xgb_model.fit(x_fused, y_train)

    def predict_proba(
        self,
        x_tab: np.ndarray,
        node_idx: np.ndarray,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        x_fused = self._fuse_features(x_tab, x_nodes, edge_index, node_idx)
        raw_margins = self.xgb_model.predict(x_fused, output_margin=True)
        return 1.0 / (1.0 + np.exp(-np.clip(raw_margins, -20.0, 20.0)))

    def evaluate_pr_auc(
        self,
        x_tab: np.ndarray,
        y_true: np.ndarray,
        node_idx: np.ndarray,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> float:
        probs = self.predict_proba(x_tab, node_idx, x_nodes, edge_index)
        return float(average_precision_score(y_true, probs))
