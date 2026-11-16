import numpy as np
import shap
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import xgboost as xgb


class HybridSHAPExplainer:
    """
    Computes exact polynomial-time O(TLD^2) local Shapley attributions phi_i
    for the fused [x_tab || z_gnn] XGBoost risk classifier via shap.TreeExplainer.
    """

    def __init__(self, xgb_model: xgb.XGBClassifier, feature_names: list[str]):
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(xgb_model)

    def explain_instance(self, x_fused_row: np.ndarray) -> dict[str, object]:
        row_2d = np.asarray(x_fused_row, dtype=np.float32).reshape(1, -1)
        shap_vals = self.explainer.shap_values(row_2d)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        phi = np.asarray(shap_vals[0], dtype=np.float64)

        ev = self.explainer.expected_value
        phi_0 = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

        attributions = {name: float(val) for name, val in zip(self.feature_names, phi)}
        ranked = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return {
            "phi_0": phi_0,
            "phi_vector": phi,
            "attributions": attributions,
            "top_drivers": ranked[:5],
            "local_accuracy_sum": float(phi_0 + np.sum(phi)),
        }


def explain_gnn_subgraph(
    gnn_model: torch.nn.Module,
    x_nodes: torch.Tensor,
    edge_index: torch.Tensor,
    target_node_idx: int,
    epochs: int = 30,
) -> np.ndarray:
    """
    Optimizes GNNExplainer mutual-information edge mask M_E over 2-hop ego-graph
    to extract the top relational edges driving node risk.
    """
    explainer = Explainer(
        model=gnn_model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode="binary_classification", task_level="node", return_type="raw"
        ),
    )
    explanation = explainer(x_nodes, edge_index, index=target_node_idx)
    return explanation.edge_mask.detach().cpu().numpy()
