import numpy as np
import torch

from src.step3_ml_engine.focal_loss import xgboost_focal_objective
from src.step3_ml_engine.graphsage_encoder import ResidualGraphSAGE
from src.step3_ml_engine.survival_collections import CreditSurvivalEngine


def test_gnn_dirichlet_energy_does_not_collapse():
    torch.manual_seed(0)
    x = torch.randn(30, 8)
    edge_index = torch.randint(0, 30, (2, 80))
    model = ResidualGraphSAGE(in_channels=8, hidden_channels=16)
    _, energies = model.encode(x, edge_index)
    assert len(energies) == 3  # Input + 2 layers
    assert energies[-1] > 0.1  # Residual connections + L2 norm prevent collapse to 0


def test_focal_gradient_matches_finite_difference():
    obj = xgboost_focal_objective(alpha=0.75, gamma=2.0)
    y = np.array([1.0, 0.0])
    z = np.array([0.4, -0.7])
    grad, hess = obj(y, z)

    def focal_val(logits):
        p = 1.0 / (1.0 + np.exp(-logits))
        pt = np.where(y == 1.0, p, 1.0 - p)
        at = np.where(y == 1.0, 0.75, 0.25)
        return -at * ((1.0 - pt) ** 2.0) * np.log(pt)

    eps = 1e-6
    num_grad = (focal_val(z + eps) - focal_val(z - eps)) / (2.0 * eps)
    assert np.allclose(grad, num_grad, atol=1e-5)
    assert np.all(hess > 0.0)


def test_survival_c_index_above_random():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(150, 4))
    t_true = rng.exponential(scale=100.0 * np.exp(-x[:, 0]))
    t_cens = rng.uniform(40.0, 160.0, size=150)
    events = t_true <= t_cens
    obs_t = np.minimum(t_true, t_cens)

    engine = CreditSurvivalEngine(use_rsf=False)
    engine.fit(x[:100], events[:100], obs_t[:100])
    c_idx = engine.evaluate_c_index(x[100:], events[100:], obs_t[100:])
    assert c_idx > 0.65
