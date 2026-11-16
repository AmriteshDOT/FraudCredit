import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree


def compute_dirichlet_energy(h: torch.Tensor, edge_index: torch.Tensor) -> float:
    """
    Computes Graph Dirichlet Energy:
    E(H) = 0.5 * sum_{(u,v) in E} || h_u / sqrt(d_u + 1) - h_v / sqrt(d_v + 1) ||_2^2
    Monitors against GNN oversmoothing (E(H) -> 0).
    """
    if edge_index.numel() == 0:
        return 0.0
    src, dst = edge_index[0], edge_index[1]
    deg = degree(dst, num_nodes=h.size(0), dtype=h.dtype)
    norm = torch.rsqrt(deg + 1.0).unsqueeze(-1)
    h_norm = h * norm
    diff = h_norm[src] - h_norm[dst]
    return float(0.5 * torch.sum(diff.pow(2)).item())


class ResidualGraphSAGE(nn.Module):
    """
    2-Layer Inductive GraphSAGE with identity residual skip connections and L2 hypersphere
    normalization to prevent Dirichlet energy collapse across bank account nodes.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="max")
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier_head = nn.Linear(hidden_channels, 1)

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, list[float]]:
        h0 = F.gelu(self.input_proj(x))
        energies = [compute_dirichlet_energy(h0, edge_index)]

        # Layer l = 1 with identity residual connection
        h1 = h0 + self.dropout(F.gelu(self.norm1(self.conv1(h0, edge_index))))
        energies.append(compute_dirichlet_energy(h1, edge_index))

        # Layer l = 2 (strictly bounded depth l=2) + unit L2 normalization
        h2 = h1 + self.dropout(F.gelu(self.norm2(self.conv2(h1, edge_index))))
        z = F.normalize(h2, p=2, dim=-1)
        energies.append(compute_dirichlet_energy(z, edge_index))

        return z, energies

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(x, edge_index)
        return self.classifier_head(z).squeeze(-1)
