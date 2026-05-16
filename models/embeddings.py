"""Time step embeddings for diffusion/flow models."""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for time steps, as used in DDPM/Rectified Flow."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1) time steps in [0, 1]

        Returns:
            embedding: (B, dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(-emb_scale * torch.arange(half_dim, device=t.device, dtype=t.dtype))
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)

        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
        return embedding


class TimeMLP(nn.Module):
    """Two-layer MLP that processes time embeddings to the target dimension."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)
