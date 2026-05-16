"""PerFlow: Physics-embedded Rectified Flow model.

Combines a U-Net backbone with a hard physics projection operator to ensure
the predicted velocity fields satisfy physical constraints (div-free, BCs).
"""

import torch
import torch.nn as nn
from models.unet import UNet
from models.projector import PhysicsProjector


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel/DDP to access the underlying module."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


class PerFlowModel(nn.Module):
    """PerFlow model = conditional U-Net + PhysicsProjector.

    The model takes a noisy/intermediate state x_t, time t, sparse observation y_obs,
    and observation mask M, then predicts a physically admissible velocity v.

    Forward pass:
        1. Concatenate [x_t, y_obs, mask] as conditional input
        2. U-Net predicts raw velocity
        3. PhysicsProjector enforces hard constraints (div-free, BCs)
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_heads: int = 4,
        constraint_type: str = "incompressible",
    ):
        super().__init__()

        self.net = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            num_heads=num_heads,
        )
        self.projector = PhysicsProjector(constraint_type=constraint_type)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict physically admissible velocity.

        Args:
            x_t: (B, 2, H, W) intermediate/noisy state
            t: (B,) time steps in [0, 1]
            y_obs: (B, 2, H, W) sparse observation field
            mask: (B, 2, H, W) binary observation mask (1 = observed)

        Returns:
            v: (B, 2, H, W) predicted physically admissible velocity
        """
        # Concatenate conditional input along channel dim
        cond = torch.cat([x_t, y_obs, mask], dim=1)  # (B, 6, H, W)

        # U-Net forward pass
        v_raw = self.net(cond, t)  # (B, 2, H, W)

        # Hard physics constraint projection
        v_constrained = self.projector.project_velocity(v_raw)

        return v_constrained

    def project_state(self, x: torch.Tensor, target_mean: torch.Tensor | None = None) -> torch.Tensor:
        """Delegate to projector.project_state (accessible even when wrapped by DataParallel)."""
        return self.projector.project_state(x, target_mean)
