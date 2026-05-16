"""Physics projection operators that enforce physical constraints on velocity fields."""

import torch
import torch.nn as nn
from utils.physics import project_div_free_fft, apply_boundary_conditions


class PhysicsProjector(nn.Module):
    """Physics-constrained projection operator Π.

    Projects both initial noise and model predictions onto the physical manifold:
    - Prior projection: ensures x_0 satisfies physical priors (e.g., mass, BCs)
    - Velocity projection: ensures each step's update preserves physical constraints
    """

    def __init__(self, constraint_type: str = "incompressible"):
        super().__init__()
        self.constraint_type = constraint_type

    def project_state(self, x: torch.Tensor, target_mean: torch.Tensor | None = None) -> torch.Tensor:
        """Project state onto the physical manifold (used for x_0 initialization).

        For velocity fields with mass conservation: ensures the spatial mean matches target.
        For incompressible flows: ensures divergence-free.

        Args:
            x: (B, 2, H, W) velocity field to project
            target_mean: (B, 2, 1, 1) optional target mean per channel

        Returns:
            x_proj: (B, 2, H, W) projected state
        """
        if self.constraint_type == "incompressible":
            x = project_div_free_fft(x)

        if target_mean is not None:
            # Adjust mean to match target while keeping the field divergence-free
            x = x - x.mean(dim=[-1, -2], keepdim=True) + target_mean

        return x

    def project_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Project velocity update onto the tangent space of the physical manifold.

        Ensures that adding this velocity preserves physical constraints.

        Args:
            v: (B, 2, H, W) raw predicted velocity

        Returns:
            v_proj: (B, 2, H, W) physically admissible velocity
        """
        if self.constraint_type == "incompressible":
            v = project_div_free_fft(v)
        elif self.constraint_type == "mass_cons":
            # For mass conservation: updates must have zero mean
            v = v - v.mean(dim=[-1, -2], keepdim=True)

        return v

    def apply_boundary_conditions(self, v: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        """Hard-apply boundary conditions (zero velocity at walls)."""
        return apply_boundary_conditions(v, boundary_mask)
