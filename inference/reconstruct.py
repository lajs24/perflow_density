"""Reconstruction pipeline: from sparse observations to full velocity field via ODE integration."""

import torch
import torch.nn as nn
from inference.solver import get_solver
from models.perflow import unwrap_model


@torch.no_grad()
def reconstruct(
    model: nn.Module,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = 50,
    solver: str = "euler",
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Reconstruct complete velocity field from sparse observations.

    Pipeline:
        1. Initialize x_0 as noise projected onto physical manifold
        2. Integrate ODE from t=0 to t=1 using the specified solver
        3. Return x_1 as the reconstructed field

    Args:
        model: PerFlow model (in eval mode)
        y_obs: (B, 2, H, W) sparse observations
        mask: (B, 2, H, W) observation mask
        num_steps: number of ODE integration steps
        solver: "euler" or "heun"
        return_trajectory: if True, return all intermediate states

    Returns:
        x_1: (B, 2, H, W) reconstructed full velocity field
        trajectory: (num_steps+1, B, 2, H, W) if return_trajectory=True
    """
    model.eval()
    device = y_obs.device
    step_fn = get_solver(solver)
    dt = 1.0 / num_steps

    # 1. Initialize noise with physical prior projection
    noise = torch.randn_like(y_obs)
    target_mean = (y_obs.sum(dim=[-1, -2], keepdim=True) /
                   mask.sum(dim=[-1, -2], keepdim=True).clamp(min=1))
    x = unwrap_model(model).project_state(noise, target_mean)

    trajectory = [x.clone()] if return_trajectory else None

    # 2. ODE integration from t=0 to t=1
    for i in range(num_steps):
        t = i * dt  # current time
        x = step_fn(model, x, t, dt, y_obs, mask)

        if return_trajectory:
            trajectory.append(x.clone())

    if return_trajectory:
        return x, torch.stack(trajectory)
    return x


@torch.no_grad()
def reconstruct_with_uncertainty(
    model: nn.Module,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = 50,
    solver: str = "euler",
    num_samples: int = 10,
) -> dict:
    """Run reconstruction multiple times with different noise seeds for UQ.

    Args:
        model: PerFlow model (in eval mode)
        y_obs: (B, 2, H, W) sparse observations
        mask: (B, 2, H, W) observation mask
        num_steps: number of ODE steps per sample
        solver: "euler" or "heun"
        num_samples: number of reconstructions with different seeds

    Returns:
        dict with:
            - "mean": (B, 2, H, W) mean of reconstructions
            - "var": (B, 2, H, W) variance across reconstructions
            - "std": (B, 2, H, W) std dev across reconstructions
            - "samples": (num_samples, B, 2, H, W) individual samples
    """
    model.eval()
    samples = []

    for s in range(num_samples):
        # Different seed per sample (implicit from different noise)
        x_1 = reconstruct(model, y_obs, mask, num_steps, solver, return_trajectory=False)
        samples.append(x_1.cpu())

    samples = torch.stack(samples)  # (num_samples, B, 2, H, W)

    mean = samples.mean(dim=0).to(y_obs.device)
    var = samples.var(dim=0, unbiased=True).to(y_obs.device)
    std = samples.std(dim=0, unbiased=True).to(y_obs.device)

    return {
        "mean": mean,
        "var": var,
        "std": std,
        "samples": samples,
    }
