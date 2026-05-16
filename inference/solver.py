"""ODE solvers for PerFlow rectified flow sampling (Euler, Heun)."""

import torch
import torch.nn as nn


@torch.no_grad()
def euler_step(
    model: nn.Module,
    x: torch.Tensor,
    t: float,
    dt: float,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Single Euler ODE step: x_{t+dt} = x_t + v(x_t, t) * dt.

    Args:
        model: PerFlow model
        x: (B, 2, H, W) current state
        t: current time
        dt: time step size
        y_obs: (B, 2, H, W) sparse observations
        mask: (B, 2, H, W) observation mask

    Returns:
        x_next: (B, 2, H, W) next state
    """
    B = x.shape[0]
    t_tensor = torch.full((B,), t, device=x.device, dtype=x.dtype)
    v = model(x, t_tensor, y_obs, mask)
    return x + v * dt


@torch.no_grad()
def heun_step(
    model: nn.Module,
    x: torch.Tensor,
    t: float,
    dt: float,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Single Heun (2nd-order RK) ODE step.

    Heun is a predictor-corrector method:
        x̃ = x_t + v(x_t, t) * dt            (Euler predictor)
        x_{t+dt} = x_t + 0.5*(v(x_t,t) + v(x̃, t+dt)) * dt   (corrector)
    """
    B = x.shape[0]
    t_tensor = torch.full((B,), t, device=x.device, dtype=x.dtype)

    # Euler predictor
    v1 = model(x, t_tensor, y_obs, mask)
    x_tilde = x + v1 * dt

    # Corrector
    t_next = torch.full((B,), t + dt, device=x.device, dtype=x.dtype)
    v2 = model(x_tilde, t_next, y_obs, mask)

    return x + 0.5 * (v1 + v2) * dt


def get_solver(name: str):
    """Get the ODE step function by name."""
    solvers = {"euler": euler_step, "heun": heun_step}
    if name not in solvers:
        raise ValueError(f"Unknown solver '{name}'. Choose from {list(solvers.keys())}")
    return solvers[name]
