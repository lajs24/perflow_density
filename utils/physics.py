"""Physics utility functions: divergence-free projection, stream function, boundary conditions."""

import torch
import torch.nn.functional as F


def project_div_free_fft(v: torch.Tensor) -> torch.Tensor:
    """Project velocity field to be divergence-free using FFT (Helmholtz decomposition).

    Args:
        v: (B, 2, H, W) velocity field (vx, vy)

    Returns:
        v_div_free: (B, 2, H, W) divergence-free velocity field
    """
    B, C, H, W = v.shape
    assert C == 2, "Expected 2-channel velocity field"

    device = v.device
    dtype = v.dtype

    # Prepare wave numbers
    kx = torch.fft.fftfreq(W, d=1.0 / W).to(device=device, dtype=dtype)  # (W,)
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(device=device, dtype=dtype)  # (H,)

    # Use rfft2 for efficiency (real inputs)
    V = torch.fft.rfft2(v)  # (B, 2, H, W//2+1) complex

    # Build wave number grids
    KX = kx[None, None, None, :]   # (1, 1, 1, W)
    KY = ky[None, None, :, None]   # (1, 1, H, 1)

    # For rfft2, kx needs to be truncated to match the rfft dimension
    KX = KX[..., :W // 2 + 1]       # (1, 1, 1, W//2+1)
    KY = KY[..., :]                  # (1, 1, H, 1)

    # Squared wave number (avoid division by zero at k=0)
    k_sq = KX ** 2 + KY ** 2        # (1, 1, H, W//2+1)
    k_sq = torch.where(k_sq == 0, torch.tensor(1.0, device=device, dtype=dtype), k_sq)

    # Helmholtz projection: V_div_free = V - (k · V_hat) * k / |k|^2
    Vx = V[:, 0:1, :, :]  # (B, 1, H, W//2+1) complex
    Vy = V[:, 1:2, :, :]  # (B, 1, H, W//2+1) complex

    # k · V_hat
    k_dot_V = KX * Vx + KY * Vy  # (B, 1, H, W//2+1) complex

    # Project out the divergent component
    Vx_proj = Vx - k_dot_V * KX / k_sq  # (B, 1, H, W//2+1) complex
    Vy_proj = Vy - k_dot_V * KY / k_sq  # (B, 1, H, W//2+1) complex

    V_proj = torch.cat([Vx_proj, Vy_proj], dim=1)  # (B, 2, H, W//2+1) complex

    # Inverse FFT
    v_proj = torch.fft.irfft2(V_proj, s=(H, W))  # (B, 2, H, W)

    return v_proj


def stream_function_to_velocity(psi: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """Convert stream function to divergence-free velocity field.

    vx = dψ/dy, vy = -dψ/dx  (guarantees ∇·v = 0)

    Args:
        psi: (B, 1, H, W) stream function
        dx, dy: grid spacing

    Returns:
        v: (B, 2, H, W) divergence-free velocity field
    """
    # Central differences for gradient
    # dψ/dy (vx)
    psi_pad_y = F.pad(psi, (0, 0, 1, 1), mode="replicate")
    dpsi_dy = (psi_pad_y[:, :, 2:, :] - psi_pad_y[:, :, :-2, :]) / (2.0 * dy)

    # dψ/dx (vy, with negative sign)
    psi_pad_x = F.pad(psi, (1, 1, 0, 0), mode="replicate")
    dpsi_dx = (psi_pad_x[:, :, :, 2:] - psi_pad_x[:, :, :, :-2]) / (2.0 * dx)

    vx = dpsi_dy  # (B, 1, H, W)
    vy = -dpsi_dx  # (B, 1, H, W)

    return torch.cat([vx, vy], dim=1)


def apply_boundary_conditions(v: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
    """Enforce boundary conditions: zero velocity at walls/obstacles.

    Args:
        v: (B, 2, H, W) velocity field
        boundary_mask: (B, 1, H, W) or (H, W), 1 = boundary, 0 = interior

    Returns:
        v: (B, 2, H, W) with zero velocity at boundary cells
    """
    if boundary_mask.dim() == 2:
        boundary_mask = boundary_mask.unsqueeze(0).unsqueeze(0)
    elif boundary_mask.dim() == 3:
        boundary_mask = boundary_mask.unsqueeze(1)

    # Ensure binary
    boundary_mask = (boundary_mask > 0.5).float()
    interior_mask = 1.0 - boundary_mask

    return v * interior_mask


def compute_divergence(v: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """Compute divergence of a velocity field.

    Args:
        v: (B, 2, H, W) velocity field
        dx, dy: grid spacing

    Returns:
        div: (B, 1, H, W) divergence field
    """
    B, C, H, W = v.shape
    assert C == 2

    vx = v[:, 0:1]
    vy = v[:, 1:2]

    # dvx/dx
    vx_pad = F.pad(vx, (1, 1, 0, 0), mode="replicate")
    dvx_dx = (vx_pad[:, :, :, 2:] - vx_pad[:, :, :, :-2]) / (2.0 * dx)

    # dvy/dy
    vy_pad = F.pad(vy, (0, 0, 1, 1), mode="replicate")
    dvy_dy = (vy_pad[:, :, 2:, :] - vy_pad[:, :, :-2, :]) / (2.0 * dy)

    return dvx_dx + dvy_dy
