"""Visualization utilities for PerFlow: velocity fields, uncertainty maps, comparison plots."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_velocity_field(ax, v: np.ndarray, title: str = "", stride: int = 4):
    """Plot a 2D velocity field as a quiver plot.

    Args:
        ax: matplotlib axis
        v: (2, H, W) velocity field (vx, vy)
        title: plot title
        stride: downsampling stride for quiver arrows
    """
    H, W = v.shape[1:]
    y, x = np.mgrid[0:H, 0:W]
    ax.quiver(
        x[::stride, ::stride],
        y[::stride, ::stride],
        v[0, ::stride, ::stride],
        v[1, ::stride, ::stride],
        scale_units="inches",
        scale=10,
        alpha=0.8,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()


def plot_reconstruction(
    ground_truth: np.ndarray,
    sparse_obs: np.ndarray,
    reconstruction: np.ndarray,
    uncertainty: np.ndarray | None = None,
    save_path: str | None = None,
):
    """Comparison plot: ground truth vs sparse obs vs reconstruction.

    Args:
        ground_truth: (2, H, W) true velocity field
        sparse_obs: (2, H, W) sparse observation
        reconstruction: (2, H, W) reconstructed field
        uncertainty: (2, H, W) optional uncertainty map
        save_path: optional path to save figure
    """
    n_panels = 4 if uncertainty is not None else 3
    fig, axes = plt.subplots(2, n_panels, figsize=(5 * n_panels, 10))

    # Magnitude fields (row 0)
    v_mag_gt = np.sqrt(ground_truth[0] ** 2 + ground_truth[1] ** 2)
    v_mag_rec = np.sqrt(reconstruction[0] ** 2 + reconstruction[1] ** 2)

    # Row 0: magnitude maps
    im0 = axes[0, 0].imshow(v_mag_gt, cmap="viridis")
    axes[0, 0].set_title("Ground Truth Magnitude")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(np.sqrt(sparse_obs[0] ** 2 + sparse_obs[1] ** 2), cmap="viridis")
    axes[0, 1].set_title("Sparse Observation Magnitude")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(v_mag_rec, cmap="viridis")
    axes[0, 2].set_title("Reconstruction Magnitude")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 1: quiver plots
    plot_velocity_field(axes[1, 0], ground_truth, "Ground Truth Flow")
    plot_velocity_field(axes[1, 1], sparse_obs, "Sparse Observation Flow")
    plot_velocity_field(axes[1, 2], reconstruction, "Reconstructed Flow")

    if uncertainty is not None:
        unc_mag = np.sqrt(uncertainty[0] ** 2 + uncertainty[1] ** 2)
        im3 = axes[0, 3].imshow(unc_mag, cmap="hot")
        axes[0, 3].set_title("Uncertainty Magnitude (Std)")
        plt.colorbar(im3, ax=axes[0, 3])
        axes[1, 3].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")
    plt.close()


def plot_training_curves(losses: list, save_path: str | None = None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("PerFlow Training Loss")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss plot saved: {save_path}")
    plt.close()
