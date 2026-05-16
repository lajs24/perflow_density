#!/usr/bin/env python3
"""PerFlow: Physics-embedded Rectified Flow for velocity field reconstruction.

Usage:
    python main.py --mode train --config configs/config.yaml
    python main.py --mode reconstruct --checkpoint checkpoint.pt --config configs/config.yaml
    python main.py --mode uq --checkpoint checkpoint.pt --num_samples 20
"""

import argparse
import yaml
import torch
import numpy as np

from models.perflow import PerFlowModel
from data.dataset import create_dataloader, generate_sparse_observation, velocity_field_from_trajectories, compute_velocities, parse_trajectories
from training.trainer import PerFlowTrainer
from inference.reconstruct import reconstruct, reconstruct_with_uncertainty
from inference.solver import get_solver
from utils.viz import plot_reconstruction, plot_training_curves


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> PerFlowModel:
    model_cfg = cfg["model"]
    return PerFlowModel(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["base_channels"],
        channel_multipliers=model_cfg["channel_multipliers"],
        num_res_blocks=model_cfg["num_res_blocks"],
        num_heads=model_cfg["num_heads"],
        constraint_type=model_cfg["constraint_type"],
    )


def train_mode(args, cfg: dict):
    """Run PerFlow training."""
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Resolve GPU IDs
    gpu_ids = args.gpu_ids
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
    device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
    print(f"Using GPUs: {gpu_ids}, primary device: {device}")

    # Coerce numeric config values (YAML 1.1 may parse e.g. "1e-4" as str)
    lr = float(train_cfg["learning_rate"])
    warmup = int(train_cfg["warmup_steps"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    num_epochs = int(train_cfg["num_epochs"])
    batch_size = int(train_cfg["batch_size"])

    # Data
    dataloader = create_dataloader(
        tracers_path=data_cfg["tracers_path"],
        grid_res=tuple(data_cfg["grid_resolution"]),
        x_range=tuple(data_cfg["x_range"]),
        y_range=tuple(data_cfg["y_range"]),
        obs_ratio=data_cfg["obs_ratio"],
        mask_type=data_cfg["mask_type"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        time_window=data_cfg.get("time_window", 150),
        file_pattern=data_cfg.get("file_pattern", "*"),
        max_samples=data_cfg.get("max_samples", None),
    )
    print(f"Dataset size: {len(dataloader.dataset)} samples")

    # Model
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    total_steps = steps_per_epoch * num_epochs
    trainer = PerFlowTrainer(
        model=model,
        dataloader=dataloader,
        learning_rate=lr,
        warmup_steps=warmup,
        total_steps=total_steps,
        device_ids=gpu_ids,
    )

    # Training
    losses = trainer.train(num_epochs=num_epochs)

    # Save final model
    torch.save(trainer.raw_model.state_dict(), "perflow_final.pt")
    print("Final model saved: perflow_final.pt")

    # Plot training curves
    plot_training_curves(losses, save_path="training_loss.png")
    print(f"Training complete. Final loss: {losses[-1]:.6f}")


def load_checkpoint(model, checkpoint_path: str, device: str):
    """Load checkpoint, stripping DataParallel 'module.' prefix if needed."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Support both raw and DataParallel-wrapped state dicts
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)
    model.to(device)


def reconstruct_mode(args, cfg: dict):
    """Run reconstruction from sparse observations."""
    model = build_model(cfg)
    gpu_ids = args.gpu_ids or ([0] if torch.cuda.is_available() else None)
    device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
    load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint} on {device}")

    # Load a sample from the dataset
    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]

    trajs = parse_trajectories(data_cfg["tracers_path"])
    vel_data = compute_velocities(trajs)
    field, _ = velocity_field_from_trajectories(
        vel_data,
        grid_res=tuple(data_cfg["grid_resolution"]),
        x_range=tuple(data_cfg["x_range"]),
        y_range=tuple(data_cfg["y_range"]),
    )

    # Generate sparse observation
    y_obs, mask = generate_sparse_observation(
        field, obs_ratio=data_cfg["obs_ratio"], mask_type=data_cfg["mask_type"]
    )

    y_obs_t = torch.from_numpy(y_obs).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    # Reconstruct
    x_1 = reconstruct(
        model, y_obs_t, mask_t,
        num_steps=infer_cfg["num_steps"],
        solver=infer_cfg["solver"],
    )

    # Visualize
    result = x_1.squeeze(0).cpu().numpy()
    plot_reconstruction(
        ground_truth=field,
        sparse_obs=y_obs,
        reconstruction=result,
        save_path="reconstruction_result.png",
    )
    print("Reconstruction saved: reconstruction_result.png")


def uq_mode(args, cfg: dict):
    """Reconstruction with uncertainty quantification."""
    model = build_model(cfg)
    gpu_ids = args.gpu_ids or ([0] if torch.cuda.is_available() else None)
    device = f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
    load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint} on {device}")

    data_cfg = cfg["data"]
    infer_cfg = cfg["inference"]
    num_samples = args.num_samples or infer_cfg["num_uq_samples"]

    # Load sample
    trajs = parse_trajectories(data_cfg["tracers_path"])
    vel_data = compute_velocities(trajs)
    field, _ = velocity_field_from_trajectories(
        vel_data,
        grid_res=tuple(data_cfg["grid_resolution"]),
        x_range=tuple(data_cfg["x_range"]),
        y_range=tuple(data_cfg["y_range"]),
    )
    y_obs, mask = generate_sparse_observation(
        field, obs_ratio=data_cfg["obs_ratio"], mask_type=data_cfg["mask_type"]
    )

    y_obs_t = torch.from_numpy(y_obs).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    # UQ reconstruction
    uq_result = reconstruct_with_uncertainty(
        model, y_obs_t, mask_t,
        num_steps=infer_cfg["num_steps"],
        solver=infer_cfg["solver"],
        num_samples=num_samples,
    )

    mean = uq_result["mean"].squeeze(0).cpu().numpy()
    std = uq_result["std"].squeeze(0).cpu().numpy()

    plot_reconstruction(
        ground_truth=field,
        sparse_obs=y_obs,
        reconstruction=mean,
        uncertainty=std,
        save_path="uq_reconstruction.png",
    )
    print(f"UQ reconstruction saved: uq_reconstruction.png (std averaged over {num_samples} samples)")


def main():
    parser = argparse.ArgumentParser(description="PerFlow: Physics-embedded Rectified Flow")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "reconstruct", "uq"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of UQ samples (overrides config)")

    # Multi-GPU: e.g. --gpu-ids 0 1 2 3
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                        help="GPU IDs to use for multi-GPU training (e.g. 0 1 2 3)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "train":
        train_mode(args, cfg)
    elif args.mode == "reconstruct":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for reconstruct mode")
        reconstruct_mode(args, cfg)
    elif args.mode == "uq":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for uq mode")
        uq_mode(args, cfg)


if __name__ == "__main__":
    main()
