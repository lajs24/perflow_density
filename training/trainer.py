"""Training loop for PerFlow (Rectified Flow training) with multi-GPU support."""

import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.perflow import unwrap_model
from utils.paths import OUTPUT_DIR, ensure_output_dir


class PerFlowTrainer:
    """Handles the Rectified Flow training loop.

    Core idea: sample t ~ U[0,1], construct linear interpolation x_t = (1-t)x_0 + t*x_1,
    predict velocity v = dx/dt = x_1 - x_0, minimize MSE.

    Multi-GPU: uses nn.DataParallel when multiple GPU IDs are provided.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        total_steps: int = 100_000,
        device_ids: list | None = None,
    ):
        self.device_ids = device_ids or [0]
        self.device = f"cuda:{self.device_ids[0]}" if torch.cuda.is_available() else "cpu"

        # Move model to base device first
        model = model.to(self.device)

        # Wrap with DataParallel if multiple GPUs
        if torch.cuda.is_available() and len(self.device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids=self.device_ids)
            print(f"DataParallel activated on GPUs: {device_ids}")
        else:
            self.model = model

        # Keep unwrapped model reference for projector access
        self.raw_model = unwrap_model(self.model)

        self.dataloader = dataloader

        self.optimizer = torch.optim.AdamW(self.raw_model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Cosine schedule with linear warmup
        warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

        self.loss_fn = nn.MSELoss()
        self.step = 0

    def train_epoch(self, epoch: int = 0, num_epochs: int = 0) -> float:
        """Train for one epoch. Returns average loss."""
        self.raw_model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch", ncols=100)
        for batch in pbar:
            # Move to device
            x_1 = batch["field"].to(self.device)   # (B, 2, H, W) ground truth
            y_obs = batch["y_obs"].to(self.device)  # (B, 2, H, W) sparse obs
            mask = batch["mask"].to(self.device)     # (B, 2, H, W) obs mask
            B = x_1.shape[0]

            # 1. Generate physically admissible noise x_0
            noise = torch.randn_like(x_1)
            target_mean = x_1.mean(dim=[-1, -2], keepdim=True)
            x_0 = self.raw_model.project_state(noise, target_mean)

            # 2. Sample random time steps t ~ U[0, 1]
            t = torch.rand(B, device=self.device)

            # 3. Rectified Flow linear interpolation
            t_flat = t.view(B, 1, 1, 1)
            x_t = (1.0 - t_flat) * x_0 + t_flat * x_1

            # 4. Target velocity (constant straight path)
            v_target = x_1 - x_0

            # 5. Model prediction with physics constraint (DataParallel handles multi-GPU)
            v_pred = self.model(x_t, t, y_obs, mask)

            # 6. Loss
            loss = self.loss_fn(v_pred, v_target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            total_loss += loss.item()
            num_batches += 1
            elapsed = time.time() - start_time
            mem = ""
            if torch.cuda.is_available():
                mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
                mem = f"{mem_gb:.1f}GB"
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                "mem": mem,
                "elapsed": f"{elapsed:.0f}s",
            })

        return total_loss / max(num_batches, 1)

    def train(self, num_epochs: int) -> list:
        """Run full training. Returns list of epoch losses."""
        total_start = time.time()
        losses = []
        print(f"\n{'='*60}")
        print(f"  Training started: {num_epochs} epochs, {len(self.dataloader)} batches/epoch")
        print(f"  Total steps: {num_epochs * len(self.dataloader)}")
        print(f"{'='*60}\n")
        for epoch in range(num_epochs):
            epoch_start = time.time()
            loss = self.train_epoch(epoch=epoch + 1, num_epochs=num_epochs)
            losses.append(loss)
            epoch_time = time.time() - epoch_start
            total_time = time.time() - total_start
            lr_now = self.scheduler.get_last_lr()[0]
            print(
                f"  ✓ Epoch {epoch + 1}/{num_epochs}  "
                f"loss={loss:.6f}  "
                f"lr={lr_now:.2e}  "
                f"time={epoch_time:.0f}s  "
                f"total={total_time:.0f}s"
            )

            # Save checkpoint every 25 epochs
            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(epoch + 1, loss)
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"  Training completed in {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Final loss: {losses[-1]:.6f}" if losses else "  No training data")
        print(f"{'='*60}\n")
        return losses

    def save_checkpoint(self, epoch: int, loss: float):
        ensure_output_dir()
        path = OUTPUT_DIR / f"checkpoint_epoch{epoch}_loss{loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "step": self.step,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, path)
        print(f"Checkpoint saved: {path}")
