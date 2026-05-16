"""Data pipeline: trajectory parsing → velocity grid fields + sparse observation masks."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm


def parse_trajectories(tracers_path: str) -> dict:
    """Parse trajectory file into {id: [(frame, x, y, t), ...]}.

    Supports both column formats:
      - LargeView:  id, frame, x, y, z, t, x_RGF, y_RGF         (8 cols)
      - TopView:    id, frame, x, y, z, id_global, t, x_RGF, y_RGF  (9 cols)
    The time column is always 3rd from the end (index -3) in both.
    """
    trajs = {}
    with open(tracers_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split()
            pid = int(parts[0])
            frame = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            # t is 3rd from end: index -3 works for both 8-col and 9-col formats
            t = float(parts[-3])
            trajs.setdefault(pid, []).append((frame, x, y, t))
    # Sort each trajectory by frame
    for pid in trajs:
        trajs[pid].sort(key=lambda row: row[0])
    return trajs


def compute_velocities(trajs: dict) -> dict:
    """Compute velocities from trajectory positions. Returns {id: [(x, y, vx, vy), ...]}."""
    vel_data = {}
    for pid, pts in trajs.items():
        out = []
        for i in range(len(pts) - 1):
            f0, x0, y0, t0 = pts[i]
            f1, x1, y1, t1 = pts[i + 1]
            dt = t1 - t0
            if dt <= 0:
                continue
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            out.append((x0, y0, vx, vy))
        if out:
            vel_data[pid] = out
    return vel_data


def velocity_field_from_trajectories(
    vel_data: dict,
    grid_res: Tuple[int, int] = (128, 128),
    x_range: Tuple[float, float] = (-20, 20),
    y_range: Tuple[float, float] = (-20, 20),
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert trajectory velocities to a regular grid velocity field.

    Returns:
        field: (2, H, W) velocity field (vx, vy)
        count: (H, W) number of samples per cell
    """
    H, W = grid_res
    vx_grid = np.zeros((H, W), dtype=np.float32)
    vy_grid = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    x_min, x_max = x_range
    y_min, y_max = y_range

    for pid, pts in vel_data.items():
        for x, y, vx, vy in pts:
            col = int((x - x_min) / (x_max - x_min) * W)
            row = int((y - y_min) / (y_max - y_min) * H)
            if 0 <= row < H and 0 <= col < W:
                vx_grid[row, col] += vx
                vy_grid[row, col] += vy
                count[row, col] += 1.0

    mask_valid = count > 0
    vx_grid[mask_valid] /= count[mask_valid]
    vy_grid[mask_valid] /= count[mask_valid]

    field = np.stack([vx_grid, vy_grid], axis=0)  # (2, H, W)
    return field, count


def generate_sparse_observation(
    field: np.ndarray, obs_ratio: float = 0.1, mask_type: str = "random"
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sparse observation y_obs and binary mask M.

    Args:
        field: (2, H, W) ground truth velocity field
        obs_ratio: fraction of cells to observe
        mask_type: "random" or "fixed"

    Returns:
        y_obs: (2, H, W) observed values (0 where unobserved)
        mask: (2, H, W) binary mask (1 = observed, 0 = unobserved)
    """
    C, H, W = field.shape
    mask = np.zeros((H, W), dtype=np.float32)

    if mask_type == "random":
        total_cells = H * W
        n_obs = max(1, int(total_cells * obs_ratio))
        indices = np.random.choice(total_cells, n_obs, replace=False)
        mask.flat[indices] = 1.0
    else:
        stride = int(np.sqrt(1.0 / obs_ratio))
        mask[::stride, ::stride] = 1.0

    mask = np.stack([mask, mask], axis=0).astype(np.float32)
    y_obs = field * mask
    return y_obs, mask


def find_trajectory_files(data_dir: str, pattern: str = "*") -> List[str]:
    """Find all trajectory files in common locations.

    Args:
        data_dir: file or directory path
        pattern: filename filter (e.g. "TopView_1*" for scene-1).
                 Supports * and ? wildcards. "*" means no filter.
    """
    base = Path(data_dir)
    if base.is_file():
        return [str(base)]

    # Collect all trajectory-like .txt files
    candidates = sorted(base.rglob("*.txt"))
    # Filter by name matching the pattern
    import fnmatch
    files = []
    for f in candidates:
        if pattern == "*" or fnmatch.fnmatch(f.stem, pattern):
            files.append(str(f))
    return files


class VelocityFieldDataset(Dataset):
    """Dataset of velocity fields from trajectory data.

    Supports loading multiple trajectory files from a directory.
    Uses time windows to generate multiple training samples per file.
    """

    def __init__(
        self,
        tracers_path: str,
        grid_res: Tuple[int, int] = (128, 128),
        x_range: Tuple[float, float] = (-20, 20),
        y_range: Tuple[float, float] = (-20, 20),
        obs_ratio: float = 0.1,
        mask_type: str = "random",
        time_window: Optional[int] = 150,
        file_pattern: str = "*",
        max_samples: Optional[int] = None,
    ):
        self.fields = []

        # Resolve all trajectory files
        traj_files = find_trajectory_files(tracers_path, file_pattern)
        if not traj_files:
            raise FileNotFoundError(
                f"No trajectory files matching '{file_pattern}' in: {tracers_path}"
            )

        pbar = tqdm(
            traj_files,
            desc="Loading trajectory files",
            unit="file",
            ncols=100,
        )
        for fpath in pbar:
            if max_samples is not None and len(self.fields) >= max_samples:
                break
            pbar.set_postfix({"file": Path(str(fpath)).name, "samples": len(self.fields)})
            try:
                trajs = parse_trajectories(str(fpath))
                if not trajs:
                    continue
                vel_data = compute_velocities(trajs)
                if not vel_data:
                    continue
                n_traj = len(vel_data)
                n_pts = sum(len(v) for v in vel_data.values())

                if time_window is not None:
                    for pid, pts in vel_data.items():
                        for start in range(0, len(pts), time_window):
                            if max_samples is not None and len(self.fields) >= max_samples:
                                break
                            chunk = {pid: pts[start : start + time_window]}
                            field, _ = velocity_field_from_trajectories(
                                chunk, grid_res, x_range, y_range
                            )
                            self.fields.append(field)
                        if max_samples is not None and len(self.fields) >= max_samples:
                            break
                else:
                    field, _ = velocity_field_from_trajectories(
                        vel_data, grid_res, x_range, y_range
                    )
                    self.fields.append(field)

                pbar.set_postfix({
                    "file": Path(str(fpath)).name,
                    "trajs": n_traj,
                    "pts": n_pts,
                    "samples": len(self.fields),
                })
            except Exception as e:
                print(f"\n  Warning: skipping {Path(str(fpath)).name} ({e})")
                continue

        if not self.fields:
            raise RuntimeError("No valid velocity fields could be generated from the data!")

        print(f"  Generated {len(self.fields)} velocity field samples")
        if max_samples is not None and len(self.fields) >= max_samples:
            self.fields = self.fields[:max_samples]
            print(f"  Trimmed to {max_samples} samples (max_samples limit)")

        self.grid_res = grid_res
        self.x_range = x_range
        self.y_range = y_range
        self.obs_ratio = obs_ratio
        self.mask_type = mask_type

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        field = self.fields[idx]
        y_obs, mask = generate_sparse_observation(field, self.obs_ratio, self.mask_type)
        return {
            # Wrap in np.asarray first to handle numpy scalar types (compat with NumPy 2.x)
            "field": torch.as_tensor(np.asarray(field)),
            "y_obs": torch.as_tensor(np.asarray(y_obs)),
            "mask": torch.as_tensor(np.asarray(mask)),
        }


def create_dataloader(
    tracers_path: str,
    grid_res=(128, 128),
    x_range=(-20, 20),
    y_range=(-20, 20),
    obs_ratio=0.1,
    mask_type="random",
    batch_size=16,
    shuffle=True,
    num_workers=0,
    time_window=150,
    file_pattern="*",
    max_samples=None,
) -> DataLoader:
    dataset = VelocityFieldDataset(
        tracers_path=tracers_path,
        grid_res=grid_res,
        x_range=x_range,
        y_range=y_range,
        obs_ratio=obs_ratio,
        mask_type=mask_type,
        time_window=time_window,
        file_pattern=file_pattern,
        max_samples=max_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
