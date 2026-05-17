"""Shared output directory paths for PerFlow training artifacts."""

from pathlib import Path

# All generated files (checkpoints, logs, plots) go to perflow_tmp/
# at the same level as the project root.
# Server:   ~/perflow_tmp
# Local:    d:/VSCode-New/mainWorkspace/perflow_tmp
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "perflow_tmp"


def ensure_output_dir() -> Path:
    """Create the output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
