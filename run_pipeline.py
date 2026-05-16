#!/usr/bin/env python3
"""
PerFlow 完整训练+推理 Pipeline (跨平台 Python 脚本)

用法:
    conda activate crowd-diffusion
    python run_pipeline.py

所有输出将写入 output.txt 并同时打印到终端。
GPU 数量自动检测: 优先使用所有可用 GPU 训练，推理使用单卡。
"""

import subprocess
import sys
import time
from datetime import datetime


def log(msg: str, f, tee: bool = True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    f.write(line + "\n")
    f.flush()
    if tee:
        print(line, flush=True)


def run_step(step_name: str, cmd: list, f, tee: bool = True) -> bool:
    log(f"----- Step: {step_name} -----", f, tee)
    log(f"Command: {' '.join(cmd)}", f, tee)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output in real-time
    returncode = None
    stdout_lines = []
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            stdout_lines.append(line)
            f.write(line)
            f.flush()
            if tee:
                print(line, end="", flush=True)
        process.stdout.close()
    returncode = process.wait()

    if returncode != 0:
        log(f"ERROR: {step_name} failed with exit code {returncode}", f, tee)
        # Show last 10 lines for quick diagnosis
        stderr_lines = [l for l in stdout_lines if l.strip()]
        tail = stderr_lines[-10:]
        log("--- stderr tail (last 10 lines) ---", f, True)
        for line in tail:
            log(f"  {line.strip()}", f, True)
        return False

    log(f"OK: {step_name} completed", f, tee)
    return True


def main():
    tee = True  # Print to terminal AND write to file

    with open("output.txt", "w", encoding="utf-8") as f:
        log("====== PerFlow Pipeline Started ======", f, tee)

        # ---- Step 1: Training ----
        # Auto-detect GPUs, use all available for training
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 4:
            gpu_ids = "0 1 2 3"
        elif num_gpus >= 2:
            gpu_ids = " ".join(str(i) for i in range(num_gpus))
        elif num_gpus == 1:
            gpu_ids = "0"
        else:
            gpu_ids = ""  # CPU mode

        train_cmd = [sys.executable, "main.py", "--mode", "train"]
        if gpu_ids:
            train_cmd += ["--gpu-ids"] + gpu_ids.split()
        log(f"Detected {num_gpus} GPU(s), using IDs: {gpu_ids or 'CPU'}", f, tee)

        if not run_step("1/3: Training", train_cmd, f, tee):
            log("====== Pipeline ABORTED (training failed) ======", f, tee)
            sys.exit(1)

        # ---- Step 2: Reconstruction ----
        rec_cmd = [
            sys.executable, "main.py", "--mode", "reconstruct",
            "--checkpoint", "perflow_final.pt",
        ]
        if gpu_ids:
            rec_cmd += ["--gpu-ids", gpu_ids.split()[0]]

        if not run_step("2/3: Reconstruction", rec_cmd, f, tee):
            log("====== Pipeline ABORTED (reconstruction failed) ======", f, tee)
            sys.exit(1)

        # ---- Step 3: Uncertainty Quantification ----
        uq_cmd = [
            sys.executable, "main.py", "--mode", "uq",
            "--checkpoint", "perflow_final.pt",
            "--num_samples", "10",
        ]
        if gpu_ids:
            uq_cmd += ["--gpu-ids", gpu_ids.split()[0]]

        if not run_step("3/3: Uncertainty Quantification", uq_cmd, f, tee):
            log("====== Pipeline ABORTED (UQ failed) ======", f, tee)
            sys.exit(1)

        log("====== PerFlow Pipeline Completed Successfully ======", f, tee)

    print(f"\nAll output written to: output.txt")


if __name__ == "__main__":
    main()
