#!/usr/bin/env python3
"""
PerFlow 完整训练+推理 Pipeline (跨平台 Python 脚本)

用法:
    python run_pipeline.py                          # 完整运行 3 步
    python run_pipeline.py --start-from 2           # 从第 2 步开始（跳过训练）
    python run_pipeline.py --skip 1                 # 跳过第 1 步
    python run_pipeline.py --skip 1 3               # 跳过第 1 步和第 3 步
    python run_pipeline.py --checkpoint path.pt     # 指定已有 checkpoint
    python run_pipeline.py --output run.log         # 指定输出文件

所有输出将写入 output.txt（或指定文件）并同时打印到终端。
GPU 数量自动检测: 优先使用所有可用 GPU 训练，推理使用单卡。
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="PerFlow Pipeline Runner")
    parser.add_argument("--start-from", type=int, default=1, choices=[1, 2, 3],
                        help="Start from this step (skip all previous steps)")
    parser.add_argument("--skip", type=int, nargs="+", default=[],
                        help="Skip specific step numbers (e.g. --skip 1 3)")
    parser.add_argument("--checkpoint", type=str, default="perflow_final.pt",
                        help="Path to existing checkpoint (for steps 2/3)")
    parser.add_argument("--output", type=str, default="output.txt",
                        help="Output log file (default: output.txt)")
    return parser.parse_args()


def main():
    args = parse_args()
    tee = True
    skip_set = set(args.skip)
    start_from = args.start_from
    output_file = args.output

    with open(output_file, "a", encoding="utf-8") as f:
        log("====== PerFlow Pipeline Started ======", f, tee)

        # Auto-detect GPUs
        import torch
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 4:
            gpu_ids = "0 1 2 3"
        elif num_gpus >= 2:
            gpu_ids = " ".join(str(i) for i in range(num_gpus))
        elif num_gpus == 1:
            gpu_ids = "0"
        else:
            gpu_ids = ""
        log(f"Detected {num_gpus} GPU(s), using IDs: {gpu_ids or 'CPU'}", f, tee)

        # ---- Step 1: Training ----
        if start_from <= 1 and 1 not in skip_set:
            train_cmd = [sys.executable, "main.py", "--mode", "train"]
            if gpu_ids:
                train_cmd += ["--gpu-ids"] + gpu_ids.split()
            if not run_step("1/3: Training", train_cmd, f, tee):
                log("====== Pipeline ABORTED (training failed) ======", f, tee)
                sys.exit(1)
        else:
            log("1/3: Training skipped", f, tee)

        # ---- Step 2: Reconstruction ----
        if start_from <= 2 and 2 not in skip_set:
            rec_cmd = [
                sys.executable, "main.py", "--mode", "reconstruct",
                "--checkpoint", args.checkpoint,
            ]
            if gpu_ids:
                rec_cmd += ["--gpu-ids", gpu_ids.split()[0]]
            if not run_step("2/3: Reconstruction", rec_cmd, f, tee):
                log("====== Pipeline ABORTED (reconstruction failed) ======", f, tee)
                sys.exit(1)
        else:
            log("2/3: Reconstruction skipped", f, tee)

        # ---- Step 3: Uncertainty Quantification ----
        if start_from <= 3 and 3 not in skip_set:
            uq_cmd = [
                sys.executable, "main.py", "--mode", "uq",
                "--checkpoint", args.checkpoint,
                "--num_samples", "10",
            ]
            if gpu_ids:
                uq_cmd += ["--gpu-ids", gpu_ids.split()[0]]
            if not run_step("3/3: Uncertainty Quantification", uq_cmd, f, tee):
                log("====== Pipeline ABORTED (UQ failed) ======", f, tee)
                sys.exit(1)
        else:
            log("3/3: Uncertainty Quantification skipped", f, tee)

        log("====== PerFlow Pipeline Completed Successfully ======", f, tee)

    print(f"\nAll output written to: {output_file}")


if __name__ == "__main__":
    main()
