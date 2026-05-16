#!/bin/bash
# ============================================================
# PerFlow 完整训练+推理 脚本 (Linux)
# 用法: 先手动激活 conda 环境，然后运行此脚本
#     conda activate crowd-diffusion
#     bash run_linux.sh
# ============================================================
# 或者: chmod +x run_linux.sh && ./run_linux.sh
# ============================================================

OUTPUT="output.txt"

echo "[$(date)] ====== PerFlow Pipeline Started ======" | tee "$OUTPUT"

# ---- Step 1: 训练 (4 GPUs) ----
echo "[$(date)] ----- Step 1/3: Training (GPUs 0,1,2,3) -----" | tee -a "$OUTPUT"
python main.py --mode train --gpu-ids 0 1 2 3 >> "$OUTPUT" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Training failed!" | tee -a "$OUTPUT"
    exit 1
fi

# ---- Step 2: 重建 (1 GPU) ----
echo "[$(date)] ----- Step 2/3: Reconstruction -----" | tee -a "$OUTPUT"
python main.py --mode reconstruct --checkpoint perflow_final.pt --gpu-ids 0 >> "$OUTPUT" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Reconstruction failed!" | tee -a "$OUTPUT"
    exit 1
fi

# ---- Step 3: 不确定性量化 (1 GPU) ----
echo "[$(date)] ----- Step 3/3: Uncertainty Quantification (10 samples) -----" | tee -a "$OUTPUT"
python main.py --mode uq --checkpoint perflow_final.pt --num_samples 10 --gpu-ids 0 >> "$OUTPUT" 2>&1
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: UQ failed!" | tee -a "$OUTPUT"
    exit 1
fi

echo "[$(date)] ====== PerFlow Pipeline Completed Successfully ======" | tee -a "$OUTPUT"
