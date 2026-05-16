@echo off
REM ============================================================
REM PerFlow 完整训练+推理 脚本 (Windows)
REM 用法: 先手动激活 conda 环境，然后运行此脚本
REM     conda activate crowd-diffusion
REM     .\run_windows.bat
REM ============================================================

echo [%date% %time%] ====== PerFlow Pipeline Started ====== >  output.txt 2>&1

REM ---- Step 1: 训练 ----
echo [%date% %time%] ----- Step 1/3: Training ----- >> output.txt 2>&1
python main.py --mode train --gpu-ids 0 >> output.txt 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] ERROR: Training failed! >> output.txt 2>&1
    exit /b %errorlevel%
)

REM ---- Step 2: 重建 ----
echo [%date% %time%] ----- Step 2/3: Reconstruction ----- >> output.txt 2>&1
python main.py --mode reconstruct --checkpoint perflow_final.pt --gpu-ids 0 >> output.txt 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] ERROR: Reconstruction failed! >> output.txt 2>&1
    exit /b %errorlevel%
)

REM ---- Step 3: 不确定性量化 ----
echo [%date% %time%] ----- Step 3/3: Uncertainty Quantification ----- >> output.txt 2>&1
python main.py --mode uq --checkpoint perflow_final.pt --num_samples 10 --gpu-ids 0 >> output.txt 2>&1
if %errorlevel% neq 0 (
    echo [%date% %time%] ERROR: UQ failed! >> output.txt 2>&1
    exit /b %errorlevel%
)

echo [%date% %time%] ====== PerFlow Pipeline Completed Successfully ====== >> output.txt 2>&1
