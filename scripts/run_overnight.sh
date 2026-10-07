#!/usr/bin/env bash
# Overnight queue: fine-tuning stage first, then the remaining main-matrix runs.
# Both runners skip runs that already have a report.json, and train.py resumes
# an interrupted run from its last completed epoch, so this is safe to rerun.
set -u
cd "$(dirname "$0")/.."
mkdir -p checkpoints/finetune-hrnet checkpoints/main-hrnet
echo "overnight start $(date)" | tee -a checkpoints/main-hrnet/runner.log
PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/run_matrix.py configs/experiments/finetune_v1.yaml \
  >> checkpoints/finetune-hrnet/runner.log 2>&1
PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/run_matrix.py configs/experiments/main_v1.yaml \
  >> checkpoints/main-hrnet/runner.log 2>&1
echo "overnight done $(date)" | tee -a checkpoints/main-hrnet/runner.log
