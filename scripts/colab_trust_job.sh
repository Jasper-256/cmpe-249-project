#!/bin/bash
# Remote job for track B on a Colab VM: set up, then train the runs listed in
# /content/remaining.txt (one run name per line). Started detached by
# scripts/colab_watchdog.sh. Synthetic data only; real data never leaves the laptop.
set -uo pipefail
COMMIT="$1"
bash /content/colab_bootstrap.sh "$COMMIT"
cd /content/repo
uv run python scripts/run_matrix.py configs/experiments/trust_v1.yaml \
  --only $(tr -d '\r' < /content/remaining.txt)
echo "trust job done $(date)"
