#!/usr/bin/env bash
# G6: freeze checkpoints, evaluate every frozen model once on the real test, run
# the FRESH-style geometric baseline, then the pre-registered analysis under the
# primary (EPnP) and secondary (RANSAC) solvers. Safe to rerun: completed real
# reports are never re-evaluated, and the freeze is written once.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=checkpoints/g6.log
echo "G6 start $(date)" | tee -a "$LOG"

if [ ! -f outputs/real/freeze_manifest.json ]; then
  $PY scripts/freeze_manifest.py configs/experiments/main_v1.yaml configs/experiments/finetune_v1.yaml \
    2>&1 | tee -a "$LOG"
fi

# Evaluate every frozen run once (one process per run keeps memory bounded).
$PY - <<'PYEOF' > checkpoints/g6_runs.txt
import json
manifest = json.load(open("outputs/real/freeze_manifest.json"))
print("\n".join(run["run_dir"] for run in manifest["runs"].values()))
PYEOF
while read -r run_dir; do
  name=$(basename "$run_dir")
  if [ -f "outputs/real/$name/real_report.json" ]; then continue; fi
  echo "evaluate $name $(date +%T)" | tee -a "$LOG"
  $PY -W ignore scripts/evaluate_real.py "$run_dir" --confirm-frozen >> "$LOG" 2>&1 \
    || echo "FAILED $name" | tee -a "$LOG"
done < checkpoints/g6_runs.txt

$PY scripts/geometric_fusion.py real >> "$LOG" 2>&1

for condition in primary primary_ransac; do
  $PY scripts/analyze_real.py configs/experiments/main_v1_comparisons.yaml \
    --runs-dir outputs/runs/main-hrnet outputs/runs/finetune-hrnet \
    --condition "$condition" --output "outputs/real/analysis_${condition}.json" 2>&1 | tee -a "$LOG"
done
echo "G6 done $(date)" | tee -a "$LOG"
