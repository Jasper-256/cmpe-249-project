#!/usr/bin/env bash
# G6: freeze checkpoints, evaluate every frozen model once on the real test, run
# the FRESH-style geometric baseline, the label-free clipping-prevalence check,
# then the pre-registered analysis under the primary (EPnP) and secondary
# (RANSAC) solvers. Safe to rerun: completed real reports are never
# re-evaluated, and the freeze is written once.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=checkpoints/g6.log
echo "G6 start $(date)" | tee -a "$LOG"

if [ ! -f outputs/real/freeze_manifest.json ]; then
  # A refused freeze (incomplete matrix) must stop G6 before any real read.
  $PY scripts/freeze_manifest.py configs/experiments/main_v1.yaml \
    configs/experiments/finetune_v1.yaml 2>&1 | tee -a "$LOG"
  [ -f outputs/real/freeze_manifest.json ] || { echo "freeze refused; stopping" | tee -a "$LOG"; exit 1; }
fi

# Windows Python writes CRLF; strip it so bash passes clean paths.
$PY -c "import json;print('\n'.join(r['run_dir'] for r in json.load(open('outputs/real/freeze_manifest.json'))['runs'].values()))" \
  | tr -d '\r' > checkpoints/g6_runs.txt
while IFS= read -r run_dir; do
  [ -n "$run_dir" ] || continue
  name=$(basename "$run_dir")
  [ -f "outputs/real/$name/real_report.json" ] && continue
  echo "evaluate $name $(date +%T)" | tee -a "$LOG"
  $PY -W ignore scripts/evaluate_real.py "$run_dir" --confirm-frozen < /dev/null >> "$LOG" 2>&1 \
    || echo "FAILED $name" | tee -a "$LOG"
done < checkpoints/g6_runs.txt

$PY scripts/geometric_fusion.py real < /dev/null >> "$LOG" 2>&1 || echo "FAILED geometric real" | tee -a "$LOG"
$PY scripts/real_clipping_prevalence.py < /dev/null 2>&1 | tee -a "$LOG" || echo "FAILED prevalence" | tee -a "$LOG"

# Completeness: every frozen run (and the geometric pairs) must have a real report.
missing=0
while IFS= read -r run_dir; do
  [ -n "$run_dir" ] || continue
  [ -f "outputs/real/$(basename "$run_dir")/real_report.json" ] || { echo "MISSING $(basename "$run_dir")" | tee -a "$LOG"; missing=$((missing+1)); }
done < checkpoints/g6_runs.txt
echo "completeness: $missing frozen runs without a real report" | tee -a "$LOG"

for condition in primary primary_ransac; do
  $PY scripts/analyze_real.py configs/experiments/main_v1_comparisons.yaml \
    --runs-dir outputs/runs/main-hrnet outputs/runs/finetune-hrnet \
    --condition "$condition" --output "outputs/real/analysis_${condition}.json" < /dev/null 2>&1 \
    | tee -a "$LOG" || echo "FAILED analysis $condition" | tee -a "$LOG"
done
echo "G6 done $(date)" | tee -a "$LOG"
