#!/usr/bin/env bash
# Real evaluation of the sim2real-v1 study: wait for the matrix, freeze, evaluate
# each frozen checkpoint once, then analyse under both pose solvers. Safe to rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
OUT=outputs/real-v2
LOG=checkpoints/real_v2.log
mkdir -p "$OUT"
until grep -q "sim2real matrix done" checkpoints/chain_round2.log 2>/dev/null; do sleep 60; done
echo "real-v2 start $(date)" | tee -a "$LOG"
if [ ! -f "$OUT/freeze_manifest.json" ]; then
  $PY scripts/freeze_manifest.py configs/experiments/sim2real_v1.yaml --output "$OUT/freeze_manifest.json" 2>&1 | tee -a "$LOG"
  [ -f "$OUT/freeze_manifest.json" ] || { echo "freeze refused; stopping" | tee -a "$LOG"; exit 1; }
fi
$PY -c "import json;print('\n'.join(r['run_dir'] for r in json.load(open('$OUT/freeze_manifest.json'))['runs'].values()))" \
  | tr -d '\r' > checkpoints/real_v2_runs.txt
while IFS= read -r run_dir; do
  [ -n "$run_dir" ] || continue
  name=$(basename "$run_dir")
  [ -f "$OUT/$name/real_report.json" ] && continue
  echo "evaluate $name $(date +%T)" | tee -a "$LOG"
  $PY -W ignore scripts/evaluate_real.py "$run_dir" --confirm-frozen --manifest "$OUT/freeze_manifest.json" \
    --output-dir "$OUT" < /dev/null >> "$LOG" 2>&1 || echo "FAILED $name" | tee -a "$LOG"
done < checkpoints/real_v2_runs.txt
missing=0
while IFS= read -r run_dir; do
  [ -n "$run_dir" ] || continue
  [ -f "$OUT/$(basename "$run_dir")/real_report.json" ] || { echo "MISSING $(basename "$run_dir")" | tee -a "$LOG"; missing=$((missing+1)); }
done < checkpoints/real_v2_runs.txt
echo "completeness: $missing frozen runs without a real report" | tee -a "$LOG"
for condition in primary primary_ransac; do
  $PY scripts/analyze_real.py configs/experiments/sim2real_v1_comparisons.yaml \
    --real-dir "$OUT" --runs-dir outputs/runs/sim2real-hrnet \
    --condition "$condition" --output "$OUT/analysis_${condition}.json" < /dev/null 2>&1 \
    | tee -a "$LOG" || echo "FAILED analysis $condition" | tee -a "$LOG"
done
echo "real-v2 done $(date)" | tee -a "$LOG"
