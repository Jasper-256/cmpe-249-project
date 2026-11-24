#!/bin/bash
# Local watchdog for track B (trust-v1) on Colab.
#
# Every PERIOD seconds: copy finished runs (report.json present) from the VM to
# this laptop; if the session is gone, start a new T4 session, bootstrap it and
# train only the runs that have no local report yet. Stops the session and exits
# when every run has a local report. Never touches other Colab sessions.
#
# Usage: bash scripts/colab_watchdog.sh <git-commit>
set -uo pipefail
export PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1
COMMIT="$1"
SESSION=trust
PERIOD=1800
MATRIX=configs/experiments/trust_v1.yaml
OUT=outputs/runs/trust-v1
CKPT=checkpoints/trust-v1
TMP=checkpoints/colab-sync
mkdir -p "$TMP" "$OUT" "$CKPT"
log() { echo "[$(date '+%F %T')] $*"; }

all_runs() {
  .venv/Scripts/python.exe -c "import yaml;[print(r['name']) for r in yaml.safe_load(open('$MATRIX'))['runs']]" | tr -d '\r'
}
remaining() { for n in $(all_runs); do [ -f "$OUT/$n/report.json" ] || echo "$n"; done; }
remote() {  # run a shell command on the VM, print stdout
  printf 'import subprocess\nprint(subprocess.run(%s, shell=True, capture_output=True, text=True).stdout)\n' \
    "$(printf '%s' "$1" | .venv/Scripts/python.exe -c 'import sys;print(repr(sys.stdin.read()))')" \
    | timeout 300 colab exec -s "$SESSION" 2>/dev/null | tr -d '\r'
}
alive() { colab sessions 2>/dev/null | grep -q "^\[$SESSION\]"; }

start_session() {
  log "starting new T4 session"
  colab stop -s "$SESSION" >/dev/null 2>&1
  timeout 600 colab new -s "$SESSION" --gpu T4 >/dev/null 2>&1 || return 1
  alive || return 1
  remaining > "$TMP/remaining.txt"
  colab upload -s "$SESSION" data/fresh/index/synthetic-split-v1.json /content/synthetic-split-v1.json >/dev/null 2>&1 || return 1
  colab upload -s "$SESSION" scripts/colab_bootstrap.sh /content/colab_bootstrap.sh >/dev/null 2>&1 || return 1
  colab upload -s "$SESSION" scripts/colab_trust_job.sh /content/colab_trust_job.sh >/dev/null 2>&1 || return 1
  colab upload -s "$SESSION" "$TMP/remaining.txt" /content/remaining.txt >/dev/null 2>&1 || return 1
  printf '%s\n' "import subprocess" \
    "p = subprocess.Popen('exec bash /content/colab_trust_job.sh $COMMIT >> /content/trust.log 2>&1', shell=True, start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)" \
    "print('job pid', p.pid)" | timeout 300 colab exec -s "$SESSION" 2>&1 | tail -1
  log "job launched for $(wc -l < "$TMP/remaining.txt") runs"
}

sync_finished() {
  local done_remote
  done_remote=$(remote "cd /content/repo 2>/dev/null && ls $OUT/*/report.json 2>/dev/null | cut -d/ -f3")
  for n in $done_remote; do
    [ -f "$OUT/$n/report.json" ] && continue
    remote "cd /content/repo && tar czf /content/sync-$n.tgz $OUT/$n $CKPT/$n/best.pt $CKPT/$n/train.log $CKPT/$n/training_state.json 2>/dev/null; echo ok" >/dev/null
    if timeout 900 colab download -s "$SESSION" "/content/sync-$n.tgz" "$TMP/$n.tgz" >/dev/null 2>&1 \
      && tar xzf "$TMP/$n.tgz" && [ -f "$OUT/$n/report.json" ]; then
      rm -f "$TMP/$n.tgz"
      log "synced $n"
    else
      log "sync failed for $n (will retry)"
    fi
  done
}

log "watchdog start (commit $COMMIT)"
while true; do
  left=$(remaining | wc -l)
  if [ "$left" -eq 0 ]; then
    log "all trust-v1 runs have local reports; stopping session"
    colab stop -s "$SESSION" >/dev/null 2>&1
    log "watchdog done"
    exit 0
  fi
  if alive; then
    sync_finished
    tail_line=$(remote "tail -1 /content/trust.log 2>/dev/null")
    log "alive; $left runs without local report; last: ${tail_line:0:160}"
  else
    start_session || log "session start failed; retrying next cycle"
  fi
  sleep "$PERIOD"
done
