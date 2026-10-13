#!/usr/bin/env bash
# v4 in ONE Kaggle session (12 h cap): the v4 data (PART=data V4=1), then PART=v4 with whatever time is left
# (align ~30% of it, then polish, then Gate 1). Everything is saved to Hugging Face as it goes: a stopped run
# can simply be started again. Cell:  !cd /tmp/repo && bash scripts/kaggle_v4_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."
TOTAL_MIN=${TOTAL_MIN:-690}
echo "[$(date +%H:%M:%S)] v4 all-in-one: HF_TOKEN $([ -n "${HF_TOKEN:-}" ] && echo set || echo MISSING), \
GPUs: $(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true), budget ${TOTAL_MIN} min"
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set: attach the secret (Add-ons > Secrets) and run the token cell"; exit 1; }
T0=$(date +%s)
PART=data V4=1 bash scripts/kaggle_v3.sh
rc=$?
USED=$(( ($(date +%s) - T0) / 60 ))
[ $rc = 0 ] || { echo "[$(date +%H:%M:%S)] the data part failed (exit $rc) after $USED min: see the lines above"; exit $rc; }
LEFT=$(( TOTAL_MIN - USED ))
echo "[$(date +%H:%M:%S)] data took $USED min, training gets $LEFT min (align $(( LEFT * 30 / 100 )) min)"
[ $LEFT -ge 240 ] || { echo "less than 4 h left: run PART=v4 bash scripts/kaggle_v3.sh in a new session"; exit 0; }
PART=v4 BUDGET_MIN=$LEFT ALIGN_MIN=$(( LEFT * 30 / 100 )) bash scripts/kaggle_v3.sh
