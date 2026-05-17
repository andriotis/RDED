#!/usr/bin/env bash
# Batch runner for Table 2 cells.
#   ./_run.sh cifar10_ipc01_conv.sh cifar10_ipc10_rn18.sh ...
# - Pins GPU 1 (scripts already do, but enforce here too).
# - Streams each run's full log into logs/table2/<name>.log.
# - Parses "Best accuracy is X@Y" and appends a CSV row to logs/table2/results.csv.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT/logs/table2"
CSV="$LOG_DIR/results.csv"
mkdir -p "$LOG_DIR"
[ -f "$CSV" ] || echo "script,best_top1,best_epoch,wall_seconds,status,timestamp" > "$CSV"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate general 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

for s in "$@"; do
    script="$ROOT/scripts/table2/$s"
    if [ ! -f "$script" ]; then
        echo "MISSING $s" >&2
        continue
    fi
    log="$LOG_DIR/${s%.sh}.log"
    ts="$(date -Iseconds)"
    echo "[$ts] >>> $s"
    t0=$SECONDS
    if bash "$script" > "$log" 2>&1; then
        status="ok"
    else
        status="fail"
    fi
    dt=$((SECONDS - t0))
    line=$(grep -E "Best accuracy is" "$log" | tail -1)
    top1=$(echo "$line" | sed -E 's/.*Best accuracy is ([0-9.]+)@([0-9]+).*/\1/')
    epoch=$(echo "$line" | sed -E 's/.*Best accuracy is ([0-9.]+)@([0-9]+).*/\2/')
    [ -z "$top1" ] && top1="NA"
    [ -z "$epoch" ] && epoch="NA"
    echo "${s%.sh},${top1},${epoch},${dt},${status},${ts}" >> "$CSV"
    echo "[$(date -Iseconds)] <<< $s  top1=$top1 epoch=$epoch wall=${dt}s status=$status"
done
