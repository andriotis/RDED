#!/usr/bin/env bash
# Single-experiment wrapper around `python ./main.py`. Accepts pretty flag
# names, applies RDED published-protocol defaults, and supports --resume
# (skip if this cell is already logged in logs/results.jsonl) and --dry-run.
#
# Usage: bash scripts/experiment.sh [options]
#   --dataset {cifar100|tinyimagenet}    required
#   --arch    {conv3|conv4|resnet18_modified}   required; teacher arch (also scores synth)
#   --stud-arch <arch>                   optional; defaults to --arch (set distinct for cross-arch)
#   --seed <int>                         required
#   --ipc <int>                          required (typically 1 | 10 | 50)
#   --w-<name> <float>                   weight on student-loss term <name> (e.g. --w-kl 1.0, --w-ockl 0.3)
#                                          unknown names are rejected by main.py argparse.
#                                          omitted weights fall back to LOSS_REGISTRY defaults.
#   --re-epochs <int>                    default: 300
#   --factor <int>                       default: 1
#   --num-crop <int>                     default: 5
#   --mipc <int>                         default: 300
#   --monitor <csv>                      comma-separated loss-term names to log under no_grad
#                                          (passed through to main.py; default empty)
#   --gce-q <float>                      q for the GCE term (default 0.7; recorded + keyed when gce active)
#   --skip-synth                         pass --skip-synth to main.py (paired protocol)
#   --resume                             skip if a matching row is already in logs/results.jsonl
#   --dry-run                            print the python command, don't execute
#   -h, --help                           show this help

set -uo pipefail
cd "$(dirname "$0")/.."

DATASET=""
ARCH=""
STUD_ARCH=""
SEED=""
IPC=""
RE_EPOCHS=300
FACTOR=1
NUM_CROP=5
MIPC=300
MONITOR=""
GCE_Q=""
SKIP_SYNTH=0
RESUME=0
DRY_RUN=0
SWEEP_NAME=""
CELL_ID=""
WEIGHTS_ARGS=()        # --w-<name> <val> pairs, verbatim passthrough to main.py
USER_WEIGHTS_JSON="{}"  # collected user-set weights as JSON, used by resume key and log filename

usage() {
  sed -n '2,25p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)    shift; DATASET="$1"; shift ;;
    --arch)       shift; ARCH="$1"; shift ;;
    --stud-arch)  shift; STUD_ARCH="$1"; shift ;;
    --seed)       shift; SEED="$1"; shift ;;
    --ipc)        shift; IPC="$1"; shift ;;
    --re-epochs)  shift; RE_EPOCHS="$1"; shift ;;
    --factor)     shift; FACTOR="$1"; shift ;;
    --num-crop)   shift; NUM_CROP="$1"; shift ;;
    --mipc)       shift; MIPC="$1"; shift ;;
    --monitor)    shift; MONITOR="$1"; shift ;;
    --gce-q)      shift; GCE_Q="$1"; shift ;;
    --skip-synth) SKIP_SYNTH=1; shift ;;
    --resume)     RESUME=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --sweep-name) shift; SWEEP_NAME="$1"; shift ;;
    --cell-id)    shift; CELL_ID="$1"; shift ;;
    -h|--help)    usage; exit 0 ;;
    --w-*)
      flag="$1"
      val="${2:-}"
      if [[ -z "$val" || "$val" == --* ]]; then
        echo "$flag requires a numeric value" >&2; exit 1
      fi
      WEIGHTS_ARGS+=("$flag" "$val")
      name="${flag#--w-}"
      name="${name//-/_}"
      # accumulate JSON {"<name>": <val>, ...}
      USER_WEIGHTS_JSON=$(python -c "
import json, sys
d = json.loads(sys.argv[1]); d[sys.argv[2]] = float(sys.argv[3]); print(json.dumps(d))
" "$USER_WEIGHTS_JSON" "$name" "$val")
      shift 2 ;;
    *)            echo "unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

# Required args
for var in DATASET ARCH SEED IPC; do
  if [[ -z "${!var}" ]]; then
    echo "missing required --${var,,}" >&2
    usage; exit 1
  fi
done

if [[ -z "$STUD_ARCH" ]]; then
  STUD_ARCH="$ARCH"
fi

is_compatible() {
  case "$1-$2" in
    cifar100-conv3)                 return 0 ;;
    cifar100-resnet18_modified)     return 0 ;;
    tinyimagenet-conv4)             return 0 ;;
    tinyimagenet-resnet18_modified) return 0 ;;
    *)                              return 1 ;;
  esac
}

if ! is_compatible "$DATASET" "$ARCH"; then
  echo "incompatible (dataset, arch): ($DATASET, $ARCH)" >&2
  echo "  cifar100 supports {conv3, resnet18_modified}; tinyimagenet supports {conv4, resnet18_modified}" >&2
  exit 1
fi
case "$STUD_ARCH" in
  conv3|conv4|resnet18_modified) ;;
  *) echo "unknown --stud-arch: $STUD_ARCH" >&2; exit 1 ;;
esac

mkdir -p logs/runs

# Resolve canonical weights = LOSS_REGISTRY defaults overridden by user-set values.
# WEIGHTS_TAG is a stable filename slug (e.g. "kl1p0_ockl0p3") of nonzero weights.
# EXTRA_TAG slugs the non-weight hyperparams that distinguish a cell (gce_q when
# gce is active and != default). "-" means "no extras" (keeps field count).
read -r WEIGHTS_TAG EXTRA_TAG CANON_JSON < <(USER_JSON="$USER_WEIGHTS_JSON" GCE_Q="$GCE_Q" python - <<'PY'
import json, os
from validation.losses import LOSS_REGISTRY
user = json.loads(os.environ["USER_JSON"])
canon = {name: float(user.get(name, default)) for name, (_, default) in LOSS_REGISTRY.items()}
nz = sorted((n, w) for n, w in canon.items() if w > 0)
def slug(w):
    return f"{w:g}".replace(".", "p").replace("-", "neg")
tag = "_".join(f"{n}{slug(w)}" for n, w in nz) or "noloss"
extras = []
gce_q = os.environ.get("GCE_Q", "")
if canon.get("gce", 0) > 0 and gce_q and float(gce_q) != 0.7:
    extras.append("q" + slug(float(gce_q)))
print(tag, ("_".join(extras) or "-"), json.dumps(canon, sort_keys=True))
PY
)

if [[ $RESUME -eq 1 && -f logs/results.jsonl ]]; then
  found=$(CANON_JSON="$CANON_JSON" DATASET="$DATASET" ARCH="$ARCH" STUD="$STUD_ARCH" IPC="$IPC" SEED="$SEED" GCE_Q="$GCE_Q" python - <<'PY'
import json, os
canon = json.loads(os.environ["CANON_JSON"])

def extras(weights, gce_q):
    """Non-weight hyperparams that distinguish a cell. Empty -> matches legacy
    rows (which never set these), so backward-compatible."""
    items = []
    if float(weights.get("gce", 0)) > 0 and gce_q is not None and float(gce_q) != 0.7:
        items.append(("gce_q", float(gce_q)))
    return tuple(items)

new_q = os.environ.get("GCE_Q") or None
key = (
    os.environ["DATASET"], os.environ["ARCH"], os.environ["STUD"],
    int(os.environ["IPC"]), int(os.environ["SEED"]),
    tuple(sorted(canon.items())),
    extras(canon, new_q),
)
with open("logs/results.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("best_top1") is None:
            continue
        w = r.get("weights")
        if not isinstance(w, dict):
            continue
        rk = (
            r.get("dataset"), r.get("arch"), r.get("stud"),
            r.get("ipc"), r.get("seed"),
            tuple(sorted((k, float(v)) for k, v in w.items())),
            extras(w, r.get("gce_q")),
        )
        if rk == key:
            print("MATCH")
            break
PY
)
  if [[ "$found" == "MATCH" ]]; then
    echo "[skip] dataset=$DATASET arch=$ARCH stud=$STUD_ARCH ipc=$IPC seed=$SEED weights=$CANON_JSON (already in results.jsonl)"
    exit 0
  fi
fi

# Fold the extras slug into filename + run tag so gce_q variants don't collide.
EXTRA_SLUG=""
[[ "$EXTRA_TAG" != "-" ]] && EXTRA_SLUG="_${EXTRA_TAG}"
log="logs/runs/${DATASET}_${ARCH}_to_${STUD_ARCH}_ipc${IPC}_seed${SEED}_w${WEIGHTS_TAG}${EXTRA_SLUG}.log"

RUN_TAG="seed${SEED}_w${WEIGHTS_TAG}${EXTRA_SLUG}"

py_args=(
  --subset       "$DATASET"
  --arch-name    "$ARCH"
  --stud-name    "$STUD_ARCH"
  --factor       "$FACTOR"
  --num-crop     "$NUM_CROP"
  --mipc         "$MIPC"
  --ipc          "$IPC"
  --re-epochs    "$RE_EPOCHS"
  --seed         "$SEED"
  --run-tag      "$RUN_TAG"
  # Seed-scope the synth dir so parallel cells (run_sweep.py packs many at once)
  # never share — and thus never rmtree — the same distilled set. exp_name
  # already carries the other synth-key fields; the seed leaf completes it.
  --syn-data-path "syn_data_seed${SEED}"
)
if [[ -n "$SWEEP_NAME" ]]; then
  py_args+=(--sweep-name "$SWEEP_NAME")
fi
if [[ -n "$CELL_ID" ]]; then
  py_args+=(--cell-id "$CELL_ID")
fi
if [[ ${#WEIGHTS_ARGS[@]} -gt 0 ]]; then
  py_args+=("${WEIGHTS_ARGS[@]}")
fi
if [[ -n "$MONITOR" ]]; then
  py_args+=(--monitor "$MONITOR")
fi
if [[ -n "$GCE_Q" ]]; then
  py_args+=(--gce-q "$GCE_Q")
fi
[[ $SKIP_SYNTH -eq 1 ]] && py_args+=(--skip-synth)

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# Pin this run to a single GPU. run_sweep.py sets CUDA_VISIBLE_DEVICES per cell
# (one GPU per slot); for a bare/direct run, default to GPU 0 so we never fan a
# tiny batch across all GPUs via DataParallel (which only adds scatter/gather
# overhead for these small models). Honor any value already set by the caller.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Make CUDA's device numbering match nvidia-smi's (PCI order) so the id
# run_sweep.py picked addresses the same physical GPU it sized for.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./main.py ${py_args[*]} -> $log"
else
  echo "[$(date +%H:%M:%S)] gpu=$CUDA_VISIBLE_DEVICES dataset=$DATASET arch=$ARCH stud=$STUD_ARCH ipc=$IPC seed=$SEED weights=$CANON_JSON -> $log"
  python ./main.py "${py_args[@]}" 2>&1 | tee "$log"
fi
