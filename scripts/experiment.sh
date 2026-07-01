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
#   --select-method <name>               stage-1 selector: stock|random|stratified|covmatch|momentmatch|qddpp|relmatch|reldist|facloc
#   --select-k|--select-beta|--select-quality   stratified clusters / qddpp beta / qddpp quality (confidence|margin)
#   --momentmatch-mean-weight <float>    momentmatch mean-vs-covariance weight
#   --facloc-space <softlabel|feature>   facloc coverage embedding space (default softlabel)
#   --diagnostics                        log the trust panel (ECE/OOD/AUROC/NC), not just accuracy
#   --ood-sets <csv> | --fit-ipc <int>   diagnostics OOD sets and Mahalanobis/temperature fit budget
#   --results-file <path>                results JSONL (default logs/results.jsonl; --resume reads this too)
#   --resume                             skip if a matching row is already in the results file
#   --dry-run                            print the python command, don't execute
#   -h, --help                           show this help

set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python}"

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
SELECT_METHOD=""
SELECT_K=""
SELECT_BETA=""
SELECT_QUALITY=""
FACLOC_SPACE=""
MOMENTMATCH_MEAN_WEIGHT=""
DIAGNOSTICS=0
OOD_SETS=""
FIT_IPC=""
RESULTS_FILE=""
WEIGHTS_ARGS=()        # --w-<name> <val> pairs, verbatim passthrough to main.py
USER_WEIGHTS_JSON="{}"  # collected user-set weights as JSON, used by resume key and log filename

usage() {
  sed -n '2,32p' "$0"
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
    --select-method)           shift; SELECT_METHOD="$1"; shift ;;
    --select-k)                shift; SELECT_K="$1"; shift ;;
    --select-beta)             shift; SELECT_BETA="$1"; shift ;;
    --select-quality)          shift; SELECT_QUALITY="$1"; shift ;;
    --facloc-space)            shift; FACLOC_SPACE="$1"; shift ;;
    --momentmatch-mean-weight) shift; MOMENTMATCH_MEAN_WEIGHT="$1"; shift ;;
    --diagnostics)             DIAGNOSTICS=1; shift ;;
    --ood-sets)                shift; OOD_SETS="$1"; shift ;;
    --fit-ipc)                 shift; FIT_IPC="$1"; shift ;;
    --results-file)            shift; RESULTS_FILE="$1"; shift ;;
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
      USER_WEIGHTS_JSON=$("$PYTHON_BIN" -c "
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

RESULTS_FILE_EFF="${RESULTS_FILE:-logs/results.jsonl}"

# Stage-1 selector exp_name suffix — MUST mirror argument.py's exp_name construction so the per-cell
# run-log name and the --resume key track the path-keyed distilled set (empty for stock/unset).
SEL_SUFFIX=$(SM="$SELECT_METHOD" SK="${SELECT_K:-8}" SB="${SELECT_BETA:-0.0}" \
             SQ="${SELECT_QUALITY:-confidence}" FS="${FACLOC_SPACE:-softlabel}" \
             MW="${MOMENTMATCH_MEAN_WEIGHT:-1.0}" \
             "$PYTHON_BIN" - <<'PY'
import os
from validation.run_key import selector_suffix
print(selector_suffix({
    "select_method": os.environ.get("SM") or "stock",
    "select_k": os.environ.get("SK") or 8,
    "select_beta": os.environ.get("SB") or 0.0,
    "select_quality": os.environ.get("SQ") or "confidence",
    "facloc_space": os.environ.get("FS") or "softlabel",
    "momentmatch_mean_weight": os.environ.get("MW") or 1.0,
}))
PY
)

# Resolve canonical weights = LOSS_REGISTRY defaults overridden by user-set values.
# WEIGHTS_TAG is a stable filename slug (e.g. "kl1p0_ockl0p3") of nonzero weights.
# EXTRA_TAG slugs the non-weight hyperparams that distinguish a cell (gce_q when
# gce is active and != default). "-" means "no extras" (keeps field count).
read -r WEIGHTS_TAG EXTRA_TAG CANON_JSON < <(USER_JSON="$USER_WEIGHTS_JSON" GCE_Q="$GCE_Q" "$PYTHON_BIN" - <<'PY'
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

if [[ $RESUME -eq 1 ]]; then
  found=$(CANON_JSON="$CANON_JSON" DATASET="$DATASET" ARCH="$ARCH" STUD="$STUD_ARCH" \
          IPC="$IPC" SEED="$SEED" FACTOR="$FACTOR" NUM_CROP="$NUM_CROP" MIPC="$MIPC" \
          RE_EPOCHS="$RE_EPOCHS" GCE_Q="$GCE_Q" SELECT_METHOD="${SELECT_METHOD:-stock}" \
          SELECT_K="${SELECT_K:-8}" SELECT_BETA="${SELECT_BETA:-0.0}" \
          SELECT_QUALITY="${SELECT_QUALITY:-confidence}" \
          FACLOC_SPACE="${FACLOC_SPACE:-softlabel}" \
          MOMENTMATCH_MEAN_WEIGHT="${MOMENTMATCH_MEAN_WEIGHT:-1.0}" \
          DIAGNOSTICS="$DIAGNOSTICS" OOD_SETS="$OOD_SETS" FIT_IPC="${FIT_IPC:-50}" \
          RESULTS_FILE_EFF="$RESULTS_FILE_EFF" "$PYTHON_BIN" - <<'PY'
import json, os
from validation.run_key import find_completed_result, load_result_history

params = {
    "dataset": os.environ["DATASET"],
    "arch": os.environ["ARCH"],
    "stud_arch": os.environ["STUD"],
    "ipc": int(os.environ["IPC"]),
    "seed": int(os.environ["SEED"]),
    "factor": int(os.environ["FACTOR"]),
    "num_crop": int(os.environ["NUM_CROP"]),
    "mipc": int(os.environ["MIPC"]),
    "re_epochs": int(os.environ["RE_EPOCHS"]),
    "weights": json.loads(os.environ["CANON_JSON"]),
    "gce_q": os.environ.get("GCE_Q") or None,
    "select_method": os.environ["SELECT_METHOD"],
    "select_k": os.environ["SELECT_K"],
    "select_beta": os.environ["SELECT_BETA"],
    "select_quality": os.environ["SELECT_QUALITY"],
    "facloc_space": os.environ["FACLOC_SPACE"],
    "momentmatch_mean_weight": os.environ["MOMENTMATCH_MEAN_WEIGHT"],
    "diagnostics": os.environ["DIAGNOSTICS"] == "1",
    "ood_sets": os.environ.get("OOD_SETS", ""),
    "fit_ipc": os.environ.get("FIT_IPC", "50"),
}
match = find_completed_result(
    params,
    load_result_history(os.environ["RESULTS_FILE_EFF"]),
)
if match:
    print(f"MATCH\t{match['path']}\t{match['line_no']}")
PY
)
  if [[ "$found" == MATCH$'\t'* ]]; then
    IFS=$'\t' read -r _match_tag match_path match_line <<< "$found"
    echo "[skip] dataset=$DATASET arch=$ARCH stud=$STUD_ARCH ipc=$IPC seed=$SEED method=${SELECT_METHOD:-stock} weights=$CANON_JSON (already in $match_path:$match_line)"
    exit 0
  fi
fi

# Fold the extras + selector slugs into filename + run tag so gce_q AND selector variants
# (e.g. stock vs relmatch at the same dataset/arch/ipc/seed/weights) get distinct logs/run-tags.
EXTRA_SLUG=""
[[ "$EXTRA_TAG" != "-" ]] && EXTRA_SLUG="_${EXTRA_TAG}"
EXTRA_SLUG="${EXTRA_SLUG}${SEL_SUFFIX}"
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

# Stage-1 selector + trustworthiness-diagnostics passthrough (all opt-in; absent => stock RDED
# with no diagnostics, i.e. identical to a plain loss-term sweep).
[[ -n "$SELECT_METHOD" ]]            && py_args+=(--select-method "$SELECT_METHOD")
[[ -n "$SELECT_K" ]]                && py_args+=(--select-k "$SELECT_K")
[[ -n "$SELECT_BETA" ]]             && py_args+=(--select-beta "$SELECT_BETA")
[[ -n "$SELECT_QUALITY" ]]          && py_args+=(--select-quality "$SELECT_QUALITY")
[[ -n "$FACLOC_SPACE" ]]            && py_args+=(--facloc-space "$FACLOC_SPACE")
[[ -n "$MOMENTMATCH_MEAN_WEIGHT" ]] && py_args+=(--momentmatch-mean-weight "$MOMENTMATCH_MEAN_WEIGHT")
[[ $DIAGNOSTICS -eq 1 ]]            && py_args+=(--diagnostics)
[[ -n "$OOD_SETS" ]]                && py_args+=(--ood-sets "$OOD_SETS")
[[ -n "$FIT_IPC" ]]                 && py_args+=(--fit-ipc "$FIT_IPC")
[[ -n "$RESULTS_FILE" ]]            && py_args+=(--results-file "$RESULTS_FILE")

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
  echo "[dry-run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON_BIN ./main.py ${py_args[*]} -> $log"
else
  echo "[$(date +%H:%M:%S)] gpu=$CUDA_VISIBLE_DEVICES dataset=$DATASET arch=$ARCH stud=$STUD_ARCH ipc=$IPC seed=$SEED weights=$CANON_JSON -> $log"
  "$PYTHON_BIN" ./main.py "${py_args[@]}" 2>&1 | tee "$log"
fi
