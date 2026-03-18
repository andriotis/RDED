#!/bin/bash
# =============================================================================
# Unified experiment runner for RDED Table 2 reproductions and OCCE variants.
#
# Runs baseline (KD-only) and/or OCCE-augmented student training across
# 7 datasets, 3 architecture families (ConvNet, ResNet-18, ResNet-101),
# and 3 IPC values (1, 10, 50).
#
# Results table columns:
#   Paper | Replicated | U-OCCE | S-OCCE | M-OCCE
#
# See --help for full usage.
# =============================================================================

set -e

# ---- Help -------------------------------------------------------------------
show_help() {
    cat <<'HELP'
Usage: bash scripts/experiments.sh [OPTIONS]

Unified experiment runner for RDED Table 2 reproductions and OCCE variants.

FILTERING OPTIONS:
  --dataset DATASET     Run only this dataset (default: all)
                        Choices: cifar10, cifar100, imagenet-nette,
                        imagenet-woof, tinyimagenet, imagenet-100, imagenet-1k
  --arch ARCH           Run only this architecture family (default: all)
                        Choices: convnet, resnet18, resnet101
  --ipc IPC             Run only this IPC value (default: all)
                        Choices: 1, 10, 50

EXPERIMENT MODE:
  --mode MODE           Which experiments to run (default: occe)
                        baseline  — KD-only replication (no OCCE loss)
                        uocce     — Uniform OCCE only
                        socce     — Soft OCCE only
                        mocce     — Margin OCCE only
                        occe      — All three OCCE variants (U/S/M)
                        all       — Baseline + all OCCE variants

TRAINING PARAMETERS:
  --seeds "S1 S2 ..."   Space-separated list of random seeds (default: "42")
  --re-epochs N         Number of re-training epochs (default: 300)
  --num-crop N          Number of crops per image (default: 5)
  --mipc N              Maximum IPC for crop generation (default: 300)

OCCE-SPECIFIC PARAMETERS:
  --occe-gamma FLOAT    Importance ratio (γ) for U-OCCE (adaptive: γ*KL/OCCE),
                        or fixed weight for S-OCCE/M-OCCE (default: 0.1)
  --occe-temp-schedule  Temperature schedule for M-OCCE (default: fixed)
                        Choices: fixed, linear, cosine
  --occe-temp-start F   Starting temperature for M-OCCE schedule
                        (default: inherits KD temperature)
  --occe-temp-final F   Final temperature for M-OCCE (default: 1.0)

LOG DIRECTORIES:
  --log-dir-baseline DIR  Log directory for baseline runs (default: ./logs/baseline)
  --log-dir-uocce DIR     Log directory for U-OCCE runs (default: ./logs/table2_uocce)
  --log-dir-socce DIR     Log directory for S-OCCE runs (default: ./logs/table2_socce)
  --log-dir-mocce DIR     Log directory for M-OCCE runs (default: ./logs/table2_mocce)

ACTIONS:
  --results             Print the combined results table and exit
  --dry-run             Print experiment configs without running them
  --force               Re-run experiments even if logs already exist
  --no-color            Disable ANSI color in results table (for pipes)
  -h, --help            Show this help message and exit

EXAMPLES:
  # Run baseline (replicated) experiments for CIFAR-10 only
  bash scripts/experiments.sh --mode baseline --dataset cifar10

  # Run all OCCE variants on ResNet-18 with 3 seeds
  bash scripts/experiments.sh --mode occe --arch resnet18 --seeds "42 43 44"

  # Run everything (baseline + all OCCE) for IPC=10
  bash scripts/experiments.sh --mode all --ipc 10

  # Show results table without running anything
  bash scripts/experiments.sh --results

  # Dry-run to preview M-OCCE commands
  bash scripts/experiments.sh --mode mocce --dry-run
HELP
}

# ---- Defaults ---------------------------------------------------------------
FILTER_DATASET="all"
FILTER_ARCH="all"
FILTER_IPC="all"
SEEDS="42"
DRY_RUN=false
SHOW_RESULTS=false
FORCE=false
NO_COLOR=false
RE_EPOCHS=300
NUM_CROP=5
MIPC=300

MODE="occe"  # baseline | uocce | socce | mocce | occe | all
LOG_DIR_BASELINE="./logs/baseline"
LOG_DIR_UOCCE="./logs/table2_uocce"
LOG_DIR_SOCCE="./logs/table2_socce"
LOG_DIR_MOCCE="./logs/table2_mocce"
OCCE_GAMMA="0.1"
OCCE_TEMP_SCHEDULE="fixed"
OCCE_TEMP_START=""
OCCE_TEMP_FINAL="1.0"

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)            FILTER_DATASET="$2";    shift 2 ;;
        --arch)               FILTER_ARCH="$2";       shift 2 ;;
        --ipc)                FILTER_IPC="$2";        shift 2 ;;
        --seeds)              SEEDS="$2";             shift 2 ;;
        --dry-run)            DRY_RUN=true;           shift   ;;
        --results)            SHOW_RESULTS=true;      shift   ;;
        --force)              FORCE=true;             shift   ;;
        --no-color)           NO_COLOR=true;          shift   ;;
        --re-epochs)          RE_EPOCHS="$2";         shift 2 ;;
        --mode)               MODE="$2";              shift 2 ;;
        --occe-gamma)         OCCE_GAMMA="$2";        shift 2 ;;
        --occe-temp-schedule) OCCE_TEMP_SCHEDULE="$2"; shift 2 ;;
        --occe-temp-start)    OCCE_TEMP_START="$2";   shift 2 ;;
        --occe-temp-final)    OCCE_TEMP_FINAL="$2";   shift 2 ;;
        --log-dir-baseline)   LOG_DIR_BASELINE="$2";  shift 2 ;;
        --log-dir-uocce)      LOG_DIR_UOCCE="$2";     shift 2 ;;
        --log-dir-socce)      LOG_DIR_SOCCE="$2";     shift 2 ;;
        --log-dir-mocce)      LOG_DIR_MOCCE="$2";     shift 2 ;;
        -h|--help)            show_help;           exit 0  ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "baseline" && "$MODE" != "uocce" && "$MODE" != "socce" && "$MODE" != "mocce" && "$MODE" != "occe" && "$MODE" != "all" ]]; then
    echo "Invalid --mode value: $MODE (expected: baseline|uocce|socce|mocce|occe|all)"
    exit 1
fi

# ---- Shared helpers ---------------------------------------------------------
should_run_dataset() {
    [[ "$FILTER_DATASET" == "all" || "$FILTER_DATASET" == "$1" ]]
}

should_run_arch() {
    [[ "$FILTER_ARCH" == "all" || "$FILTER_ARCH" == "$1" ]]
}

should_run_ipc() {
    [[ "$FILTER_IPC" == "all" || "$FILTER_IPC" == "$1" ]]
}

# ---- Baseline experiments (KD-only, no OCCE) --------------------------------
# run_experiment_baseline subset arch_name stud_name factor ipc seed arch_label [extra_args]
run_experiment_baseline() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local ipc="$5"
    local seed="$6"
    local arch_label="$7"
    local extra_args="${8:-}"

    local exp_tag="${subset}_${arch_label}_ipc${ipc}_seed${seed}"
    local log_file="${LOG_DIR_BASELINE}/${exp_tag}.log"

    echo "============================================================"
    echo " Running (baseline): ${exp_tag}"
    echo "   subset=${subset}  arch=${arch_name}  stud=${stud_name}"
    echo "   factor=${factor}  ipc=${ipc}  seed=${seed}"
    echo "============================================================"

    local cmd="python ./main.py \
--subset ${subset} \
--arch-name ${arch_name} \
--stud-name ${stud_name} \
--factor ${factor} \
--num-crop ${NUM_CROP} \
--mipc ${MIPC} \
--ipc ${ipc} \
--re-epochs ${RE_EPOCHS} \
--seed ${seed}"
    [[ -n "$extra_args" ]] && cmd="${cmd} ${extra_args}"

    if [ "$DRY_RUN" = true ]; then
        printf "  %-18s %s\n" "--subset"    "${subset}"
        printf "  %-18s %s\n" "--arch-name" "${arch_name}"
        printf "  %-18s %s\n" "--stud-name" "${stud_name}"
        printf "  %-18s %s\n" "--factor"    "${factor}"
        printf "  %-18s %s\n" "--num-crop"  "${NUM_CROP}"
        printf "  %-18s %s\n" "--mipc"      "${MIPC}"
        printf "  %-18s %s\n" "--ipc"       "${ipc}"
        printf "  %-18s %s\n" "--re-epochs" "${RE_EPOCHS}"
        printf "  %-18s %s\n" "--seed"      "${seed}"
        echo ""
        return
    fi

    if [[ "$FORCE" = false && -f "$log_file" ]] && grep -q "Best accuracy is" "$log_file"; then
        local prev_acc
        prev_acc=$(grep -oP 'Best accuracy is \K[0-9]+\.?[0-9]*' "$log_file" | tail -1)
        echo "  SKIPPED (already completed, best acc = ${prev_acc}%)"
        echo ""
        return
    fi

    mkdir -p "${LOG_DIR_BASELINE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${exp_tag}" | tee -a "${log_file}"
    eval "${cmd}" 2>&1 | tee -a "${log_file}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${exp_tag}" | tee -a "${log_file}"
    echo ""
}

run_all_ipcs_baseline() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local arch_label="$5"

    for ipc in 1 10 50; do
        if should_run_ipc "$ipc"; then
            for seed in $SEEDS; do
                run_experiment_baseline "$subset" "$arch_name" "$stud_name" \
                                    "$factor" "$ipc" "$seed" "$arch_label"
            done
        fi
    done
}

# ---- U-OCCE experiments (hard uniform one-cold target) ----------------------
# run_experiment_uocce subset arch_name stud_name factor ipc seed arch_label [extra_args]
run_experiment_uocce() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local ipc="$5"
    local seed="$6"
    local arch_label="$7"
    local extra_args="${8:-}"

    local exp_tag="${subset}_${arch_label}_ipc${ipc}_seed${seed}_uocce"
    local log_file="${LOG_DIR_UOCCE}/${exp_tag}.log"

    echo "============================================================"
    echo " Running (U-OCCE): ${exp_tag}"
    echo "   subset=${subset}  arch=${arch_name}  stud=${stud_name}"
    echo "   factor=${factor}  ipc=${ipc}  seed=${seed}"
    echo "   occe_gamma=${OCCE_GAMMA}  occe_mode=uniform"
    echo "============================================================"

    local cmd="python ./main.py \
--subset ${subset} \
--arch-name ${arch_name} \
--stud-name ${stud_name} \
--factor ${factor} \
--num-crop ${NUM_CROP} \
--mipc ${MIPC} \
--ipc ${ipc} \
--re-epochs ${RE_EPOCHS} \
--seed ${seed} \
--use-occe \
--occe-mode uniform \
--occe-gamma ${OCCE_GAMMA}"
    [[ -n "$extra_args" ]] && cmd="${cmd} ${extra_args}"

    if [ "$DRY_RUN" = true ]; then
        printf "  %-18s %s\n" "--subset"     "${subset}"
        printf "  %-18s %s\n" "--arch-name"  "${arch_name}"
        printf "  %-18s %s\n" "--stud-name"  "${stud_name}"
        printf "  %-18s %s\n" "--factor"     "${factor}"
        printf "  %-18s %s\n" "--num-crop"   "${NUM_CROP}"
        printf "  %-18s %s\n" "--mipc"       "${MIPC}"
        printf "  %-18s %s\n" "--ipc"        "${ipc}"
        printf "  %-18s %s\n" "--re-epochs"  "${RE_EPOCHS}"
        printf "  %-18s %s\n" "--seed"       "${seed}"
        printf "  %-18s %s\n" "--occe-mode"  "uniform"
        printf "  %-18s %s\n" "--occe-gamma" "${OCCE_GAMMA}"
        echo ""
        return
    fi

    if [[ "$FORCE" = false && -f "$log_file" ]] && grep -q "Best accuracy is" "$log_file"; then
        local prev_acc
        prev_acc=$(grep -oP 'Best accuracy is \K[0-9]+\.?[0-9]*' "$log_file" | tail -1)
        echo "  SKIPPED (already completed, best acc = ${prev_acc}%)"
        echo ""
        return
    fi

    mkdir -p "${LOG_DIR_UOCCE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${exp_tag}" | tee -a "${log_file}"
    eval "${cmd}" 2>&1 | tee -a "${log_file}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${exp_tag}" | tee -a "${log_file}"
    echo ""
}

run_all_ipcs_uocce() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local arch_label="$5"

    for ipc in 1 10 50; do
        if should_run_ipc "$ipc"; then
            for seed in $SEEDS; do
                run_experiment_uocce "$subset" "$arch_name" "$stud_name" \
                                     "$factor" "$ipc" "$seed" "$arch_label"
            done
        fi
    done
}

# ---- S-OCCE experiments (teacher-soft-proportional one-cold target) ----------
# run_experiment_socce subset arch_name stud_name factor ipc seed arch_label [extra_args]
run_experiment_socce() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local ipc="$5"
    local seed="$6"
    local arch_label="$7"
    local extra_args="${8:-}"

    local exp_tag="${subset}_${arch_label}_ipc${ipc}_seed${seed}_socce"
    local log_file="${LOG_DIR_SOCCE}/${exp_tag}.log"

    echo "============================================================"
    echo " Running (S-OCCE): ${exp_tag}"
    echo "   subset=${subset}  arch=${arch_name}  stud=${stud_name}"
    echo "   factor=${factor}  ipc=${ipc}  seed=${seed}"
    echo "   occe_gamma=${OCCE_GAMMA}  occe_mode=soft"
    echo "============================================================"

    local cmd="python ./main.py \
--subset ${subset} \
--arch-name ${arch_name} \
--stud-name ${stud_name} \
--factor ${factor} \
--num-crop ${NUM_CROP} \
--mipc ${MIPC} \
--ipc ${ipc} \
--re-epochs ${RE_EPOCHS} \
--seed ${seed} \
--use-occe \
--occe-mode soft \
--occe-gamma ${OCCE_GAMMA}"
    [[ -n "$extra_args" ]] && cmd="${cmd} ${extra_args}"

    if [ "$DRY_RUN" = true ]; then
        printf "  %-18s %s\n" "--subset"     "${subset}"
        printf "  %-18s %s\n" "--arch-name"  "${arch_name}"
        printf "  %-18s %s\n" "--stud-name"  "${stud_name}"
        printf "  %-18s %s\n" "--factor"     "${factor}"
        printf "  %-18s %s\n" "--num-crop"   "${NUM_CROP}"
        printf "  %-18s %s\n" "--mipc"       "${MIPC}"
        printf "  %-18s %s\n" "--ipc"        "${ipc}"
        printf "  %-18s %s\n" "--re-epochs"  "${RE_EPOCHS}"
        printf "  %-18s %s\n" "--seed"       "${seed}"
        printf "  %-18s %s\n" "--occe-mode"  "soft"
        printf "  %-18s %s\n" "--occe-gamma" "${OCCE_GAMMA}"
        echo ""
        return
    fi

    if [[ "$FORCE" = false && -f "$log_file" ]] && grep -q "Best accuracy is" "$log_file"; then
        local prev_acc
        prev_acc=$(grep -oP 'Best accuracy is \K[0-9]+\.?[0-9]*' "$log_file" | tail -1)
        echo "  SKIPPED (already completed, best acc = ${prev_acc}%)"
        echo ""
        return
    fi

    mkdir -p "${LOG_DIR_SOCCE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${exp_tag}" | tee -a "${log_file}"
    eval "${cmd}" 2>&1 | tee -a "${log_file}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${exp_tag}" | tee -a "${log_file}"
    echo ""
}

run_all_ipcs_socce() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local arch_label="$5"

    for ipc in 1 10 50; do
        if should_run_ipc "$ipc"; then
            for seed in $SEEDS; do
                run_experiment_socce "$subset" "$arch_name" "$stud_name" \
                                     "$factor" "$ipc" "$seed" "$arch_label"
            done
        fi
    done
}

# ---- M-OCCE experiments (teacher logit-margin-proportional one-cold target) --
# run_experiment_mocce subset arch_name stud_name factor ipc seed arch_label [extra_args]
run_experiment_mocce() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local ipc="$5"
    local seed="$6"
    local arch_label="$7"
    local extra_args="${8:-}"

    local exp_tag="${subset}_${arch_label}_ipc${ipc}_seed${seed}_mocce"
    local log_file="${LOG_DIR_MOCCE}/${exp_tag}.log"

    echo "============================================================"
    echo " Running (M-OCCE): ${exp_tag}"
    echo "   subset=${subset}  arch=${arch_name}  stud=${stud_name}"
    echo "   factor=${factor}  ipc=${ipc}  seed=${seed}"
    echo "   occe_gamma=${OCCE_GAMMA}  occe_mode=margin"
    echo "   occe_temp_schedule=${OCCE_TEMP_SCHEDULE}  occe_temp_start=${OCCE_TEMP_START:-<kd_temp>}  occe_temp_final=${OCCE_TEMP_FINAL}"
    echo "============================================================"

    local temp_args="--occe-temp-schedule ${OCCE_TEMP_SCHEDULE} --occe-temp-final ${OCCE_TEMP_FINAL}"
    [[ -n "$OCCE_TEMP_START" ]] && temp_args="${temp_args} --occe-temp-start ${OCCE_TEMP_START}"

    local cmd="python ./main.py \
--subset ${subset} \
--arch-name ${arch_name} \
--stud-name ${stud_name} \
--factor ${factor} \
--num-crop ${NUM_CROP} \
--mipc ${MIPC} \
--ipc ${ipc} \
--re-epochs ${RE_EPOCHS} \
--seed ${seed} \
--use-occe \
--occe-mode margin \
--occe-gamma ${OCCE_GAMMA} \
${temp_args}"
    [[ -n "$extra_args" ]] && cmd="${cmd} ${extra_args}"

    if [ "$DRY_RUN" = true ]; then
        printf "  %-22s %s\n" "--subset"              "${subset}"
        printf "  %-22s %s\n" "--arch-name"           "${arch_name}"
        printf "  %-22s %s\n" "--stud-name"           "${stud_name}"
        printf "  %-22s %s\n" "--factor"              "${factor}"
        printf "  %-22s %s\n" "--num-crop"            "${NUM_CROP}"
        printf "  %-22s %s\n" "--mipc"                "${MIPC}"
        printf "  %-22s %s\n" "--ipc"                 "${ipc}"
        printf "  %-22s %s\n" "--re-epochs"           "${RE_EPOCHS}"
        printf "  %-22s %s\n" "--seed"                "${seed}"
        printf "  %-22s %s\n" "--occe-mode"           "margin"
        printf "  %-22s %s\n" "--occe-gamma"          "${OCCE_GAMMA}"
        printf "  %-22s %s\n" "--occe-temp-schedule"  "${OCCE_TEMP_SCHEDULE}"
        printf "  %-22s %s\n" "--occe-temp-start"     "${OCCE_TEMP_START:-<kd_temp>}"
        printf "  %-22s %s\n" "--occe-temp-final"     "${OCCE_TEMP_FINAL}"
        echo ""
        return
    fi

    if [[ "$FORCE" = false && -f "$log_file" ]] && grep -q "Best accuracy is" "$log_file"; then
        local prev_acc
        prev_acc=$(grep -oP 'Best accuracy is \K[0-9]+\.?[0-9]*' "$log_file" | tail -1)
        echo "  SKIPPED (already completed, best acc = ${prev_acc}%)"
        echo ""
        return
    fi

    mkdir -p "${LOG_DIR_MOCCE}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${exp_tag}" | tee -a "${log_file}"
    eval "${cmd}" 2>&1 | tee -a "${log_file}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${exp_tag}" | tee -a "${log_file}"
    echo ""
}

run_all_ipcs_mocce() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local arch_label="$5"

    for ipc in 1 10 50; do
        if should_run_ipc "$ipc"; then
            for seed in $SEEDS; do
                run_experiment_mocce "$subset" "$arch_name" "$stud_name" \
                                     "$factor" "$ipc" "$seed" "$arch_label"
            done
        fi
    done
}

# =============================================================================
# Paper results lookup  (Table 2, RDED columns only)
# Key format: "{dataset}|{arch_family}|{ipc}"  Value: "acc ± std"
# =============================================================================
declare -A PAPER_RESULTS
# ---- CIFAR-10 ----
PAPER_RESULTS["cifar10|convnet|1"]="23.5 ± 0.3"
PAPER_RESULTS["cifar10|convnet|10"]="50.2 ± 0.3"
PAPER_RESULTS["cifar10|convnet|50"]="68.4 ± 0.1"
PAPER_RESULTS["cifar10|resnet18|1"]="22.9 ± 0.4"
PAPER_RESULTS["cifar10|resnet18|10"]="37.1 ± 0.3"
PAPER_RESULTS["cifar10|resnet18|50"]="62.1 ± 0.1"
PAPER_RESULTS["cifar10|resnet101|1"]="18.7 ± 0.1"
PAPER_RESULTS["cifar10|resnet101|10"]="33.7 ± 0.3"
PAPER_RESULTS["cifar10|resnet101|50"]="51.6 ± 0.4"
# ---- CIFAR-100 ----
PAPER_RESULTS["cifar100|convnet|1"]="19.6 ± 0.3"
PAPER_RESULTS["cifar100|convnet|10"]="48.1 ± 0.3"
PAPER_RESULTS["cifar100|convnet|50"]="57.0 ± 0.1"
PAPER_RESULTS["cifar100|resnet18|1"]="11.0 ± 0.3"
PAPER_RESULTS["cifar100|resnet18|10"]="42.6 ± 0.2"
PAPER_RESULTS["cifar100|resnet18|50"]="62.6 ± 0.1"
PAPER_RESULTS["cifar100|resnet101|1"]="10.8 ± 0.1"
PAPER_RESULTS["cifar100|resnet101|10"]="41.1 ± 0.2"
PAPER_RESULTS["cifar100|resnet101|50"]="63.4 ± 0.3"
# ---- ImageNette ----
PAPER_RESULTS["imagenet-nette|convnet|1"]="33.8 ± 0.8"
PAPER_RESULTS["imagenet-nette|convnet|10"]="63.2 ± 0.7"
PAPER_RESULTS["imagenet-nette|convnet|50"]="83.8 ± 0.2"
PAPER_RESULTS["imagenet-nette|resnet18|1"]="35.8 ± 1.0"
PAPER_RESULTS["imagenet-nette|resnet18|10"]="61.4 ± 0.4"
PAPER_RESULTS["imagenet-nette|resnet18|50"]="80.4 ± 0.4"
PAPER_RESULTS["imagenet-nette|resnet101|1"]="25.1 ± 2.7"
PAPER_RESULTS["imagenet-nette|resnet101|10"]="54.0 ± 0.4"
PAPER_RESULTS["imagenet-nette|resnet101|50"]="75.0 ± 1.2"
# ---- ImageWoof ----
PAPER_RESULTS["imagenet-woof|convnet|1"]="18.5 ± 0.9"
PAPER_RESULTS["imagenet-woof|convnet|10"]="40.6 ± 2.0"
PAPER_RESULTS["imagenet-woof|convnet|50"]="61.5 ± 0.3"
PAPER_RESULTS["imagenet-woof|resnet18|1"]="20.8 ± 1.2"
PAPER_RESULTS["imagenet-woof|resnet18|10"]="38.5 ± 2.1"
PAPER_RESULTS["imagenet-woof|resnet18|50"]="68.5 ± 0.7"
PAPER_RESULTS["imagenet-woof|resnet101|1"]="19.6 ± 1.8"
PAPER_RESULTS["imagenet-woof|resnet101|10"]="31.3 ± 1.3"
PAPER_RESULTS["imagenet-woof|resnet101|50"]="59.1 ± 0.7"
# ---- Tiny-ImageNet ----
PAPER_RESULTS["tinyimagenet|convnet|1"]="12.0 ± 0.1"
PAPER_RESULTS["tinyimagenet|convnet|10"]="39.6 ± 0.1"
PAPER_RESULTS["tinyimagenet|convnet|50"]="47.6 ± 0.2"
PAPER_RESULTS["tinyimagenet|resnet18|1"]="9.7 ± 0.4"
PAPER_RESULTS["tinyimagenet|resnet18|10"]="41.9 ± 0.2"
PAPER_RESULTS["tinyimagenet|resnet18|50"]="58.2 ± 0.1"
PAPER_RESULTS["tinyimagenet|resnet101|1"]="3.8 ± 0.1"
PAPER_RESULTS["tinyimagenet|resnet101|10"]="22.9 ± 3.3"
PAPER_RESULTS["tinyimagenet|resnet101|50"]="41.2 ± 0.4"
# ---- ImageNet-100 ----
PAPER_RESULTS["imagenet-100|convnet|1"]="7.1 ± 0.2"
PAPER_RESULTS["imagenet-100|convnet|10"]="29.6 ± 0.1"
PAPER_RESULTS["imagenet-100|convnet|50"]="50.2 ± 0.2"
PAPER_RESULTS["imagenet-100|resnet18|1"]="8.1 ± 0.3"
PAPER_RESULTS["imagenet-100|resnet18|10"]="36.0 ± 0.3"
PAPER_RESULTS["imagenet-100|resnet18|50"]="61.6 ± 0.1"
PAPER_RESULTS["imagenet-100|resnet101|1"]="6.1 ± 0.8"
PAPER_RESULTS["imagenet-100|resnet101|10"]="33.9 ± 0.1"
PAPER_RESULTS["imagenet-100|resnet101|50"]="66.0 ± 0.6"
# ---- ImageNet-1K ----
PAPER_RESULTS["imagenet-1k|convnet|1"]="6.4 ± 0.1"
PAPER_RESULTS["imagenet-1k|convnet|10"]="20.4 ± 0.1"
PAPER_RESULTS["imagenet-1k|convnet|50"]="38.4 ± 0.2"
PAPER_RESULTS["imagenet-1k|resnet18|1"]="6.6 ± 0.2"
PAPER_RESULTS["imagenet-1k|resnet18|10"]="42.0 ± 0.1"
PAPER_RESULTS["imagenet-1k|resnet18|50"]="56.5 ± 0.1"
PAPER_RESULTS["imagenet-1k|resnet101|1"]="5.9 ± 0.4"
PAPER_RESULTS["imagenet-1k|resnet101|10"]="48.3 ± 1.0"
PAPER_RESULTS["imagenet-1k|resnet101|50"]="61.2 ± 0.4"

# =============================================================================
# Result extraction helpers
# =============================================================================
format_cell() {
    local val="$1"
    if [[ "$val" == *"±"* ]]; then
        printf "%11s" "$val"
    else
        printf "%10s" "$val"
    fi
}

# Extract numeric mean from "X ± Y" or "X" for comparison
extract_mean() {
    local val="$1"
    if [[ -z "$val" || "$val" == "-" ]]; then
        echo ""
        return
    fi
    echo "$val" | grep -oE '[0-9]+\.?[0-9]*' | head -1
}

# Format cell with ANSI color: green if val > paper, red if val < paper
format_cell_colored() {
    local val="$1"
    local paper_val="$2"
    local formatted
    formatted=$(format_cell "$val")

    if [[ "$NO_COLOR" == true ]]; then
        printf "%s" "$formatted"
        return
    fi

    local val_mean paper_mean
    val_mean=$(extract_mean "$val")
    paper_mean=$(extract_mean "$paper_val")

    if [[ -z "$val_mean" || -z "$paper_mean" ]]; then
        printf "%s" "$formatted"
        return
    fi

    if awk "BEGIN {exit !($val_mean > $paper_mean)}" 2>/dev/null; then
        printf '\033[32m%s\033[0m' "$formatted"
    elif awk "BEGIN {exit !($val_mean < $paper_mean)}" 2>/dev/null; then
        printf '\033[31m%s\033[0m' "$formatted"
    else
        printf "%s" "$formatted"
    fi
}

extract_best_acc() {
    local log_file="$1"
    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi
    grep -oP 'Best accuracy is \K[0-9]+\.?[0-9]*' "$log_file" | tail -1
}

# mean/std aggregation for a set of log files
aggregate_logs() {
    local files=("$@")
    local accs=()

    for log_file in "${files[@]}"; do
        local acc
        acc=$(extract_best_acc "$log_file")
        if [[ -n "$acc" ]]; then
            accs+=("$acc")
        fi
    done

    if [[ ${#accs[@]} -eq 0 ]]; then
        echo "-"
        return
    fi

    local result
    result=$(printf '%s\n' "${accs[@]}" | LC_ALL=C awk '{
        sum += $1; sumsq += $1*$1; n++
    } END {
        mean = sum / n
        if (n > 1) {
            std = sqrt((sumsq - sum*sum/n) / (n-1))
            printf "%.1f ± %.1f", mean, std
        } else {
            printf "%.1f", mean
        }
    }')
    echo "$result"
}

get_result_baseline() {
    local subset="$1"
    local arch_label="$2"
    local ipc="$3"

    local files=()
    for seed in $SEEDS; do
        files+=("${LOG_DIR_BASELINE}/${subset}_${arch_label}_ipc${ipc}_seed${seed}.log")
    done
    aggregate_logs "${files[@]}"
}

get_result_uocce() {
    local subset="$1"
    local arch_label="$2"
    local ipc="$3"

    local files=()
    for seed in $SEEDS; do
        files+=("${LOG_DIR_UOCCE}/${subset}_${arch_label}_ipc${ipc}_seed${seed}_uocce.log")
    done
    aggregate_logs "${files[@]}"
}

get_result_socce() {
    local subset="$1"
    local arch_label="$2"
    local ipc="$3"

    local files=()
    for seed in $SEEDS; do
        files+=("${LOG_DIR_SOCCE}/${subset}_${arch_label}_ipc${ipc}_seed${seed}_socce.log")
    done
    aggregate_logs "${files[@]}"
}

get_result_mocce() {
    local subset="$1"
    local arch_label="$2"
    local ipc="$3"

    local files=()
    for seed in $SEEDS; do
        files+=("${LOG_DIR_MOCCE}/${subset}_${arch_label}_ipc${ipc}_seed${seed}_mocce.log")
    done
    aggregate_logs "${files[@]}"
}

# =============================================================================
# Print unified results table (Paper | Replicated | U-OCCE | S-OCCE | M-OCCE)
# =============================================================================
print_results_table() {
    local DATASETS=("cifar10" "cifar100" "imagenet-nette" "imagenet-woof" "tinyimagenet" "imagenet-100" "imagenet-1k")
    local DISPLAY=("CIFAR-10" "CIFAR-100" "ImageNette" "ImageWoof" "Tiny-ImageNet" "ImageNet-100" "ImageNet-1K")
    local IPCS=(1 10 50)

    local SEP_5="────────────┬────────────┬────────────┬────────────┬────────────"
    local SPAN_5="────────────────────────────────────────────────────────────────"
    local SEP_ALL="────────────┼────────────┼────────────┼────────────┼────────────"

    echo ""
    echo "┌───────────────┬─────┬${SPAN_5}┬${SPAN_5}┬${SPAN_5}┐"
    echo "│               │     │                            ConvNet                             │                           ResNet-18                            │                          ResNet-101                            │"
    echo "│   Dataset     │ IPC ├${SEP_5}┼${SEP_5}┼${SEP_5}┤"
    echo "│               │     │    RDED    │ Replicated │   U-OCCE   │   S-OCCE   │   M-OCCE   │    RDED    │ Replicated │   U-OCCE   │   S-OCCE   │   M-OCCE   │    RDED    │ Replicated │   U-OCCE   │   S-OCCE   │   M-OCCE   │"
    echo "├───────────────┼─────┼${SEP_ALL}┼${SEP_ALL}┼${SEP_ALL}┤"

    for d_idx in "${!DATASETS[@]}"; do
        local ds="${DATASETS[$d_idx]}"
        local ds_display="${DISPLAY[$d_idx]}"

        for ipc_idx in "${!IPCS[@]}"; do
            local ipc="${IPCS[$ipc_idx]}"

            local label=""
            if [[ $ipc_idx -eq 0 ]]; then
                label="$ds_display"
            fi

            local paper_conv="${PAPER_RESULTS[${ds}|convnet|${ipc}]:-"-"}"
            local paper_rn18="${PAPER_RESULTS[${ds}|resnet18|${ipc}]:-"-"}"
            local paper_rn101="${PAPER_RESULTS[${ds}|resnet101|${ipc}]:-"-"}"

            local conv_label
            case "$ds" in
                cifar10|cifar100) conv_label="conv3" ;;
                tinyimagenet)     conv_label="conv4" ;;
                imagenet-nette)   conv_label="conv5" ;;
                imagenet-woof)    conv_label="conv5" ;;
                imagenet-100)     conv_label="conv6" ;;
                imagenet-1k)      conv_label="conv4" ;;
            esac

            local base_conv base_rn18 base_rn101
            base_conv=$(get_result_baseline "$ds" "$conv_label" "$ipc")
            base_rn18=$(get_result_baseline "$ds" "resnet18" "$ipc")
            base_rn101=$(get_result_baseline "$ds" "resnet101" "$ipc")

            local uocce_conv socce_conv mocce_conv
            local uocce_rn18 socce_rn18 mocce_rn18
            local uocce_rn101 socce_rn101 mocce_rn101
            uocce_conv=$(get_result_uocce "$ds" "$conv_label" "$ipc")
            uocce_rn18=$(get_result_uocce "$ds" "resnet18" "$ipc")
            uocce_rn101=$(get_result_uocce "$ds" "resnet101" "$ipc")
            socce_conv=$(get_result_socce "$ds" "$conv_label" "$ipc")
            socce_rn18=$(get_result_socce "$ds" "resnet18" "$ipc")
            socce_rn101=$(get_result_socce "$ds" "resnet101" "$ipc")
            mocce_conv=$(get_result_mocce "$ds" "$conv_label" "$ipc")
            mocce_rn18=$(get_result_mocce "$ds" "resnet18" "$ipc")
            mocce_rn101=$(get_result_mocce "$ds" "resnet101" "$ipc")

            printf "│ %-13s │ %3d │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │ %s │\n" \
                "$label" "$ipc" \
                "$(format_cell "$paper_conv")"  "$(format_cell_colored "$base_conv" "$paper_conv")"  "$(format_cell_colored "$uocce_conv" "$paper_conv")"  "$(format_cell_colored "$socce_conv" "$paper_conv")"  "$(format_cell_colored "$mocce_conv" "$paper_conv")" \
                "$(format_cell "$paper_rn18")"  "$(format_cell_colored "$base_rn18" "$paper_rn18")"  "$(format_cell_colored "$uocce_rn18" "$paper_rn18")"  "$(format_cell_colored "$socce_rn18" "$paper_rn18")"  "$(format_cell_colored "$mocce_rn18" "$paper_rn18")" \
                "$(format_cell "$paper_rn101")" "$(format_cell_colored "$base_rn101" "$paper_rn101")" "$(format_cell_colored "$uocce_rn101" "$paper_rn101")" "$(format_cell_colored "$socce_rn101" "$paper_rn101")" "$(format_cell_colored "$mocce_rn101" "$paper_rn101")"
        done

        if [[ $d_idx -lt $(( ${#DATASETS[@]} - 1 )) ]]; then
            echo "├───────────────┼─────┼${SEP_ALL}┼${SEP_ALL}┼${SEP_ALL}┤"
        fi
    done

    echo "└───────────────┴─────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘"
    echo ""
    echo "  Paper      = reported in Table 2 of the RDED paper"
    printf "  Replicated = KD-only baseline without OCCE (logs: %s/)\n" "${LOG_DIR_BASELINE}"
    printf "  U-OCCE     = KD + uniform one-cold CE, gamma=%s (logs: %s/)\n" "${OCCE_GAMMA}" "${LOG_DIR_UOCCE}"
    printf "  S-OCCE     = KD + soft-proportional one-cold CE, gamma=%s (logs: %s/)\n" "${OCCE_GAMMA}" "${LOG_DIR_SOCCE}"
    printf "  M-OCCE     = KD + margin-proportional one-cold CE, gamma=%s (logs: %s/)\n" "${OCCE_GAMMA}" "${LOG_DIR_MOCCE}"
    echo "  -          = not yet run or log not found"
    echo ""
}

# =============================================================================
# If --results flag, just show the table and exit
# =============================================================================
if [ "$SHOW_RESULTS" = true ]; then
    print_results_table
    exit 0
fi

# =============================================================================
#  Experiment definitions
# =============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  RDED Experiments                                          ║"
echo "║  Datasets: CIFAR-10, CIFAR-100, ImageNette, ImageWoof,     ║"
echo "║            Tiny-ImageNet, ImageNet-100, ImageNet-1K        ║"
echo "║  IPCs:     1, 10, 50                                       ║"
printf "║  Seeds:    %-48s║\n" "${SEEDS}"
printf "║  Mode:     %-48s║\n" "${MODE}"
printf "║  OCCE γ:   %-48s║\n" "${OCCE_GAMMA}"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# CIFAR-10  (10 classes, 32x32, factor=1)
if should_run_dataset "cifar10"; then
    echo ">>> CIFAR-10"

    if should_run_arch "convnet"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "cifar10" "conv3" "conv3" 1 "conv3"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "cifar10" "conv3" "conv3" 1 "conv3"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "cifar10" "conv3" "conv3" 1 "conv3"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "cifar10" "conv3" "conv3" 1 "conv3"
    fi

    if should_run_arch "resnet18"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "cifar10" "resnet18_modified" "resnet18_modified" 1 "resnet18"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "cifar10" "resnet18_modified" "resnet18_modified" 1 "resnet18"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "cifar10" "resnet18_modified" "resnet18_modified" 1 "resnet18"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "cifar10" "resnet18_modified" "resnet18_modified" 1 "resnet18"
    fi

    if should_run_arch "resnet101"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "cifar10" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "cifar10" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "cifar10" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "cifar10" "resnet101_modified" "resnet101_modified" 1 "resnet101"
    fi
fi

# CIFAR-100  (100 classes, 32x32, factor=1)
if should_run_dataset "cifar100"; then
    echo ">>> CIFAR-100"

    if should_run_arch "convnet"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "cifar100" "conv3" "conv3" 1 "conv3"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "cifar100" "conv3" "conv3" 1 "conv3"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "cifar100" "conv3" "conv3" 1 "conv3"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "cifar100" "conv3" "conv3" 1 "conv3"
    fi

    if should_run_arch "resnet18"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "cifar100" "resnet18_modified" "resnet18_modified" 1 "resnet18"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "cifar100" "resnet18_modified" "resnet18_modified" 1 "resnet18"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "cifar100" "resnet18_modified" "resnet18_modified" 1 "resnet18"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "cifar100" "resnet18_modified" "resnet18_modified" 1 "resnet18"
    fi

    if should_run_arch "resnet101"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "cifar100" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "cifar100" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "cifar100" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "cifar100" "resnet101_modified" "resnet101_modified" 1 "resnet101"
    fi
fi

# ImageNette  (10 classes, 224x224 for ResNet / 128x128 for ConvNet, factor=2)
if should_run_dataset "imagenet-nette"; then
    echo ">>> ImageNette"

    if should_run_arch "convnet"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-nette" "conv5" "conv5" 2 "conv5"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-nette" "conv5" "conv5" 2 "conv5"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-nette" "conv5" "conv5" 2 "conv5"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-nette" "conv5" "conv5" 2 "conv5"
    fi

    if should_run_arch "resnet18"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-nette" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-nette" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-nette" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-nette" "resnet18" "resnet18" 2 "resnet18"
    fi

    if should_run_arch "resnet101"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-nette" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-nette" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-nette" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-nette" "resnet101" "resnet101" 2 "resnet101"
    fi
fi

# ImageWoof  (10 classes, 224x224 for ResNet / 128x128 for ConvNet, factor=2)
if should_run_dataset "imagenet-woof"; then
    echo ">>> ImageWoof"

    if should_run_arch "convnet"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-woof" "conv5" "conv5" 2 "conv5"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-woof" "conv5" "conv5" 2 "conv5"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-woof" "conv5" "conv5" 2 "conv5"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-woof" "conv5" "conv5" 2 "conv5"
    fi

    if should_run_arch "resnet18"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-woof" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-woof" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-woof" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-woof" "resnet18" "resnet18" 2 "resnet18"
    fi

    if should_run_arch "resnet101"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-woof" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-woof" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-woof" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-woof" "resnet101" "resnet101" 2 "resnet101"
    fi
fi

# Tiny-ImageNet  (200 classes, 64x64, factor=1)
if should_run_dataset "tinyimagenet"; then
    echo ">>> Tiny-ImageNet"

    if should_run_arch "convnet"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "tinyimagenet" "conv4" "conv4" 1 "conv4"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "tinyimagenet" "conv4" "conv4" 1 "conv4"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "tinyimagenet" "conv4" "conv4" 1 "conv4"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "tinyimagenet" "conv4" "conv4" 1 "conv4"
    fi

    # ResNet-18: use half batch size (50) for ipc=50 to avoid segfaults
    if should_run_arch "resnet18"; then
        for ipc in 1 10 50; do
            if should_run_ipc "$ipc"; then
                for seed in $SEEDS; do
                    if [[ "$ipc" -eq 50 ]]; then
                        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_experiment_baseline  "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18" "--re-batch-size 50"
                        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_experiment_uocce "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18" "--re-batch-size 50"
                        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_experiment_socce "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18" "--re-batch-size 50"
                        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_experiment_mocce "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18" "--re-batch-size 50"
                    else
                        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_experiment_baseline  "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18"
                        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_experiment_uocce "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18"
                        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_experiment_socce "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18"
                        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_experiment_mocce "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                                           1 "$ipc" "$seed" "resnet18"
                    fi
                done
            fi
        done
    fi

    if should_run_arch "resnet101"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "tinyimagenet" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "tinyimagenet" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "tinyimagenet" "resnet101_modified" "resnet101_modified" 1 "resnet101"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "tinyimagenet" "resnet101_modified" "resnet101_modified" 1 "resnet101"
    fi
fi

# ImageNet-100  (100 classes, 224x224 for ResNet / 128x128 for ConvNet, factor=2)
if should_run_dataset "imagenet-100"; then
    echo ">>> ImageNet-100"

    if should_run_arch "convnet"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-100" "conv6" "conv6" 2 "conv6"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-100" "conv6" "conv6" 2 "conv6"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-100" "conv6" "conv6" 2 "conv6"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-100" "conv6" "conv6" 2 "conv6"
    fi

    if should_run_arch "resnet18"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-100" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-100" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-100" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-100" "resnet18" "resnet18" 2 "resnet18"
    fi

    if should_run_arch "resnet101"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-100" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-100" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-100" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-100" "resnet101" "resnet101" 2 "resnet101"
    fi
fi

# ImageNet-1K  (1000 classes, 224x224, factor=2)
if should_run_dataset "imagenet-1k"; then
    echo ">>> ImageNet-1K"

    if should_run_arch "resnet18"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-1k" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-1k" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-1k" "resnet18" "resnet18" 2 "resnet18"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-1k" "resnet18" "resnet18" 2 "resnet18"
    fi

    if should_run_arch "resnet101"; then
        [[ "$MODE" == "baseline" || "$MODE" == "all" ]] && run_all_ipcs_baseline  "imagenet-1k" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-1k" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-1k" "resnet101" "resnet101" 2 "resnet101"
        [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-1k" "resnet101" "resnet101" 2 "resnet101"
    fi

    # ConvNet column for ImageNet-1K is opt-in and requires additional support
    # in load_model(), mirroring scripts/run_table2.sh / run_table2_occe.sh.
    # Uncomment when supported.
    # if should_run_arch "convnet"; then
    #     [[ "$MODE" == "uocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_uocce "imagenet-1k" "conv4" "conv4" 1 "conv4"
    #     [[ "$MODE" == "socce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_socce "imagenet-1k" "conv4" "conv4" 1 "conv4"
    #     [[ "$MODE" == "mocce" || "$MODE" == "occe" || "$MODE" == "all" ]] && run_all_ipcs_mocce "imagenet-1k" "conv4" "conv4" 1 "conv4"
    # fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete                                  ║"
printf "║  Baseline logs: %-43s║\n" "${LOG_DIR_BASELINE}"
printf "║  U-OCCE logs:   %-43s║\n" "${LOG_DIR_UOCCE}"
printf "║  S-OCCE logs:   %-43s║\n" "${LOG_DIR_SOCCE}"
printf "║  M-OCCE logs:   %-43s║\n" "${LOG_DIR_MOCCE}"
echo "╚════════════════════════════════════════════════════════════╝"

if [ "$DRY_RUN" = false ]; then
    print_results_table
fi

