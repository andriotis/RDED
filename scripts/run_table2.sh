#!/bin/bash
# =============================================================================
# Master script to reproduce Table 2 (RDED columns) from the paper:
# "On the Diversity and Realism of Distilled Dataset"
#
# Table 2 compares RDED across 7 datasets, 3 IPC values (1, 10, 50),
# and 3 architecture families (ConvNet, ResNet-18, ResNet-101).
#
# Usage:
#   bash scripts/run_table2.sh                          # Run everything
#   bash scripts/run_table2.sh --dataset cifar10        # Single dataset
#   bash scripts/run_table2.sh --arch convnet           # Single arch family
#   bash scripts/run_table2.sh --ipc 10                 # Single IPC
#   bash scripts/run_table2.sh --seeds "42 43 44"       # Multiple seeds
#   bash scripts/run_table2.sh --dry-run                # Print commands only
#   bash scripts/run_table2.sh --results                # Show paper vs your results
#   bash scripts/run_table2.sh --force                  # Rerun even if logs exist
#
# Prerequisites:
#   1. Datasets in ./data/{subset}/train/ and ./data/{subset}/val/
#   2. Pretrained observer models in ./data/pretrain_models/
#      Download from: https://drive.google.com/drive/folders/1HmrheO6MgX453a5UPJdxPHK4UTv-4aVt
#   3. torch==2.1.0, torchvision==0.16.0
#
# NOTE on ResNet-101:
#   - ImageNet-1K uses torchvision pretrained weights (no extra files needed).
#   - All other datasets need custom pretrained checkpoints NOT provided in
#     the Google Drive. You must train them first (see the SRe2L training
#     reference) and place them at:
#       ./data/pretrain_models/{dataset}_resnet101.pth         (ImageNet subsets)
#       ./data/pretrain_models/{dataset}_resnet101_modified.pth (CIFAR/TinyImageNet)
# =============================================================================

set -e

# ---- Defaults ---------------------------------------------------------------
FILTER_DATASET="all"
FILTER_ARCH="all"
FILTER_IPC="all"
SEEDS="42"
DRY_RUN=false
SHOW_RESULTS=false
FORCE=false
RE_EPOCHS=300
NUM_CROP=5
MIPC=300
LOG_DIR="./logs/table2"

# ---- Parse arguments --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)   FILTER_DATASET="$2"; shift 2 ;;
        --arch)      FILTER_ARCH="$2";    shift 2 ;;
        --ipc)       FILTER_IPC="$2";     shift 2 ;;
        --seeds)     SEEDS="$2";          shift 2 ;;
        --dry-run)   DRY_RUN=true;        shift   ;;
        --results)   SHOW_RESULTS=true;   shift   ;;
        --force)     FORCE=true;          shift   ;;
        --log-dir)   LOG_DIR="$2";        shift 2 ;;
        --re-epochs) RE_EPOCHS="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Helpers ----------------------------------------------------------------
# run_experiment subset arch_name stud_name factor ipc seed arch_label [extra_args]
run_experiment() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local ipc="$5"
    local seed="$6"
    local arch_label="$7"
    local extra_args="${8:-}"

    local exp_tag="${subset}_${arch_label}_ipc${ipc}_seed${seed}"
    local log_file="${LOG_DIR}/${exp_tag}.log"

    echo "============================================================"
    echo " Running: ${exp_tag}"
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

    # Skip if log already contains a completed result (unless --force)
    if [[ "$FORCE" = false && -f "$log_file" ]] && grep -q "Best accuracy is" "$log_file"; then
        local prev_acc
        prev_acc=$(grep -oP 'Best accuracy is \K[0-9]+\.?[0-9]*' "$log_file" | tail -1)
        echo "  SKIPPED (already completed, best acc = ${prev_acc}%)"
        echo ""
        return
    fi

    mkdir -p "${LOG_DIR}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${exp_tag}" | tee -a "${log_file}"
    eval "${cmd}" 2>&1 | tee -a "${log_file}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${exp_tag}" | tee -a "${log_file}"
    echo ""
}

should_run_dataset() {
    [[ "$FILTER_DATASET" == "all" || "$FILTER_DATASET" == "$1" ]]
}

should_run_arch() {
    [[ "$FILTER_ARCH" == "all" || "$FILTER_ARCH" == "$1" ]]
}

should_run_ipc() {
    [[ "$FILTER_IPC" == "all" || "$FILTER_IPC" == "$1" ]]
}

# ---- Run experiments for a single (dataset, arch_family) combo --------------
# Arguments: subset, arch_name, stud_name, factor, arch_family_label
run_all_ipcs() {
    local subset="$1"
    local arch_name="$2"
    local stud_name="$3"
    local factor="$4"
    local arch_label="$5"

    for ipc in 1 10 50; do
        if should_run_ipc "$ipc"; then
            for seed in $SEEDS; do
                run_experiment "$subset" "$arch_name" "$stud_name" \
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
# Extract best accuracy from a log file
# Looks for: "Train Finish! Best accuracy is {acc}@{epoch}"
# =============================================================================
# Right-align a cell value to exactly 10 display columns.
# printf counts bytes, but ± is 2 bytes yet only 1 display column wide,
# so we bump the field width by 1 to compensate.
format_cell() {
    local val="$1"
    if [[ "$val" == *"±"* ]]; then
        printf "%11s" "$val"
    else
        printf "%10s" "$val"
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

# =============================================================================
# Collect reproduced result for a (dataset, arch, ipc) across all seeds
# Returns: "mean" or "mean ± std" or "-" if no logs found
# =============================================================================
get_reproduced_result() {
    local subset="$1"
    local arch_label="$2"
    local ipc="$3"

    local accs=()
    for seed in $SEEDS; do
        local log_file="${LOG_DIR}/${subset}_${arch_label}_ipc${ipc}_seed${seed}.log"
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

    # Compute mean and std with awk (LC_ALL=C forces period decimal separator)
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

# =============================================================================
# Print the full results comparison table
# =============================================================================
print_results_table() {
    local DATASETS=("cifar10" "cifar100" "imagenet-nette" "imagenet-woof" "tinyimagenet" "imagenet-100" "imagenet-1k")
    local DISPLAY=("CIFAR-10" "CIFAR-100" "ImageNette" "ImageWoof" "Tiny-ImageNet" "ImageNet-100" "ImageNet-1K")
    local ARCHS=("convnet" "resnet18" "resnet101")
    local IPCS=(1 10 50)

    # Column widths (chars between │ markers):
    #   Dataset=15, IPC=5, each data col=12  (fits " 23.5 ± 0.3 ")
    #   Group span = 12 + 1(│) + 12 = 25

    echo ""
    echo "┌───────────────┬─────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐"
    echo "│               │     │         ConvNet         │        ResNet-18        │       ResNet-101        │"
    echo "│   Dataset     │ IPC ├────────────┬────────────┼────────────┬────────────┼────────────┬────────────┤"
    echo "│               │     │    RDED    │ Replicated │    RDED    │ Replicated │    RDED    │ Replicated │"
    echo "├───────────────┼─────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤"

    for d_idx in "${!DATASETS[@]}"; do
        local ds="${DATASETS[$d_idx]}"
        local ds_display="${DISPLAY[$d_idx]}"

        for ipc_idx in "${!IPCS[@]}"; do
            local ipc="${IPCS[$ipc_idx]}"

            # Dataset name only on first row of each group
            local label=""
            if [[ $ipc_idx -eq 0 ]]; then
                label="$ds_display"
            fi

            local paper_conv="${PAPER_RESULTS[${ds}|convnet|${ipc}]:-"-"}"
            local paper_rn18="${PAPER_RESULTS[${ds}|resnet18|${ipc}]:-"-"}"
            local paper_rn101="${PAPER_RESULTS[${ds}|resnet101|${ipc}]:-"-"}"

            local ours_conv
            local ours_rn18
            local ours_rn101

            # ConvNet arch label varies per dataset
            case "$ds" in
                cifar10|cifar100) ours_conv=$(get_reproduced_result "$ds" "conv3" "$ipc") ;;
                tinyimagenet)     ours_conv=$(get_reproduced_result "$ds" "conv4" "$ipc") ;;
                imagenet-nette)   ours_conv=$(get_reproduced_result "$ds" "conv5" "$ipc") ;;
                imagenet-woof)    ours_conv=$(get_reproduced_result "$ds" "conv5" "$ipc") ;;
                imagenet-100)     ours_conv=$(get_reproduced_result "$ds" "conv6" "$ipc") ;;
                imagenet-1k)      ours_conv=$(get_reproduced_result "$ds" "conv4" "$ipc") ;;
            esac
            ours_rn18=$(get_reproduced_result "$ds" "resnet18" "$ipc")
            ours_rn101=$(get_reproduced_result "$ds" "resnet101" "$ipc")

            printf "│ %-13s │ %3d │ %s │ %s │ %s │ %s │ %s │ %s │\n" \
                "$label" "$ipc" \
                "$(format_cell "$paper_conv")" "$(format_cell "$ours_conv")" \
                "$(format_cell "$paper_rn18")" "$(format_cell "$ours_rn18")" \
                "$(format_cell "$paper_rn101")" "$(format_cell "$ours_rn101")"
        done

        # Separator between datasets (except after the last one)
        if [[ $d_idx -lt $(( ${#DATASETS[@]} - 1 )) ]]; then
            echo "├───────────────┼─────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤"
        fi
    done

    echo "└───────────────┴─────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘"
    echo ""
    echo "  RDED       = reported in Table 2 of the RDED paper"
    echo "  Replicated = reproduced from logs in ${LOG_DIR}/"
    echo "  -      = not yet run or log not found"
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
#  Table 2 experiment definitions
#
#  Architecture mapping (from Table 2 caption):
#    ConvNet column : conv3 (CIFAR), conv4 (TinyImageNet), conv5 (Nette/Woof),
#                     conv6 (ImageNet-100)
#    ResNet-18 col  : resnet18_modified (CIFAR/TinyImageNet),
#                     resnet18 (ImageNet variants)
#    ResNet-101 col : resnet101_modified (CIFAR/TinyImageNet),
#                     resnet101 (ImageNet variants)
#
#  Factor: 1 for resolution <= 64, 2 for resolution >= 128
# =============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Reproducing Table 2 — RDED columns                        ║"
echo "║  Datasets: CIFAR-10, CIFAR-100, ImageNette, ImageWoof,     ║"
echo "║            Tiny-ImageNet, ImageNet-100, ImageNet-1K        ║"
echo "║  IPCs:     1, 10, 50                                       ║"
printf "║  Seeds:    %-48s║\n" "${SEEDS}"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# CIFAR-10  (10 classes, 32x32, factor=1)
# =============================================================================
if should_run_dataset "cifar10"; then
    echo ">>> CIFAR-10"

    # ConvNet column: conv3
    if should_run_arch "convnet"; then
        run_all_ipcs "cifar10" "conv3" "conv3" 1 "conv3"
    fi

    # ResNet-18 column: resnet18_modified
    if should_run_arch "resnet18"; then
        run_all_ipcs "cifar10" "resnet18_modified" "resnet18_modified" 1 "resnet18"
    fi

    # ResNet-101 column: resnet101_modified
    # NOTE: Requires pretrained ./data/pretrain_models/cifar10_resnet101_modified.pth
    if should_run_arch "resnet101"; then
        run_all_ipcs "cifar10" "resnet101_modified" "resnet101_modified" 1 "resnet101"
    fi
fi

# =============================================================================
# CIFAR-100  (100 classes, 32x32, factor=1)
# =============================================================================
if should_run_dataset "cifar100"; then
    echo ">>> CIFAR-100"

    # ConvNet column: conv3
    if should_run_arch "convnet"; then
        run_all_ipcs "cifar100" "conv3" "conv3" 1 "conv3"
    fi

    # ResNet-18 column: resnet18_modified
    if should_run_arch "resnet18"; then
        run_all_ipcs "cifar100" "resnet18_modified" "resnet18_modified" 1 "resnet18"
    fi

    # ResNet-101 column: resnet101_modified
    # NOTE: Requires pretrained ./data/pretrain_models/cifar100_resnet101_modified.pth
    if should_run_arch "resnet101"; then
        run_all_ipcs "cifar100" "resnet101_modified" "resnet101_modified" 1 "resnet101"
    fi
fi

# =============================================================================
# ImageNette  (10 classes, 224x224 for ResNet / 128x128 for ConvNet, factor=2)
# =============================================================================
if should_run_dataset "imagenet-nette"; then
    echo ">>> ImageNette"

    # ConvNet column: conv5 (auto-resolves to 128x128 in argument.py)
    if should_run_arch "convnet"; then
        run_all_ipcs "imagenet-nette" "conv5" "conv5" 2 "conv5"
    fi

    # ResNet-18 column: resnet18 (224x224)
    if should_run_arch "resnet18"; then
        run_all_ipcs "imagenet-nette" "resnet18" "resnet18" 2 "resnet18"
    fi

    # ResNet-101 column: resnet101
    # NOTE: Requires pretrained ./data/pretrain_models/imagenet-nette_resnet101.pth
    if should_run_arch "resnet101"; then
        run_all_ipcs "imagenet-nette" "resnet101" "resnet101" 2 "resnet101"
    fi
fi

# =============================================================================
# ImageWoof  (10 classes, 224x224 for ResNet / 128x128 for ConvNet, factor=2)
# =============================================================================
if should_run_dataset "imagenet-woof"; then
    echo ">>> ImageWoof"

    # ConvNet column: conv5 (auto-resolves to 128x128)
    if should_run_arch "convnet"; then
        run_all_ipcs "imagenet-woof" "conv5" "conv5" 2 "conv5"
    fi

    # ResNet-18 column: resnet18 (224x224)
    if should_run_arch "resnet18"; then
        run_all_ipcs "imagenet-woof" "resnet18" "resnet18" 2 "resnet18"
    fi

    # ResNet-101 column: resnet101
    # NOTE: Requires pretrained ./data/pretrain_models/imagenet-woof_resnet101.pth
    if should_run_arch "resnet101"; then
        run_all_ipcs "imagenet-woof" "resnet101" "resnet101" 2 "resnet101"
    fi
fi

# =============================================================================
# Tiny-ImageNet  (200 classes, 64x64, factor=1)
# =============================================================================
if should_run_dataset "tinyimagenet"; then
    echo ">>> Tiny-ImageNet"

    # ConvNet column: conv4
    if should_run_arch "convnet"; then
        run_all_ipcs "tinyimagenet" "conv4" "conv4" 1 "conv4"
    fi

    # ResNet-18 column: resnet18_modified
    # Use half batch size (50) for ipc=50 to avoid segfaults
    if should_run_arch "resnet18"; then
        for ipc in 1 10 50; do
            if should_run_ipc "$ipc"; then
                for seed in $SEEDS; do
                    if [[ "$ipc" -eq 50 ]]; then
                        run_experiment "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                      1 "$ipc" "$seed" "resnet18" "--re-batch-size 50"
                    else
                        run_experiment "tinyimagenet" "resnet18_modified" "resnet18_modified" \
                                      1 "$ipc" "$seed" "resnet18"
                    fi
                done
            fi
        done
    fi

    # ResNet-101 column: resnet101_modified
    # NOTE: Requires pretrained ./data/pretrain_models/tinyimagenet_resnet101_modified.pth
    if should_run_arch "resnet101"; then
        run_all_ipcs "tinyimagenet" "resnet101_modified" "resnet101_modified" 1 "resnet101"
    fi
fi

# =============================================================================
# ImageNet-100  (100 classes, 224x224 for ResNet / 128x128 for ConvNet, factor=2)
# =============================================================================
if should_run_dataset "imagenet-100"; then
    echo ">>> ImageNet-100"

    # ConvNet column: conv6 (auto-resolves to 128x128)
    if should_run_arch "convnet"; then
        run_all_ipcs "imagenet-100" "conv6" "conv6" 2 "conv6"
    fi

    # ResNet-18 column: resnet18 (224x224)
    if should_run_arch "resnet18"; then
        run_all_ipcs "imagenet-100" "resnet18" "resnet18" 2 "resnet18"
    fi

    # ResNet-101 column: resnet101
    # NOTE: Requires pretrained ./data/pretrain_models/imagenet-100_resnet101.pth
    if should_run_arch "resnet101"; then
        run_all_ipcs "imagenet-100" "resnet101" "resnet101" 2 "resnet101"
    fi
fi

# =============================================================================
# ImageNet-1K  (1000 classes, 224x224, factor=2)
#
# ResNet-18 / ResNet-101: Use torchvision pretrained weights (auto-downloaded).
#
# ConvNet (conv4): The pretrained conv4 model for ImageNet-1K is provided at
#   64x64 resolution, but load_model() does not support loading custom models
#   for imagenet-1k. To run the ConvNet column you must patch load_model() in
#   synthesize/utils.py to add "imagenet-1k" to the dataset list that loads
#   from ./data/pretrain_models/ (line 228). You also need to ensure the
#   input_size is set to 64 when using conv4. This is left as an opt-in.
# =============================================================================
if should_run_dataset "imagenet-1k"; then
    echo ">>> ImageNet-1K"

    # ResNet-18 column: resnet18 (torchvision pretrained)
    if should_run_arch "resnet18"; then
        run_all_ipcs "imagenet-1k" "resnet18" "resnet18" 2 "resnet18"
    fi

    # ResNet-101 column: resnet101 (torchvision pretrained)
    if should_run_arch "resnet101"; then
        run_all_ipcs "imagenet-1k" "resnet101" "resnet101" 2 "resnet101"
    fi

    # ConvNet column: conv4 — REQUIRES CODE PATCH (see note above)
    # Uncomment below after patching load_model() for imagenet-1k conv support
    # if should_run_arch "convnet"; then
    #     run_all_ipcs "imagenet-1k" "conv4" "conv4" 1 "conv4"
    # fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  All experiments complete                                 ║"
printf "║  Logs saved to: %-41s║\n" "${LOG_DIR}"
echo "╚════════════════════════════════════════════════════════════╝"

# Print the comparison table at the end of a real run
if [ "$DRY_RUN" = false ]; then
    print_results_table
fi
