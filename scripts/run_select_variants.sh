#!/usr/bin/env bash
# Variance-aware selection experiment (Phase 2 — the causal intervention).
#
# For each (method, cell, ipc, seed) it synthesizes the distilled set with the chosen
# stage-1 selector and trains a student WITH --diagnostics, so every run logs the full
# trust panel (ECE/+TS, OSCR/AUROC/FPR95 for msp/energy/feat_norm/maha, NC). Non-stock
# methods land in their own exp/<...>_sel<method>/ dir (path-keyed in argument.py), so
# nothing clobbers the stock control set.
#
# Runs sequentially on ONE GPU (default 0). To parallelize, launch several instances with
# disjoint METHODS/CELLS pinned to different GPUs, e.g.:
#     CUDA_VISIBLE_DEVICES=0 METHODS="stock random"    bash scripts/run_select_variants.sh &
#     CUDA_VISIBLE_DEVICES=1 METHODS="stratified"       bash scripts/run_select_variants.sh &
#     CUDA_VISIBLE_DEVICES=2 METHODS="covmatch"         bash scripts/run_select_variants.sh &
#
# Config via env (defaults shown):
#     METHODS="stock random stratified covmatch"
#     CELLS="cifar100:conv3 cifar100:resnet18_modified tinyimagenet:conv4 tinyimagenet:resnet18_modified"
#     IPCS="1 10 50"
#     SEEDS="42 43 44"
#     RE_EPOCHS=300  MIPC=300  NUM_CROP=5  FACTOR=1
#     RESULTS_FILE=logs/results_select.jsonl
#     SELECT_K=8  FIT_IPC=50
#     DRY_RUN=0   # 1 = print commands only
set -uo pipefail
cd "$(dirname "$0")/.."

METHODS="${METHODS:-stock random stratified covmatch}"
CELLS="${CELLS:-cifar100:conv3 cifar100:resnet18_modified tinyimagenet:conv4 tinyimagenet:resnet18_modified}"
IPCS="${IPCS:-1 10 50}"
SEEDS="${SEEDS:-42 43 44}"
RE_EPOCHS="${RE_EPOCHS:-300}"
MIPC="${MIPC:-300}"; NUM_CROP="${NUM_CROP:-5}"; FACTOR="${FACTOR:-1}"
RESULTS_FILE="${RESULTS_FILE:-logs/results_select.jsonl}"
SELECT_K="${SELECT_K:-8}"; FIT_IPC="${FIT_IPC:-50}"
OOD_SETS="${OOD_SETS:-svhn}"          # comma list; svhn,dtd,cifar10 for the multi-OOD panel
SAVE_STUDENT="${SAVE_STUDENT:-0}"     # 1 = persist student ckpt for --diagnostics-only re-eval
SELECT_BETA="${SELECT_BETA:-0.0}"     # qddpp quality<->diversity knob (one value/invocation; sweep by
                                      # launching several instances with different SELECT_BETA per GPU)
SELECT_QUALITY="${SELECT_QUALITY:-confidence}"   # qddpp quality score: confidence (far-OOD) | margin (near-OOD)
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

for method in $METHODS; do
  for cell in $CELLS; do
    dataset="${cell%%:*}"; arch="${cell##*:}"
    for ipc in $IPCS; do
      for seed in $SEEDS; do
        args=(
          --subset "$dataset" --arch-name "$arch" --stud-name "$arch"
          --factor "$FACTOR" --num-crop "$NUM_CROP" --mipc "$MIPC" --ipc "$ipc"
          --re-epochs "$RE_EPOCHS" --seed "$seed"
          --syn-data-path "syn_data_seed${seed}"
          --select-method "$method"
          --select-k "$SELECT_K"
          --fit-ipc "$FIT_IPC"
          --diagnostics --ood-sets "$OOD_SETS"
          --results-file "$RESULTS_FILE" --disable-aim
        )
        # qddpp needs its one-knob params; tag run-log/cache by them (matches argument.py exp_name).
        qtag=""
        if [[ "$method" == "qddpp" ]]; then
          args+=(--select-beta "$SELECT_BETA" --select-quality "$SELECT_QUALITY")
          qtag="_b${SELECT_BETA}"; [[ "$SELECT_QUALITY" != "confidence" ]] && qtag="${qtag}_q${SELECT_QUALITY}"
        fi
        [[ "$SAVE_STUDENT" == "1" ]] && args+=(--save-student)
        # Reuse an existing seed-scoped distilled set when present (synthesis is deterministic
        # per seed, so the cached set is identical). Stock always reuses; other methods reuse
        # only when SKIP_SYNTH=1 (e.g. a multi-OOD / checkpoint backfill that must not re-synth).
        tag=""; [[ "$method" != "stock" ]] && tag="_sel${method}${qtag}"
        seed_dir="./exp/${dataset}_${arch}_f${FACTOR}_mipc${MIPC}_ipc${ipc}_cr${NUM_CROP}${tag}/syn_data_seed${seed}"
        if [[ -d "$seed_dir" ]] && { [[ "$method" == "stock" ]] || [[ "${SKIP_SYNTH:-0}" == "1" ]]; }; then
          args+=(--skip-synth)
        fi
        echo "[$(date +%H:%M:%S)] gpu=$CUDA_VISIBLE_DEVICES method=$method $dataset/$arch ipc=$ipc seed=$seed -> $RESULTS_FILE"
        if [[ "$DRY_RUN" == "1" ]]; then
          echo "    python ./main.py ${args[*]}"
        else
          python ./main.py "${args[@]}" > "logs/runs/select_${method}${qtag}_${dataset}_${arch}_ipc${ipc}_seed${seed}.log" 2>&1 \
            || echo "    !! FAILED (see log)"
        fi
      done
    done
  done
done
echo "Done. Results -> $RESULTS_FILE"
