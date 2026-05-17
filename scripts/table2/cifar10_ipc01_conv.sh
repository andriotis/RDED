#!/usr/bin/env bash
# Table 2 cell: subset=cifar10 ipc=1 teacher=conv3 student=conv3
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python ./main.py \
--subset "cifar10" \
--arch-name "conv3" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 1 \
--stud-name "conv3" \
--re-epochs 300
