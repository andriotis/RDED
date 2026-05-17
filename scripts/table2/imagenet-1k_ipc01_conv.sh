#!/usr/bin/env bash
# Table 2 cell: subset=imagenet-1k ipc=1 teacher=conv4 student=conv4
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python ./main.py \
--subset "imagenet-1k" \
--arch-name "conv4" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 1 \
--stud-name "conv4" \
--re-epochs 300
