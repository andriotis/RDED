#!/usr/bin/env bash
# Table 2 cell: subset=imagenet-100 ipc=1 teacher=conv6 student=conv6
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python ./main.py \
--subset "imagenet-100" \
--arch-name "conv6" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 1 \
--stud-name "conv6" \
--re-epochs 300
