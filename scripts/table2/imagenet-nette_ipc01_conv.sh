#!/usr/bin/env bash
# Table 2 cell: subset=imagenet-nette ipc=1 teacher=conv5 student=conv5
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python ./main.py \
--subset "imagenet-nette" \
--arch-name "conv5" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 1 \
--stud-name "conv5" \
--re-epochs 300
