#!/usr/bin/env bash
# Table 2 cell: subset=imagenet-woof ipc=10 teacher=resnet18 student=resnet18
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python ./main.py \
--subset "imagenet-woof" \
--arch-name "resnet18" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18" \
--re-epochs 300
