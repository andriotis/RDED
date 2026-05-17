#!/usr/bin/env bash
# Table 2 cell: subset=cifar10 ipc=50 teacher=resnet18_modified student=resnet18_modified
set -e
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
python ./main.py \
--subset "cifar10" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18_modified" \
--re-epochs 300
