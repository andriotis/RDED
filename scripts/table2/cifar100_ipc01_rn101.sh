        #!/usr/bin/env bash
        # Table 2 cell: subset=cifar100 ipc=1 teacher=resnet18_modified student=resnet101_modified
        set -e
        cd "$(dirname "$0")/../.."
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
        python ./main.py \
        --subset "cifar100" \
        --arch-name "resnet18_modified" \
        --factor 1 \
        --num-crop 5 \
        --mipc 300 \
        --ipc 1 \
        --stud-name "resnet101_modified" \
        --re-epochs 300 \
--skip-synth
