        #!/usr/bin/env bash
        # Table 2 cell: subset=imagenet-woof ipc=50 teacher=resnet18 student=resnet101
        set -e
        cd "$(dirname "$0")/../.."
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
        python ./main.py \
        --subset "imagenet-woof" \
        --arch-name "resnet18" \
        --factor 2 \
        --num-crop 5 \
        --mipc 300 \
        --ipc 50 \
        --stud-name "resnet101" \
        --re-epochs 300 \
--skip-synth
