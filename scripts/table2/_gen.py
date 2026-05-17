"""
Generate replication scripts for Table 2 of the RDED paper.

Each cell in Table 2 is (dataset, IPC, evaluator-arch). Per paper Sec. 5.1 and
the table footnote:
- ConvNet column: arch = Conv-3/4/5/6 depending on dataset; teacher == student.
- ResNet-18 column: ResNet-18 retrieves the distilled data; student = ResNet-18.
- ResNet-101 column: ResNet-18 retrieves the distilled data; student = ResNet-101
  (reuses the same syn_data via --skip-synth).

CIFAR-10/100 and Tiny-ImageNet only have *_modified ResNet checkpoints in the
pretrained-models table, so we use resnet18_modified / resnet101_modified there.

ImageNette and ImageWoof report only IPC=10 in Table 2; other rows are "-".

`factor` follows Sec. 5.1: N=4 (factor=2) for >=64x64 datasets, N=1 (factor=1)
for CIFAR.
"""
import os
from textwrap import dedent

OUT = os.path.dirname(os.path.abspath(__file__))

# (subset, conv_arch, rn18_arch, rn101_arch, factor)
DATASETS = [
    ("cifar10",        "conv3", "resnet18_modified", "resnet101_modified", 1),
    ("cifar100",       "conv3", "resnet18_modified", "resnet101_modified", 1),
    ("tinyimagenet",   "conv4", "resnet18_modified", "resnet101_modified", 2),
    ("imagenet-nette", "conv5", "resnet18",          "resnet101",          2),
    ("imagenet-woof",  "conv5", "resnet18",          "resnet101",          2),
    ("imagenet-100",   "conv6", "resnet18",          "resnet101",          2),
    ("imagenet-1k",    "conv4", "resnet18",          "resnet101",          2),
]

# Table 2 reports RDED at IPC=1, 10, 50 for every dataset.
# The "-" entries in Table 2 are for other methods, not RDED.
IPC_BY_DATASET = {}
DEFAULT_IPCS = [1, 10, 50]

NUM_CROP = 5
MIPC = 300
EPOCHS = 300


def write(path, body):
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, 0o755)


def script_body(subset, arch, stud, factor, ipc, skip_synth=False):
    skip = " \\\n--skip-synth" if skip_synth else ""
    return dedent(f"""\
        #!/usr/bin/env bash
        # Table 2 cell: subset={subset} ipc={ipc} teacher={arch} student={stud}
        set -e
        cd "$(dirname "$0")/../.."
        export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-1}}"
        python ./main.py \\
        --subset "{subset}" \\
        --arch-name "{arch}" \\
        --factor {factor} \\
        --num-crop {NUM_CROP} \\
        --mipc {MIPC} \\
        --ipc {ipc} \\
        --stud-name "{stud}" \\
        --re-epochs {EPOCHS}{skip}
        """)


def main():
    rows = []
    for subset, conv, rn18, rn101, factor in DATASETS:
        ipcs = IPC_BY_DATASET.get(subset, DEFAULT_IPCS)
        for ipc in ipcs:
            # ConvNet column
            name = f"{subset}_ipc{ipc:02d}_conv.sh"
            write(os.path.join(OUT, name),
                  script_body(subset, conv, conv, factor, ipc))
            rows.append(name)
            # ResNet-18 column
            name = f"{subset}_ipc{ipc:02d}_rn18.sh"
            write(os.path.join(OUT, name),
                  script_body(subset, rn18, rn18, factor, ipc))
            rows.append(name)
            # ResNet-101 column (reuses ResNet-18 syn_data via --skip-synth)
            name = f"{subset}_ipc{ipc:02d}_rn101.sh"
            write(os.path.join(OUT, name),
                  script_body(subset, rn18, rn101, factor, ipc, skip_synth=True))
            rows.append(name)
    print(f"Wrote {len(rows)} scripts under {OUT}")
    for r in rows:
        print(" ", r)


if __name__ == "__main__":
    main()
