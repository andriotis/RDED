import os
import fcntl
import random
import argparse
import collections
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from synthesize.utils import *
from validation.utils import ImageFolder, make_loader_kwargs


def init_images(args, model=None):
    trainset = ImageFolder(
        classes=args.classes,
        ipc=args.mipc,
        shuffle=True,
        root=args.train_dir,
        transform=None,
    )

    trainset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MultiRandomCrop(
                num_crop=args.num_crop, size=args.input_size, factor=args.factor
            ),
            normalize,
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        **make_loader_kwargs(args.seed),
    )

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        images = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
            method=getattr(args, "select_method", "stock"),
            k=getattr(args, "select_k", 8),
            rng_seed=args.seed * 100003 + c,
            mean_weight=getattr(args, "momentmatch_mean_weight", 1.0),
            beta=getattr(args, "select_beta", 0.0),
            quality=getattr(args, "select_quality", "confidence"),
            diag_weight=getattr(args, "relmatch_diag_weight", 0.0),
        )
        images = mix_images(images, args.input_size, args.factor, args.ipc)
        save_images(args, denormalize(images), c)


def save_images(args, images, class_id):
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:05d}.jpg".format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main(args):
    print(args)
    with torch.no_grad():
        _synthesize_locked(args)


def _synthesize_locked(args):
    """Synthesize the distilled set at args.syn_data_path, guarded against
    concurrent sweep cells that target the same directory.

    Concurrency model (run_sweep.py may run many cells in parallel, packed
    across GPU slots):
      * A per-path flock serializes any two processes writing the same dir, so
        one cell's rmtree can never race another's synth-write or train-read.
      * A ``.done`` sentinel records the synth "run id" (env RDED_SYNTH_RUN_ID,
        set once per sweep invocation by run_sweep.py). Another cell of the
        *same* sweep whose synth key resolves to this same path reuses the data
        instead of re-synthesizing it — e.g. weight-variant cells that share a
        seed (loss_matrix/gce sweeps) synth once, not once per cell.
      * With no run id (direct ``python main.py`` / diagnose.sh), the sentinel
        never matches, so behavior is the legacy "always re-synthesize".

    The synth key is fully encoded by the path: exp_name carries
    (subset, arch, factor, mipc, ipc, num_crop) and the leaf carries the seed,
    so two cells share a path iff their distilled sets are byte-identical
    (synthesis is deterministic per seed).
    """
    syn = args.syn_data_path
    parent = os.path.dirname(syn) or "."
    leaf = os.path.basename(syn)
    os.makedirs(parent, exist_ok=True)
    lock_path = os.path.join(parent, f".{leaf}.synth.lock")
    done_path = os.path.join(parent, f".{leaf}.synth.done")
    run_id = os.environ.get("RDED_SYNTH_RUN_ID", "")

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        if (
            run_id
            and os.path.isfile(done_path)
            and os.path.isdir(syn)
            and os.listdir(syn)
        ):
            with open(done_path) as f:
                if f.read().strip() == run_id:
                    print(f"[synth] reuse existing distilled set at {syn} (run_id={run_id})")
                    return

        if os.path.exists(syn):
            shutil.rmtree(syn)
        os.makedirs(syn)

        model_teacher = load_model(
            model_name=args.arch_name,
            dataset=args.subset,
            pretrained=True,
            classes=args.classes,
        )

        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False

        init_images(args, model_teacher)

        with open(done_path, "w") as f:
            f.write(run_id)


if __name__ == "__main__":
    pass
