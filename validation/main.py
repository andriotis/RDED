import os
import random
import warnings
import argparse

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from synthesize.utils import load_model
from validation.utils import (
    ImageFolder,
    ShufflePatches,
    mix_aug,
    AverageMeter,
    accuracy,
    get_parameters,
    seed_everything,
    make_loader_kwargs,
    eval_transform,
    IMAGENET_NORM,
    _find_last_linear,
)
from validation.losses import (
    LOSS_REGISTRY,
    MixInfo,
    TERMS_NEEDING_UNMIXED_TEACHER,
    TERMS_NEEDING_FEATURES,
)
from validation.nc_metrics import compute_nc_metrics
from validation.diagnostics import build_svhn_ood_loader, run_diagnostics
from validation.results_logger import log_run
from validation.aim_logger import AimLogger


def forward_capture_feats(model, x, fc):
    """Run `model(x)` while capturing the input to the last Linear (= penultimate
    features) for this single forward.

    Same hook mechanism as the NC capture in validate(), but it returns the live
    tensor WITHOUT detaching, so when called under grad (the student's mixed-image
    pass) gradients flow through the captured features into the loss. The hook is
    scoped to one forward, so it never picks up the separate accuracy-only pass.
    """
    feats = {}
    handle = fc.register_forward_hook(
        lambda module, inp, out: feats.__setitem__("f", inp[0])
    )
    try:
        out = model(x)
    finally:
        handle.remove()
    return out, feats["f"]


def main(args):
    main_worker(args)


def main_worker(args):
    seed_everything(args.seed)
    print("=> using pytorch pre-trained teacher model '{}'".format(args.arch_name))
    teacher_model = load_model(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
    )

    student_model = load_model(
        model_name=args.stud_name,
        dataset=args.subset,
        pretrained=False,
        classes=args.classes,
    )
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    student_model = torch.nn.DataParallel(student_model).cuda()

    teacher_model.eval()
    student_model.train()

    # freeze all layers
    for param in teacher_model.parameters():
        param.requires_grad = False

    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(
            get_parameters(student_model),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            get_parameters(student_model),
            lr=args.adamw_lr,
            betas=[0.9, 0.999],
            weight_decay=args.adamw_weight_decay,
        )

    # lr scheduler
    if args.cos == True:
        scheduler = LambdaLR(
            optimizer,
            lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.re_epochs / 2))
            if step <= args.re_epochs
            else 0,
            last_epoch=-1,
        )
    else:
        scheduler = LambdaLR(
            optimizer,
            lambda step: (1.0 - step / args.re_epochs) if step <= args.re_epochs else 0,
            last_epoch=-1,
        )

    print("process data from {}".format(args.syn_data_path))
    normalize = IMAGENET_NORM

    augment = []
    augment.append(transforms.ToTensor())
    augment.append(ShufflePatches(args.factor))
    augment.append(
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(1 / args.factor, args.max_scale_crops),
            antialias=True,
        )
    )
    augment.append(transforms.RandomHorizontalFlip())
    augment.append(normalize)

    train_dataset = ImageFolder(
        classes=range(args.nclass),
        ipc=args.ipc,
        mem=True,
        shuffle=True,
        root=args.syn_data_path,
        transform=transforms.Compose(augment),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.re_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        **make_loader_kwargs(args.seed),
    )

    val_loader = torch.utils.data.DataLoader(
        ImageFolder(
            classes=args.classes,
            ipc=args.val_ipc,
            mem=True,
            root=args.val_dir,
            transform=eval_transform(args.input_size),
        ),
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        **make_loader_kwargs(args.seed),
    )
    print("load data successfully")

    best_acc1 = 0
    best_epoch = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    last_top1 = 0.0
    last_nc = None
    tracker = AimLogger(args)
    try:
        for epoch in range(args.re_epochs):
            tr_loss, tr_top1, tr_top5, tr_components = train(
                epoch, train_loader, teacher_model, student_model, args
            )
            tracker.log_train(epoch, tr_loss, components=tr_components)

            if epoch % 10 == 9 or epoch == args.re_epochs - 1:
                if epoch > args.re_epochs * 0.8:
                    vl_loss, top1, vl_top5, nc = validate(student_model, args, epoch)
                    tracker.log_val(epoch, vl_loss)
                    last_top1 = top1
                    last_nc = nc
                else:
                    top1 = 0
            else:
                top1 = 0

            scheduler.step()
            if top1 > best_acc1:
                best_acc1 = max(top1, best_acc1)
                best_epoch = epoch

        print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")
        if last_nc is not None:
            print(f"Final NC metrics: {last_nc}")

        # Optional trustworthiness panel (NC + calibration + open-set) on the
        # trained student. Purely additive: does not touch the training above.
        diagnostics = None
        if getattr(args, "diagnostics", False):
            print("=> running diagnostics (ECE / OSCR / AUROC / FPR95 / NC) ...")
            ood_loader = build_svhn_ood_loader(args)
            diagnostics = run_diagnostics(
                student_model, args.val_loader, ood_loader, args.nclass
            )
            print(f"Diagnostics: {diagnostics}")

        results_path = log_run(
            args,
            best_top1=best_acc1,
            final_top1=last_top1,
            nc_metrics=last_nc,
            diagnostics=diagnostics,
        )
        print(f"Logged run to {results_path}")
        tracker.log_hparams(args, best_top1=best_acc1, final_top1=last_top1)
    finally:
        tracker.close()


def train(epoch, train_loader, teacher_model, student_model, args):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer

    # Resolve active losses from the registry once per train() call.
    # weights[name] = w; active_terms = [(name, term_fn, w), ...] where w > 0.
    weights = {name: float(getattr(args, f"w_{name}")) for name in LOSS_REGISTRY}
    active_terms = [
        (name, term_fn, weights[name])
        for name, (term_fn, _default) in LOSS_REGISTRY.items()
        if weights[name] > 0
    ]
    # Monitor terms: computed under no_grad and logged, never in the backward graph.
    # Dedup against active to avoid double-tracking.
    active_names = {n for n, _, _ in active_terms}
    monitor_names = [s.strip() for s in (getattr(args, "monitor", "") or "").split(",") if s.strip()]
    unknown = [n for n in monitor_names if n not in LOSS_REGISTRY]
    if unknown:
        raise ValueError(f"--monitor names not in LOSS_REGISTRY: {unknown}")
    monitor_terms = [
        (name, LOSS_REGISTRY[name][0])
        for name in monitor_names
        if name not in active_names
    ]
    term_meters = {name: AverageMeter() for name, _, _ in active_terms}
    for name, _ in monitor_terms:
        term_meters[name] = AverageMeter()

    # Gate the extra un-mixed teacher forward to terms that need it (currently mxce).
    # Active terms and monitor terms both count.
    needs_unmixed_teacher = (
        {n for n, _, _ in active_terms} | {n for n, _ in monitor_terms}
    ) & TERMS_NEEDING_UNMIXED_TEACHER
    # Gate penultimate-feature capture to relational terms (pkt/spkt). When no
    # feature term is active, the forwards below stay identical to stock RDED.
    needs_features = bool(
        ({n for n, _, _ in active_terms} | {n for n, _ in monitor_terms})
        & TERMS_NEEDING_FEATURES
    )
    fc_student = _find_last_linear(student_model) if needs_features else None
    fc_teacher = _find_last_linear(teacher_model) if needs_features else None

    teacher_model.eval()
    student_model.train()
    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            mix_images, rand_index, lam, _ = mix_aug(images, args)
            teacher_logits_unmixed = teacher_model(images) if needs_unmixed_teacher else None

            pred_label = student_model(images)
            if needs_features:
                teacher_logits, teacher_feats_mixed = forward_capture_feats(
                    teacher_model, mix_images, fc_teacher
                )
            else:
                teacher_logits = teacher_model(mix_images)
                teacher_feats_mixed = None

        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()

        prec1, prec5 = accuracy(pred_label, labels, topk=(1, 5))

        # Student forward on the mixed image (under grad). Capture penultimate
        # features for relational terms; the captured tensor keeps its graph.
        if needs_features:
            pred_mix_label, student_feats_mixed = forward_capture_feats(
                student_model, mix_images, fc_student
            )
        else:
            pred_mix_label = student_model(mix_images)
            student_feats_mixed = None

        # Built after the student pass so it can carry student_feats_mixed; only
        # consumed by the loss terms below, so the later construction is harmless.
        mix_info = MixInfo(
            labels=labels,
            rand_index=rand_index,
            lam=lam if lam is not None else 1.0,
            num_classes=args.nclass,
            teacher_logits_unmixed=teacher_logits_unmixed,
            teacher_feats_mixed=teacher_feats_mixed,
            student_feats_mixed=student_feats_mixed,
        )

        # Composite student loss = sum_i w_i * term_i(student, teacher, T).
        # All terms share args.temperature.
        # Skip the multiply on the first contributing term when w == 1.0
        # (preserves the upstream graph for stock RDED-style w_kl=1.0 runs).
        loss = None
        n = images.size(0)
        for _name, term_fn, w in active_terms:
            term = term_fn(pred_mix_label, teacher_logits, args.temperature, args, mix_info)
            term_meters[_name].update(term.item(), n)
            if loss is None:
                loss = term if w == 1.0 else w * term
            else:
                loss = loss + w * term

        # Monitor pass: compute + log, no gradient contribution.
        if monitor_terms:
            with torch.no_grad():
                for _name, term_fn in monitor_terms:
                    term = term_fn(pred_mix_label, teacher_logits, args.temperature, args, mix_info)
                    term_meters[_name].update(term.item(), n)

        loss = loss / args.re_accum_steps

        loss.backward()
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    printInfo = (
        "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "train_time = {:.6f}".format((time.time() - t1))
    )
    print(printInfo)
    t1 = time.time()
    return (
        objs.avg,
        top1.avg,
        top5.avg,
        {name: m.avg for name, m in term_meters.items()},
    )


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    # Hook the last Linear to capture its input (= penultimate features).
    fc = _find_last_linear(model)
    feat_buf = []
    label_buf = []
    pred_buf = []

    def _hook(module, inp, out):
        feat_buf.append(inp[0].detach().cpu())

    handle = fc.register_forward_hook(_hook) if fc is not None else None

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            label_buf.append(target.detach().cpu())
            pred_buf.append(output.argmax(dim=1).detach().cpu())

    if handle is not None:
        handle.remove()

    nc = None
    if fc is not None and feat_buf:
        features = torch.cat(feat_buf, dim=0)
        labels_all = torch.cat(label_buf, dim=0)
        preds_all = torch.cat(pred_buf, dim=0)
        nc = compute_nc_metrics(features, labels_all, args.nclass, classifier_preds=preds_all)

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    if nc is not None:
        logInfo += "\tNC1 = {:.6f}, NC2 = {:.6f}, NC3 = {:.6f}, NC4 = {}".format(
            nc["nc1"], nc["nc2"], nc["nc3"], nc["nc4"]
        )
    print(logInfo)
    return objs.avg, top1.avg, top5.avg, nc


if __name__ == "__main__":
    pass
    # main(args)
