import os
import random
import warnings
import argparse

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
)
from validation.losses import OCCELoss
from validation.nc_metrics import compute_nc_metrics
from validation.results_logger import log_run


def _find_last_linear(model):
    """Return the last nn.Linear module in `model` (unwraps DataParallel)."""
    if isinstance(model, nn.DataParallel):
        model = model.module
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    main_worker(args)


def main_worker(args):
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

    cudnn.benchmark = True

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
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

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
        worker_init_fn=set_worker_sharing_strategy,
    )

    val_loader = torch.utils.data.DataLoader(
        ImageFolder(
            classes=args.classes,
            ipc=args.val_ipc,
            mem=True,
            root=args.val_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize(args.input_size // 7 * 8, antialias=True),
                    transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
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
    for epoch in range(args.re_epochs):
        train(epoch, train_loader, teacher_model, student_model, args)

        if epoch % 10 == 9 or epoch == args.re_epochs - 1:
            if epoch > args.re_epochs * 0.8:
                top1, nc = validate(student_model, args, epoch)
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
    results_path = log_run(args, best_top1=best_acc1, final_top1=last_top1, nc_metrics=last_nc)
    print(f"Logged run to {results_path}")


def train(epoch, train_loader, teacher_model, student_model, args):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    ce_criterion = nn.CrossEntropyLoss() if args.w_ce > 0 else None
    occe_criterion = OCCELoss() if args.w_occe > 0 else None
    # CE/OCCE need a real target class; we apply them to the un-mixed student
    # pred against the original labels (avoids cutmix-label ambiguity).
    need_pred_label_grad = (args.w_ce > 0) or (args.w_occe > 0)

    teacher_model.eval()
    student_model.train()
    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            mix_images, _, _, _ = mix_aug(images, args)

            if not need_pred_label_grad:
                pred_label = student_model(images)

            teacher_logits = teacher_model(mix_images)
            if args.soft_label == "teacher":
                soft_mix_label = F.softmax(teacher_logits / args.temperature, dim=1)
            elif args.soft_label == "one-cold":
                N = teacher_logits.shape[1]
                soft_mix_label = torch.full_like(teacher_logits, 1.0 / (N - 1))
                soft_mix_label.scatter_(1, labels.unsqueeze(1), 0.0)
            elif args.soft_label == "blend":
                teacher_soft = F.softmax(teacher_logits / args.temperature, dim=1)
                N = teacher_logits.shape[1]
                one_cold = torch.full_like(teacher_logits, 1.0 / (N - 1))
                one_cold.scatter_(1, labels.unsqueeze(1), 0.0)
                soft_mix_label = args.blend_alpha * teacher_soft + (1 - args.blend_alpha) * one_cold

        if need_pred_label_grad:
            pred_label = student_model(images)

        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()

        prec1, prec5 = accuracy(pred_label, labels, topk=(1, 5))

        pred_mix_label = student_model(mix_images)

        loss = None
        if args.w_kd > 0:
            soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)
            kd_loss = loss_function_kl(soft_pred_mix_label, soft_mix_label)
            # Preserve bit-identical behavior when w_kd == 1.0 and no other terms.
            loss = kd_loss if args.w_kd == 1.0 else args.w_kd * kd_loss
        if args.w_ce > 0:
            ce_loss = args.w_ce * ce_criterion(pred_label, labels)
            loss = ce_loss if loss is None else loss + ce_loss
        if args.w_occe > 0:
            occe_loss = args.w_occe * occe_criterion(pred_label, labels)
            loss = occe_loss if loss is None else loss + occe_loss

        loss = loss / args.re_accum_steps

        loss.backward()
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        n = images.size(0)
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
    return top1.avg, nc


if __name__ == "__main__":
    pass
    # main(args)
