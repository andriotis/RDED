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


_GAMMA_EPS = 1e-8  # guards against division by zero in soft one-cold normalization

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    main_worker(args)


def occe_loss_fn(student_logits, soft_labels, hard_labels):
    """
    KD and U-OCCE loss terms (unweighted).

    The one-cold target assigns 0 probability to the true class and distributes
    probability uniformly across the remaining C-1 classes (1/(C-1) each),
    matching the formula: L_OCCE = -sum_{k != c} 1/(C-1) * log(p_k).
    """
    # 1. Use standard logits for KL Divergence (KD)
    kd_log_probs = F.log_softmax(student_logits, dim=1)
    kd_loss = F.kl_div(kd_log_probs, soft_labels, reduction="batchmean")

    # 2. Use NEGATED logits for OCCE
    occe_log_probs = F.log_softmax(-student_logits, dim=1)

    num_classes = student_logits.shape[1]
    one_hot = F.one_hot(hard_labels, num_classes=num_classes).float()
    one_cold_target = (1.0 - one_hot) / (num_classes - 1)

    # 3. Apply the one-cold target to the negated log probs
    occe_loss = -(one_cold_target * occe_log_probs).sum(dim=1).mean()

    return kd_loss, occe_loss


def soft_occe_loss_fn(student_logits, soft_labels, hard_labels):
    # 1. Use standard logits for KL Divergence (KD)
    kd_log_probs = F.log_softmax(student_logits, dim=1)
    kd_loss = F.kl_div(kd_log_probs, soft_labels, reduction="batchmean")

    # 2. Use NEGATED logits for OCCE
    occe_log_probs = F.log_softmax(-student_logits, dim=1)

    num_classes = student_logits.shape[1]
    one_hot = F.one_hot(hard_labels, num_classes=num_classes).bool()
    teacher_probs = soft_labels.masked_fill(one_hot, 0.0)
    soft_one_cold_target = teacher_probs / (teacher_probs.sum(dim=1, keepdim=True) + _GAMMA_EPS)

    # 3. Apply the soft one-cold target to the negated log probs
    occe_loss = -(soft_one_cold_target * occe_log_probs).sum(dim=1).mean()

    return kd_loss, occe_loss


def compute_margin_temperature(epoch, total_epochs, T_start, T_final, schedule):
    """Returns the M-OCCE margin temperature for a given epoch.

    Only the margin distribution temperature is annealed; the KD distillation
    temperature remains fixed at args.temperature throughout training.
    """
    if schedule == "linear":
        return T_start + (epoch / total_epochs) * (T_final - T_start)
    if schedule == "cosine":
        return T_final + 0.5 * (T_start - T_final) * (
            1 + math.cos(math.pi * epoch / total_epochs)
        )
    return T_start  # "fixed"


def margin_occe_loss_fn(student_logits, soft_labels, hard_labels,
                        teacher_logits, temperature):
    # 1. Use standard logits for KL Divergence (KD)
    kd_log_probs = F.log_softmax(student_logits, dim=1)
    kd_loss = F.kl_div(kd_log_probs, soft_labels, reduction="batchmean")

    # 2. Use NEGATED logits for OCCE
    occe_log_probs = F.log_softmax(-student_logits, dim=1)

    num_classes = student_logits.shape[1]
    true_logits = teacher_logits[
        torch.arange(teacher_logits.size(0)), hard_labels
    ].unsqueeze(1)
    margin_scores = torch.exp((teacher_logits - true_logits) / temperature)

    one_hot = F.one_hot(hard_labels, num_classes=num_classes).bool()
    margin_scores = margin_scores.masked_fill(one_hot, 0.0)
    margin_one_cold_target = margin_scores / (
        margin_scores.sum(dim=1, keepdim=True) + _GAMMA_EPS
    )

    # 3. Apply the margin one-cold target to the negated log probs
    occe_loss = -(margin_one_cold_target * occe_log_probs).sum(dim=1).mean()

    return kd_loss, occe_loss


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

    for epoch in range(args.re_epochs):
        train(epoch, train_loader, teacher_model, student_model, args)

        top1 = 0
        if epoch % 10 == 9 or epoch == args.re_epochs - 1:
            if epoch > args.re_epochs * 0.8:
                top1 = validate(student_model, args, epoch)

        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_epoch = epoch

    print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")


def train(epoch, train_loader, teacher_model, student_model, args):
    """Generate soft labels and train."""
    objs = AverageMeter()
    objs_kd = AverageMeter()
    objs_occe = AverageMeter()
    objs_temp = AverageMeter()
    objs_gamma_eff = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    use_occe = getattr(args, "use_occe", False)
    gamma = getattr(args, "occe_gamma", 0.1) if use_occe else 0.0

    optimizer = args.optimizer
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    teacher_model.eval()
    student_model.train()
    t1 = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            mix_images, _, _, _ = mix_aug(images, args)

            pred_label = student_model(images)

            teacher_mix_logits = teacher_model(mix_images)
            soft_mix_label = F.softmax(teacher_mix_logits / args.temperature, dim=1)

        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()

        prec1, prec5 = accuracy(pred_label, labels, topk=(1, 5))

        pred_mix_label = student_model(mix_images)

        if use_occe:
            occe_mode = getattr(args, "occe_mode", "uniform")
            if occe_mode == "soft":
                kd_loss_val, occe_loss_val = soft_occe_loss_fn(
                    pred_mix_label / args.temperature, soft_mix_label, labels
                )
            elif occe_mode == "margin":
                T_start = getattr(args, "occe_temp_start", None) or args.temperature
                T_final = getattr(args, "occe_temp_final", 1.0)
                schedule = getattr(args, "occe_temp_schedule", "fixed")
                margin_temp = compute_margin_temperature(
                    epoch, args.re_epochs, T_start, T_final, schedule
                )
                kd_loss_val, occe_loss_val = margin_occe_loss_fn(
                    pred_mix_label / args.temperature, soft_mix_label, labels,
                    teacher_mix_logits, margin_temp,
                )
            else:  # uniform
                kd_loss_val, occe_loss_val = occe_loss_fn(
                    pred_mix_label / args.temperature, soft_mix_label, labels
                )

            # Compute total loss
            if occe_mode == "uniform":
                # Adaptive normalization: γ_eff = r · (L_KL / L_OCCE)
                with torch.no_grad():
                    adaptive_gamma = gamma * (kd_loss_val / (occe_loss_val + 1e-8))
                total_loss = kd_loss_val + adaptive_gamma * occe_loss_val
            else:
                # Soft / Margin: fixed weighting
                total_loss = kd_loss_val + gamma * occe_loss_val
        else:
            soft_pred_mix_label = F.log_softmax(
                pred_mix_label / args.temperature, dim=1
            )
            kd_loss_val = loss_function_kl(soft_pred_mix_label, soft_mix_label)
            occe_loss_val = None
            total_loss = kd_loss_val

        loss = total_loss / args.re_accum_steps

        loss.backward()
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        n = images.size(0)
        objs.update(loss.item(), n)
        objs_kd.update(kd_loss_val.item(), n)
        if occe_loss_val is not None:
            objs_occe.update(occe_loss_val.item(), n)
            if occe_mode == "uniform":
                objs_gamma_eff.update(adaptive_gamma.item(), n)
            if occe_mode == "margin":
                objs_temp.update(margin_temp, n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    if use_occe:
        occe_mode = getattr(args, "occe_mode", "uniform")
        extra_field = ""
        if occe_mode == "uniform":
            extra_field = "\u03b3_eff = {:.6f},\t".format(objs_gamma_eff.avg)
        elif occe_mode == "margin":
            extra_field = "margin_T = {:.4f},\t".format(objs_temp.avg)
        printInfo = (
            "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
            + "KL = {:.6f},\t".format(objs_kd.avg)
            + "OCCE = {:.6f},\t".format(objs_occe.avg)
            + extra_field
            + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
            + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
            + "train_time = {:.6f}".format((time.time() - t1))
        )
    else:
        printInfo = (
            "TRAIN Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
            + "KL = {:.6f},\t".format(objs_kd.avg)
            + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
            + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
            + "train_time = {:.6f}".format((time.time() - t1))
        )
    print(printInfo)


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

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

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    print(logInfo)

    return top1.avg


if __name__ == "__main__":
    pass
    # main(args)
