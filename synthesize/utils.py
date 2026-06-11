import math

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision.models as thmodels
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from synthesize.models import ConvNet


# use 0 to pad "other three picture"
def pad(input_tensor, target_height, target_width=None):
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)
    horizontal_padding = target_width - input_tensor.size(3)

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    padded_tensor = F.pad(
        input_tensor, (left_padding, right_padding, top_padding, bottom_padding)
    )

    return padded_tensor


def batched_forward(model, tensor, batch_size):
    total_samples = tensor.size(0)

    all_outputs = []

    model.eval()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i : min(i + batch_size, total_samples)]

            output = model(batch_data)

            all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)

    return final_output


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=224, factor=2):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

denormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict


def cross_entropy(y_pre, y):
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def _forward_logits_feats(model, tensor, batch_size, capture_feats):
    """Batched teacher forward returning (logits, feats|None).

    With capture_feats=False this reproduces ``batched_forward`` exactly (so the stock
    selection path is byte-identical). When features are needed we run the *unwrapped*
    module with a hook on the last Linear's input: registering the hook on a DataParallel
    wrapper would miss the replicas on non-primary GPUs, so we unwrap and forward on one
    device (synthesis is per-class, batches are small).
    """
    if not capture_feats:
        return batched_forward(model, tensor, batch_size), None

    from validation.utils import _find_last_linear

    core = model.module if isinstance(model, nn.DataParallel) else model
    core.eval()
    fc = _find_last_linear(core)
    buf = {}
    handle = fc.register_forward_hook(
        lambda mod, inp, out: buf.__setitem__("f", inp[0].detach())
    )
    logits, feats = [], []
    try:
        with torch.no_grad():
            for i in range(0, tensor.size(0), batch_size):
                out = core(tensor[i : i + batch_size])
                logits.append(out.detach())
                feats.append(buf["f"].flatten(1))
    finally:
        handle.remove()
    return torch.cat(logits, 0), torch.cat(feats, 0)


def _realism_floor(dist, n, floor_mult):
    """Indices (into the per-class candidate pool) of the eligible, realistic crops:
    the ceil(floor_mult * n) lowest teacher-CE-loss candidates (never fewer than n).

    Variance-seeking selectors choose within this floored pool so they trade confidence
    for diversity without ever picking crops the teacher reads as garbage.
    """
    pool = min(dist.numel(), max(n, int(math.ceil(floor_mult * n))))
    return torch.argsort(dist, descending=False)[:pool]


def _kmeans(x, k, gen, iters=10):
    """Tiny Lloyd k-means on CPU features (no sklearn). Returns a [P] cluster assignment.
    Deterministic given `gen` (random-point init)."""
    P = x.shape[0]
    centers = x[torch.randperm(P, generator=gen)[:k]].clone()
    assign = torch.zeros(P, dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(dim=1)
        for c in range(k):
            mask = assign == c
            if mask.any():
                centers[c] = x[mask].mean(dim=0)
    return assign


def _select_stratified(n, dist, feats, k, gen):
    """v0: k-means the floored pool's features, then pick n round-robin across clusters
    (most-confident-first within each), so the selection spans the within-class spread
    instead of piling onto the single most-confident mode.
    """
    P = feats.shape[0]
    k = max(1, min(k, P))
    assign = _kmeans(feats.float(), k, gen)
    buckets = []
    for c in range(k):
        idx = (assign == c).nonzero(as_tuple=True)[0]
        if idx.numel():
            buckets.append(idx[torch.argsort(dist[idx])].tolist())  # ascending loss
    sel, depth = [], 0
    while len(sel) < n:
        progressed = False
        for b in buckets:
            if depth < len(b):
                sel.append(b[depth])
                progressed = True
                if len(sel) >= n:
                    break
        if not progressed:
            break
        depth += 1
    return torch.tensor(sel[:n], dtype=torch.long)


def _select_covmatch(n, feats, gen):
    """v1: greedy log-det (DPP-MAP) maximization of the selected set's feature volume —
    fast greedy of Chen et al. (2018) on the cosine kernel of the floored pool. Maximizing
    log det spreads the picks across the class's feature subspace, which is exactly the
    within-class variance the stock top-confidence selection collapses (so NC1 should rise
    toward real). Optimization-free and deterministic; `gen` is unused.
    """
    n = min(n, feats.shape[0])
    Z = F.normalize(feats.double(), dim=1)
    P = Z.shape[0]
    K = Z @ Z.t() + 1e-4 * torch.eye(P, dtype=torch.float64)

    cis = torch.zeros(n, P, dtype=torch.float64)
    di2 = torch.diagonal(K).clone()
    selected = [int(torch.argmax(di2))]
    for it in range(1, n):
        j = selected[-1]
        if it == 1:
            ei = K[j] / torch.sqrt(di2[j].clamp_min(1e-12))
        else:
            ei = (K[j] - cis[: it - 1].t() @ cis[: it - 1, j]) / torch.sqrt(di2[j].clamp_min(1e-12))
        cis[it - 1] = ei
        di2 = di2 - ei * ei
        di2[selected] = -float("inf")
        selected.append(int(torch.argmax(di2)))
    return torch.tensor(selected[:n], dtype=torch.long)


def _select_variant(method, n, dist, feats, floor_mult, k, gen):
    """Pick n indices (into the per-class pool) by a variance-aware rule, on CPU tensors.

    `dist` [P] is the per-candidate teacher CE loss (realism); `feats` [P, D] the teacher
    penultimate features. All variants first restrict to the realism-floored pool.
    """
    eligible = _realism_floor(dist, n, floor_mult)
    if eligible.numel() <= n:
        return eligible[:n]

    if method == "random":
        perm = torch.randperm(eligible.numel(), generator=gen)[:n]
        return eligible[perm]
    if method == "stratified":
        local = _select_stratified(n, dist[eligible], feats[eligible], k, gen)
        return eligible[local]
    if method == "covmatch":
        local = _select_covmatch(n, feats[eligible], gen)
        return eligible[local]
    raise ValueError(f"unknown --select-method: {method}")


def selector(n, model, images, labels, size, m=5, method="stock",
             floor_mult=3.0, k=8, rng_seed=0):
    """Select n crops per class from the m-crop candidates.

    method="stock" reproduces RDED exactly: keep the most teacher-confident crop per image,
    then take the n globally-most-confident. The variance-aware methods (random/stratified/
    covmatch) instead spread the selection across the class's within-class feature spread,
    over a realism-floored pool, at the same n (= fixed IPC).
    """
    need_feats = method != "stock"
    with torch.no_grad():
        # [mipc, m, C, H, W]
        images = images.cuda()
        s = images.shape

        # [mipc * m, C, H, W], crop-major: row = crop_idx * mipc + img_idx
        images = images.permute(1, 0, 2, 3, 4)
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])

        labels_rep = labels.repeat(m).cuda()

        batch_size = s[0]  # Change it for small GPU memory
        preds, feats = _forward_logits_feats(model, pad(images, size).cuda(), batch_size, need_feats)

        # [m, mipc]
        dist = cross_entropy(preds, labels_rep).reshape(m, s[0])

        # best crop per image -> [mipc]
        index = torch.argmin(dist, 0)
        best_dist = dist[index, torch.arange(s[0])]

        # gather the best crop's image (and feature) per source image
        sa = images.shape
        images = images.reshape(m, s[0], sa[1], sa[2], sa[3])[index, torch.arange(s[0])]
        if need_feats:
            feats = feats.reshape(m, s[0], feats.shape[1])[index, torch.arange(s[0])]

    if method == "stock":
        sel = torch.argsort(best_dist, descending=False)[:n]
    else:
        gen = torch.Generator().manual_seed(int(rng_seed))
        sel = _select_variant(method, n, best_dist.cpu(), feats.cpu(), floor_mult, k, gen)

    torch.cuda.empty_cache()
    return images[sel.to(images.device)].detach()


def mix_images(input_img, out_size, factor, n):
    s = out_size // factor
    remained = out_size % factor
    k = 0
    mixed_images = torch.zeros(
        (n, 3, out_size, out_size),
        requires_grad=False,
        dtype=torch.float,
    )
    h_loc = 0
    for i in range(factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(factor):
            w_r = s + 1 if j < remained else s
            img_part = F.interpolate(
                input_img.data[k * n : (k + 1) * n], size=(h_r, w_r)
            )
            mixed_images.data[
                0:n,
                :,
                h_loc : h_loc + h_r,
                w_loc : w_loc + w_r,
            ] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images


def load_model(model_name="resnet18", dataset="cifar10", pretrained=True, classes=[]):
    def get_model(model_name="resnet18"):
        if "conv" in model_name:
            if dataset in ["cifar10", "cifar100"]:
                size = 32
            elif dataset == "tinyimagenet":
                size = 64
            elif dataset in ["imagenet-nette", "imagenet-woof", "imagenet-100"]:
                size = 128
            else:
                size = 224

            nclass = len(classes)

            model = ConvNet(
                num_classes=nclass,
                net_norm="batch",
                net_act="relu",
                net_pooling="avgpooling",
                net_depth=int(model_name[-1]),
                net_width=128,
                channel=3,
                im_size=(size, size),
            )
        elif model_name == "resnet18_modified":
            model = thmodels.__dict__["resnet18"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif model_name == "resnet101_modified":
            model = thmodels.__dict__["resnet101"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            model = thmodels.__dict__[model_name](pretrained=False)

        return model

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")

        return model

    # "imagenet-100" "imagenet-10" "imagenet-first" "imagenet-nette" "imagenet-woof"
    model = get_model(model_name)
    model = pruning_classifier(model, classes)
    if pretrained:
        if dataset in [
            "imagenet-100",
            "imagenet-10",
            "imagenet-nette",
            "imagenet-woof",
            "tinyimagenet",
            "cifar10",
            "cifar100",
        ]:
            checkpoint = torch.load(
                f"./data/pretrain_models/{dataset}_{model_name}.pth", map_location="cpu"
            )
            model.load_state_dict(checkpoint["model"])
        elif dataset in ["imagenet-1k"]:
            if model_name == "efficientNet-b0":
                # Specifically, for loading the pre-trained EfficientNet model, the following modifications are made
                from torchvision.models._api import WeightsEnum
                from torch.hub import load_state_dict_from_url

                def get_state_dict(self, *args, **kwargs):
                    kwargs.pop("check_hash")
                    return load_state_dict_from_url(self.url, *args, **kwargs)

                WeightsEnum.get_state_dict = get_state_dict

            model = thmodels.__dict__[model_name](pretrained=True)

    return model
