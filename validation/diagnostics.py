"""Trustworthiness diagnostics for distilled-data-trained models.

Adds the metrics RDED never reports — calibration (ECE), open-set / OOD
detection (OSCR, AUROC, FPR@95) — on top of the Papyan-Han-Donoho neural-collapse
metrics already in ``validation/nc_metrics.py``. Pure numpy/torch, no sklearn.

Score convention everywhere: a *higher* score means *more in-distribution* (ID).
So both MSP (max softmax prob) and the negative free energy follow the same
"higher = ID" rule, and AUROC/OSCR treat ID as the positive class.

Used by two callers that share this one implementation:
  * ``validation/main.py::main_worker`` (post-training, on the in-memory student),
  * ``tools/diagnose_geometry.py`` (teacher + distilled-set geometry, no training).
"""

import numpy as np
import torch
import torch.nn.functional as F

from validation.nc_metrics import compute_nc_metrics
from validation.utils import _find_last_linear, eval_transform


@torch.no_grad()
def collect_outputs(model, loader, capture_features=True):
    """Run `model` over `loader`, returning CPU tensors.

    Returns a dict with:
        logits   [N, K]  raw classifier outputs
        labels   [N]     ground-truth labels (as provided by the loader)
        preds    [N]     argmax predictions
        features [N, D]  penultimate-layer activations (None if not captured)

    Features are captured with the same last-Linear forward hook used by
    validate(); the hook detaches, so this is inference-only.
    """
    model.eval()

    feat_buf = []
    handle = None
    if capture_features:
        fc = _find_last_linear(model)
        if fc is not None:
            handle = fc.register_forward_hook(
                lambda module, inp, out: feat_buf.append(inp[0].detach().cpu())
            )

    logit_buf, label_buf, pred_buf = [], [], []
    try:
        for data, target in loader:
            data = data.cuda()
            output = model(data)
            logit_buf.append(output.detach().cpu())
            label_buf.append(target.detach().cpu().long())
            pred_buf.append(output.argmax(dim=1).detach().cpu())
    finally:
        if handle is not None:
            handle.remove()

    out = {
        "logits": torch.cat(logit_buf, dim=0),
        "labels": torch.cat(label_buf, dim=0),
        "preds": torch.cat(pred_buf, dim=0),
        "features": torch.cat(feat_buf, dim=0) if feat_buf else None,
    }
    return out


def compute_ece(logits, labels, n_bins=15):
    """Expected/Maximum Calibration Error via equal-width confidence bins.

    Returns {ece, mce, avg_conf, acc, overconf_gap}; overconf_gap = avg_conf - acc
    (positive => overconfident).
    """
    probs = F.softmax(logits.float(), dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()

    conf = conf.numpy()
    correct = correct.numpy()
    N = len(conf)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # first bin is closed on the left so conf==0 lands somewhere
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = in_bin.sum()
        if m == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = conf[in_bin].mean()
        gap = abs(bin_acc - bin_conf)
        ece += (m / N) * gap
        mce = max(mce, gap)

    avg_conf = float(conf.mean())
    acc = float(correct.mean())
    return {
        "ece": float(ece),
        "mce": float(mce),
        "avg_conf": avg_conf,
        "acc": acc,
        "overconf_gap": avg_conf - acc,
    }


def msp_scores(logits):
    """Maximum softmax probability (higher => more in-distribution)."""
    return F.softmax(logits.float(), dim=1).max(dim=1).values.numpy()


def energy_scores(logits, T=1.0):
    """Negative free energy = T * logsumexp(logits / T) (higher => more ID).

    The free energy E(x) = -T*logsumexp(logits/T) is low for ID inputs (Liu et al.,
    2020); we return -E so the "higher = ID" convention holds for AUROC/OSCR.
    """
    return (T * torch.logsumexp(logits.float() / T, dim=1)).numpy()


def _rankdata_avg(a):
    """Ranks of `a` (1..N) with ties assigned their average rank."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def ood_metrics(id_scores, ood_scores):
    """AUROC and FPR@95 for ID (positive) vs OOD (negative), higher score = ID.

    AUROC via the Mann-Whitney U statistic (exact, tie-aware). FPR@95 is the
    OOD pass-rate at the threshold that admits 95% of ID samples.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    n_pos, n_neg = len(id_scores), len(ood_scores)

    all_scores = np.concatenate([id_scores, ood_scores])
    ranks = _rankdata_avg(all_scores)
    sum_ranks_pos = ranks[:n_pos].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    # threshold admitting 95% of ID (95% of ID >= thr) => 5th percentile of ID
    thr = np.percentile(id_scores, 5)
    fpr95 = float(np.mean(ood_scores >= thr))

    return {"auroc": float(auroc), "fpr95": fpr95}


def oscr(id_scores, id_correct, ood_scores):
    """Open-Set Classification Rate: area under the CCR-vs-FPR curve.

    CCR(delta) = #{ID correctly classified and score >= delta} / N_id
    FPR(delta) = #{OOD with score >= delta} / N_ood
    Swept over all scores (descending threshold), integrated by trapezoid.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    id_correct = np.asarray(id_correct, dtype=bool)
    N, M = len(id_scores), len(ood_scores)

    scores = np.concatenate([id_scores, ood_scores])
    is_id = np.concatenate([np.ones(N, bool), np.zeros(M, bool)])
    correct = np.concatenate([id_correct, np.zeros(M, bool)])

    order = np.argsort(-scores, kind="mergesort")  # descending threshold
    is_id = is_id[order]
    correct = correct[order]

    ccr = np.cumsum(correct & is_id) / N
    fpr = np.cumsum(~is_id) / M
    ccr = np.concatenate([[0.0], ccr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(ccr, fpr))


class _LabeledWrapper(torch.utils.data.Dataset):
    """Wrap a (img, *) dataset to yield (transform(img), 0) — OOD labels unused."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img = self.base[i][0]
        return self.transform(img), 0


def build_svhn_ood_loader(args, max_samples=10000):
    """SVHN test split as OOD negatives, preprocessed like the ID val_loader.

    SVHN is native 32x32 (matches CIFAR-100); for Tiny-ImageNet (input_size 64)
    the shared eval_transform resizes it up. Downloads to ./data/_torchvision_cache.
    """
    import torchvision.datasets as datasets
    from validation.utils import make_loader_kwargs

    root = args.ood_data_path or "./data/_torchvision_cache"
    base = datasets.SVHN(root=root, split="test", download=True)

    # Deterministic subsample for speed / rough class balance with the ID set.
    if max_samples and len(base) > max_samples:
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(base), generator=g)[:max_samples].tolist()
        base = torch.utils.data.Subset(base, idx)

    ds = _LabeledWrapper(base, eval_transform(args.input_size))
    return torch.utils.data.DataLoader(
        ds,
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        **make_loader_kwargs(args.seed),
    )


def _id_panel(out, num_classes):
    """NC + calibration + top1 for one in-distribution collect_outputs() result."""
    nc = compute_nc_metrics(
        out["features"], out["labels"], num_classes, classifier_preds=out["preds"]
    )
    cal = compute_ece(out["logits"], out["labels"])
    top1 = float(out["preds"].eq(out["labels"]).float().mean().item() * 100.0)
    return {**nc, **cal, "top1": top1, "n": int(len(out["labels"]))}


def feature_geometry(model, loader, num_classes):
    """NC metrics (and basic calibration) of `model` over a single labeled loader.

    Used for the teacher-induced geometry of the distilled set vs a real subset.
    """
    return _id_panel(collect_outputs(model, loader, capture_features=True), num_classes)


def run_diagnostics(model, val_loader, ood_loader, num_classes):
    """Full trustworthiness panel for a classifier: NC + calibration + open-set.

    Args:
        model: classifier (already on CUDA; DataParallel ok).
        val_loader: in-distribution test loader (real val).
        ood_loader: OOD negatives loader (e.g. SVHN).
        num_classes: K.

    Returns a flat dict ready to log: nc1..nc4, ece/mce/avg_conf/acc/overconf_gap,
    top1, and {oscr,auroc,fpr95} for both msp and energy scores.
    """
    idd = collect_outputs(model, val_loader, capture_features=True)
    ood = collect_outputs(model, ood_loader, capture_features=False)

    result = _id_panel(idd, num_classes)
    id_correct = idd["preds"].eq(idd["labels"]).numpy()
    for name, score_fn in (("msp", msp_scores), ("energy", energy_scores)):
        id_s = score_fn(idd["logits"])
        ood_s = score_fn(ood["logits"])
        om = ood_metrics(id_s, ood_s)
        result[f"oscr_{name}"] = oscr(id_s, id_correct, ood_s)
        result[f"auroc_{name}"] = om["auroc"]
        result[f"fpr95_{name}"] = om["fpr95"]
    return result
