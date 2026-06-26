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


def soft_label_stats(logits):
    """Entropy and sharpness of teacher soft-labels.

    Args:
        logits: [N, K] tensor (raw logits; softmax applied internally)
    Returns:
        dict: sl_entropy_mean, sl_entropy_std, sl_conf_mean, sl_eff_n_mean
              sl_eff_n = exp(H) — effective number of classes with non-trivial mass;
              1.0 = perfectly one-hot, K = uniform.
    """
    p = torch.softmax(logits.float(), dim=1)
    H    = -(p * torch.log(p + 1e-12)).sum(dim=1)  # [N] nats
    conf = p.max(dim=1).values                      # [N]
    eff_n = H.exp()                                 # [N] per-sample effective classes
    return {
        "sl_entropy_mean": float(H.mean()),
        "sl_entropy_std":  float(H.std()),
        "sl_conf_mean":    float(conf.mean()),
        "sl_conf_std":     float(conf.std()),
        "sl_eff_n_mean":   float(eff_n.mean()),
        "sl_eff_n_std":    float(eff_n.std()),
    }


def feat_norm_scores(features):
    """L2 norm of penultimate features (higher => more in-distribution).

    A crude *geometry-aware* score that needs no fitting: ID inputs tend to land far
    from the origin (large activations) while OOD inputs collapse toward it. Useful as
    a foil to MSP because it reads the feature magnitude, not the softmax.
    """
    return torch.linalg.norm(features.float(), dim=1).numpy()


def fit_mahalanobis(features, labels, num_classes, shrinkage=1e-2, normalize=True):
    """Class means + a single tied precision for a Mahalanobis OOD score (Lee et al. 2018).

    One shared covariance is estimated by pooling per-class-centered features, then shrunk
    toward a scaled identity so the (possibly high-dim / rank-deficient) covariance inverts
    cleanly:  Sigma_shrunk = (1-a)*Sigma + a*(tr Sigma / D)*I.

    Features are L2-normalized by default. A raw tied-covariance Mahalanobis on these
    flattened high-dim penultimate features is dominated by the x'Px norm term and *inverts*
    (OOD scores "closer" merely because it has smaller feature norm). Normalizing makes the
    score read *directional* (cosine-space) consistency with the class-conditional structure;
    the discarded magnitude is captured separately by feat_norm, so the two are complementary.
    fit_mahalanobis and maha_scores must share the same `normalize` setting.

    Returns (class_means [K, D] float64, precision [D, D] float64). Absent classes get a
    mean of +inf so they can never win the nearest-class score.
    """
    feats = features.detach().to(torch.float64)
    if normalize:
        feats = F.normalize(feats, dim=1)
    labels = labels.detach().long()
    N, D = feats.shape

    means = torch.zeros(num_classes, D, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    means.index_add_(0, labels, feats)
    counts.index_add_(0, labels, torch.ones(N, dtype=torch.float64))
    present = counts > 0
    means[present] = means[present] / counts[present].unsqueeze(1)

    centered = feats - means[labels]
    sigma = centered.t() @ centered / N
    if shrinkage > 0:
        scaled_diag = (torch.trace(sigma) / D)
        sigma = (1.0 - shrinkage) * sigma + shrinkage * scaled_diag * torch.eye(D, dtype=torch.float64)
    precision = torch.linalg.pinv(sigma)
    means[~present] = float("inf")
    return means, precision


def maha_scores(features, class_means, precision, normalize=True):
    """Max over classes of -(Mahalanobis distance)^2 to the class means (higher => ID).

    Computed in the expanded form d_c(x) = x'Px - 2 x'P mu_c + mu_c' P mu_c so it is one
    pair of matmuls over all samples/classes rather than a per-class loop. `normalize` must
    match the value passed to fit_mahalanobis (see its docstring for why we normalize).
    """
    feats = features.detach().to(torch.float64)
    if normalize:
        feats = F.normalize(feats, dim=1)
    present = ~torch.isinf(class_means).any(dim=1)
    mu = class_means[present]                       # [Kp, D]

    Px = feats @ precision                          # [N, D]
    term_x = (feats * Px).sum(dim=1, keepdim=True)  # [N, 1]  x'Px
    Pmu = mu @ precision                            # [Kp, D]
    cross = feats @ Pmu.t()                         # [N, Kp]  x'P mu_c
    term_mu = (mu * Pmu).sum(dim=1)                 # [Kp]     mu_c'P mu_c
    dist2 = term_x - 2.0 * cross + term_mu          # [N, Kp]
    return (-dist2.min(dim=1).values).numpy()       # higher = closer to a class = ID


def within_class_cov(features, labels, num_classes, normalize=True):
    """Pooled within-class covariance of (optionally L2-normalized) features — the tied
    Sigma the Mahalanobis OOD score reads, and the target the momentmatch selector matches.

    Returns [D, D] float64 (no shrinkage; this is the raw geometry, not for inversion). The
    Frobenius norm of the gap between this on a distilled set vs a real subset is the mediator
    that links variance-preserving selection to the downstream Mahalanobis-AUROC gain.
    """
    feats = features.detach().to(torch.float64)
    if normalize:
        feats = F.normalize(feats, dim=1)
    labels = labels.detach().long()
    N, D = feats.shape
    means = torch.zeros(num_classes, D, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    means.index_add_(0, labels, feats)
    counts.index_add_(0, labels, torch.ones(N, dtype=torch.float64))
    present = counts > 0
    means[present] = means[present] / counts[present].unsqueeze(1)
    centered = feats - means[labels]
    return centered.t() @ centered / N


def class_relation_matrix(logits, labels, num_classes):
    """K×K class-relation matrix R: R[i, j] = average teacher soft-label mass class-i samples
    place on class j (Formalization 1 in the trustworthy-distillation thread).

    Row i is the mean softmax vector over all samples labelled i, so R is row-stochastic and its
    off-diagonal carries the inter-class relational geometry. The relmatch selector targets these
    rows; comparing R on a distilled set vs a real reference (see `relation_divergence`) measures
    how well the distilled set preserves that geometry. Returns [K, K] float64 (absent classes get
    a zero row).
    """
    probs = F.softmax(logits.detach().to(torch.float64), dim=1)
    labels = labels.detach().long()
    N = probs.shape[0]
    R = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.float64)
    R.index_add_(0, labels, probs)
    counts.index_add_(0, labels, torch.ones(N, dtype=torch.float64))
    present = counts > 0
    R[present] = R[present] / counts[present].unsqueeze(1)
    return R


def relation_divergence(R_real, R_syn, top_m=5):
    """Divergence between two class-relation matrices (real reference vs distilled set).

    Reports (lower = closer, except the eigen-overlap):
        rel_frob             ||R_real - R_syn||_F                          (overall)
        rel_frob_offdiag     same with diagonals zeroed                    (inter-class only)
        rel_row_cos_offdiag  1 - mean_i cos(off-diagonal rows)  in [0, 2]  (relational *shape*)
        rel_eig_overlap      top-m eigenvector subspace overlap of the     (in [0, 1]; 1 = identical
                             symmetrized (R + R^T)/2, ||U^T V||_F^2 / m      principal relational axes)

    The off-diagonal terms are the headline: stock (top-confidence) selection collapses the
    off-diagonal mass, so relmatch should lower rel_frob_offdiag / rel_row_cos_offdiag and raise
    rel_eig_overlap relative to stock at matched IPC.
    """
    Rr = R_real.detach().to(torch.float64)
    Rs = R_syn.detach().to(torch.float64)
    K = Rr.shape[0]
    rel_frob = float(torch.linalg.norm(Rr - Rs).item())

    eye = torch.eye(K, dtype=torch.bool)
    Rr_off = Rr.clone(); Rr_off[eye] = 0.0
    Rs_off = Rs.clone(); Rs_off[eye] = 0.0
    rel_frob_offdiag = float(torch.linalg.norm(Rr_off - Rs_off).item())

    cos = F.cosine_similarity(Rr_off, Rs_off, dim=1, eps=1e-12)   # per-row, off-diagonal
    rel_row_cos_offdiag = float((1.0 - cos.mean()).item())

    m = max(1, min(int(top_m), K))
    Sr = (Rr + Rr.t()) / 2.0
    Ss = (Rs + Rs.t()) / 2.0
    # eigh returns ascending eigenvalues, so the last m columns are the top-m eigenvectors.
    Ur = torch.linalg.eigh(Sr).eigenvectors[:, -m:]
    Us = torch.linalg.eigh(Ss).eigenvectors[:, -m:]
    rel_eig_overlap = float((torch.linalg.norm(Ur.t() @ Us) ** 2 / m).item())

    return {
        "rel_frob": rel_frob,
        "rel_frob_offdiag": rel_frob_offdiag,
        "rel_row_cos_offdiag": rel_row_cos_offdiag,
        "rel_eig_overlap": rel_eig_overlap,
    }


def fit_temperature(logits, labels):
    """Single calibration scalar T minimizing NLL of softmax(logits / T).

    Deterministic two-stage grid search (coarse geometric grid, then a local refine) over
    T in [~0.05, 10] — no optimizer state, robust, reproducible. T > 1 softens an
    overconfident model.
    """
    logits = logits.detach().float()
    labels = labels.detach().long()

    def nll(T):
        return float(F.cross_entropy(logits / float(T), labels).item())

    coarse = np.geomspace(0.5, 10.0, 60)
    t0 = min(coarse, key=nll)
    fine = np.linspace(max(0.05, t0 * 0.75), t0 * 1.3, 40)
    return float(min(fine, key=nll))


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


def _ood_base_dataset(name, root):
    """Return the raw (PIL, label) OOD test dataset by name, downloading to `root`.

    Sets are grouped along the far/near axis the study reports:
      far  — semantically/modally distant from CIFAR/Tiny natural objects: svhn (digits),
             mnist / fashionmnist (grayscale), places365 (scenes).
      near — natural images / textures overlapping the ID manifold: cifar10, dtd, stl10.
    Each is constructed (and downloaded) only when requested in --ood-sets, so the heavier
    sets (places365 ~2GB, stl10 ~2.5GB) cost nothing unless asked for.
    """
    import torchvision.datasets as datasets
    import torchvision.transforms as T

    # Grayscale sets must become 3-channel PIL before the shared 3-channel eval_transform.
    gray3 = T.Grayscale(num_output_channels=3)

    if name == "svhn":
        return datasets.SVHN(root=root, split="test", download=True)              # far
    if name == "mnist":
        return datasets.MNIST(root=root, train=False, download=True, transform=gray3)        # far
    if name == "fashionmnist":
        return datasets.FashionMNIST(root=root, train=False, download=True, transform=gray3)  # far
    if name == "places365":
        # far-OOD scenes; small (256px) validation split (~2GB on first use).
        return datasets.Places365(root=root, split="val", small=True, download=True)
    if name == "cifar10":
        # near-OOD for CIFAR-100 (natural images, disjoint classes).
        return datasets.CIFAR10(root=root, train=False, download=True)
    if name == "stl10":
        # near-OOD natural images (classes overlap CIFAR); ~2.5GB on first use.
        return datasets.STL10(root=root, split="test", download=True)
    if name == "dtd":
        # Describable Textures — near-OOD textures (torchvision >= 0.13).
        return datasets.DTD(root=root, split="test", download=True)
    raise ValueError(
        f"unknown OOD set '{name}' "
        "(known far: svhn, mnist, fashionmnist, places365; near: cifar10, stl10, dtd)"
    )


def build_ood_loader(name, args, max_samples=10000):
    """A named OOD test set as negatives, preprocessed exactly like the ID val_loader.

    All sets are resized to args.input_size by the shared eval_transform (SVHN/CIFAR-10 are
    native 32x32; DTD is variable). Deterministically subsampled to max_samples for speed.
    Downloads to args.ood_data_path or ./data/_torchvision_cache.
    """
    from validation.utils import make_loader_kwargs

    root = args.ood_data_path or "./data/_torchvision_cache"
    base = _ood_base_dataset(name, root)
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


def build_svhn_ood_loader(args, max_samples=10000):
    """Back-compat wrapper: SVHN test split as OOD negatives (see build_ood_loader)."""
    return build_ood_loader("svhn", args, max_samples)


def build_real_train_fit_loader(args, fit_ipc=None):
    """A held-out slice of the real TRAIN split, preprocessed like the ID val loader.

    The student trained only on the distilled set, so real-train images are genuinely
    unseen by it — a clean held-out ID split for fitting the Mahalanobis class statistics
    and the calibration temperature, without spending any of the real-val test set.
    """
    from validation.utils import ImageFolder, make_loader_kwargs

    if fit_ipc is None:
        fit_ipc = getattr(args, "fit_ipc", 50)
    ds = ImageFolder(
        classes=args.classes,
        ipc=fit_ipc,
        mem=True,
        shuffle=True,
        root=args.train_dir,
        transform=eval_transform(args.input_size),
    )
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


def run_diagnostics(model, val_loader, ood_loader, num_classes, fit_loader=None):
    """Full trustworthiness panel for a classifier: NC + calibration + open-set.

    Args:
        model: classifier (already on CUDA; DataParallel ok).
        val_loader: in-distribution test loader (real val).
        ood_loader: OOD negatives. A single DataLoader (legacy) OR an ordered dict
            {name: loader} for multiple OOD sets.
        num_classes: K.
        fit_loader: optional held-out ID loader (real-train slice). When given, enables the
            geometry-aware Mahalanobis score and temperature scaling, both fit on it so no
            metric is fit and evaluated on the same samples.

    Returns a flat dict ready to log: nc1..nc4, ece/mce/avg_conf/acc/overconf_gap, top1,
    {oscr,auroc,fpr95} for msp / energy / feat_norm (and maha when fit_loader is given),
    and — with fit_loader — temperature, ece_ts, auroc_msp_ts. The ID-only metrics and the
    Mahalanobis/temperature fits are computed once and shared across OOD sets. With a dict
    ood_loader, the OOD-dependent metrics are also returned per set under result["ood"][name];
    the FIRST set is mirrored at the top level so single-loader callers are byte-unchanged.
    """
    if isinstance(ood_loader, dict):
        ood_loaders, multi = dict(ood_loader), True
    else:
        ood_loaders, multi = {"_primary": ood_loader}, False

    idd = collect_outputs(model, val_loader, capture_features=True)
    result = _id_panel(idd, num_classes)
    id_correct = idd["preds"].eq(idd["labels"]).numpy()
    have_feats = idd["features"] is not None

    # score_fn signature: takes a collect_outputs() dict, returns higher=ID scores.
    score_fns = {
        "msp": lambda o: msp_scores(o["logits"]),
        "energy": lambda o: energy_scores(o["logits"]),
    }
    if have_feats:
        score_fns["feat_norm"] = lambda o: feat_norm_scores(o["features"])

    # ID-side fits (Mahalanobis class stats + calibration temperature), shared across sets.
    T, id_msp_t = None, None
    if fit_loader is not None:
        fit = collect_outputs(model, fit_loader, capture_features=have_feats)
        if have_feats and fit["features"] is not None:
            means, precision = fit_mahalanobis(fit["features"], fit["labels"], num_classes)
            score_fns["maha"] = lambda o: maha_scores(o["features"], means, precision)
        T = fit_temperature(fit["logits"], fit["labels"])
        result["temperature"] = T
        result["ece_ts"] = compute_ece(idd["logits"] / T, idd["labels"])["ece"]
        id_msp_t = msp_scores(idd["logits"] / T)

    ood_block = {}
    for i, (name, oloader) in enumerate(ood_loaders.items()):
        ood = collect_outputs(model, oloader, capture_features=have_feats)
        per = {}
        if T is not None:
            ood_msp_t = msp_scores(ood["logits"] / T)
            per["auroc_msp_ts"] = ood_metrics(id_msp_t, ood_msp_t)["auroc"]
        for sname, score_fn in score_fns.items():
            id_s = score_fn(idd)
            ood_s = score_fn(ood)
            om = ood_metrics(id_s, ood_s)
            per[f"oscr_{sname}"] = oscr(id_s, id_correct, ood_s)
            per[f"auroc_{sname}"] = om["auroc"]
            per[f"fpr95_{sname}"] = om["fpr95"]
        if i == 0:
            result.update(per)  # primary OOD mirrored at top level (back-compat)
        ood_block[name] = per
    if multi:
        result["ood"] = ood_block
    return result
