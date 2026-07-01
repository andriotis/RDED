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
    """v0: k-means the pool's features, then pick n round-robin across clusters
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
    fast greedy of Chen et al. (2018) on the cosine kernel of the candidate pool. Maximizing
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


def _select_momentmatch(n, cand_feats, target_feats, mean_weight, gen):
    """v2: greedy moment-matching — choose the n candidates whose selected-set feature
    mean+second-moment best match the class's *full-pool* moments, in L2-normalized
    (cosine) space (same normalization as _select_covmatch / fit_mahalanobis).

    Where covmatch maximizes the *volume* of the candidate pool (independent of the target
    distribution's shape), momentmatch minimizes

        J(S) = ||M_S - M_t||_F^2  +  mean_weight * ||mu_S - mu_t||^2

    with M = (1/|.|) sum z z^T the uncentered second moment and mu the mean. Matching both
    M and mu is equivalent to matching the centered covariance Sigma = M - mu mu^T — the
    exact statistic the tied-covariance Mahalanobis OOD score reads — so the selected set
    reproduces the class's anisotropic spread instead of merely spanning it.

    Greedy forward selection: each pick adds one z_j, a rank-1 update to the running
    G = sum_{i in S} z_i z_i^T. The j-dependent part of ||(G + z_j z_j^T)/k - M_t||_F^2 is
    (2/k^2) z_j^T G z_j + 1/k^2 - (2/k) z_j^T M_t z_j; maintaining ZcG = Zc @ G by a rank-1
    update keeps every step O(P*D). Deterministic; `gen` is unused.

    cand_feats [P, D]: the full per-class selection candidates.
    target_feats [M, D]: the full per-class pool defining (mu_t, M_t); M >= P.
    Returns n distinct LOCAL indices into cand_feats.
    """
    n = min(n, cand_feats.shape[0])
    Zc = F.normalize(cand_feats.double(), dim=1)          # [P, D] candidates
    Zt = F.normalize(target_feats.double(), dim=1)        # [M, D] target pool
    P, D = Zc.shape

    mu_t = Zt.mean(dim=0)                                  # [D]
    M_t = (Zt.t() @ Zt) / Zt.shape[0]                      # [D, D] uncentered 2nd moment
    quad_Mt = ((Zc @ M_t) * Zc).sum(dim=1)                # [P]  z_j^T M_t z_j  (constant)

    s1 = torch.zeros(D, dtype=torch.float64)               # running sum of selected z
    ZcG = torch.zeros(P, D, dtype=torch.float64)           # Zc @ G, G = sum z_i z_i^T
    avail = torch.ones(P, dtype=torch.bool)
    selected = []
    for step in range(1, n + 1):
        k = float(step)
        mu = (s1.unsqueeze(0) + Zc) / k                    # [P, D] candidate-augmented mean
        mean_term = ((mu - mu_t.unsqueeze(0)) ** 2).sum(dim=1)         # [P]
        quad_G = (ZcG * Zc).sum(dim=1)                     # [P]  z_j^T G z_j
        sm_term = (2.0 / k**2) * quad_G + 1.0 / k**2 - (2.0 / k) * quad_Mt
        J = sm_term + mean_weight * mean_term              # [P]  (shared consts dropped)
        J[~avail] = float("inf")
        j = int(torch.argmin(J))
        selected.append(j)
        avail[j] = False
        zj = Zc[j]                                         # [D]
        ZcG += (Zc @ zj).unsqueeze(1) * zj.unsqueeze(0)    # rank-1: G += zj zj^T
        s1 += zj
    return torch.tensor(selected[:n], dtype=torch.long)


def _select_qddpp(n, feats, qual_raw, beta, gen):
    """v3: quality-diversity DPP — greedy log-det MAP of the L-ensemble

        L = diag(q) (Z Z^T) diag(q),   q_i = exp(beta * r_i),   r = -standardize(qual_raw),

    on L2-normalized features (same cosine kernel as _select_covmatch). The diversity term
    Z Z^T is exactly covmatch's feature-volume objective (restores within-class variance /
    NC1); the quality q is a *soft* confidence weighting that up-weights the
    desirable crops (low `qual_raw`: high teacher confidence, or small margin in boundary
    mode). beta is the single continuous knob between the two regimes:

        beta -> 0    : q -> 1, L -> Z Z^T            => reproduces covmatch (pure volume)
        beta -> inf  : quality dominates the log-det  => collapses onto the top-quality crops
                       (= stock when qual_raw is the teacher CE loss)

    so stock and covmatch are the endpoints of one method and intermediate beta *targets*
    real within-class variance instead of overshooting it. Same Chen et al. (2018) fast
    greedy as _select_covmatch, on L instead of K. Optimization-free, deterministic; `gen`
    is unused (kept for a uniform variant signature).

    feats [P, D]: the full per-class selection candidates' features.
    qual_raw [P]: per-candidate "lower is better" score (teacher CE loss, or top1-top2 margin).
    Returns n distinct LOCAL indices into feats.
    """
    n = min(n, feats.shape[0])
    Z = F.normalize(feats.double(), dim=1)
    P = Z.shape[0]
    r = qual_raw.double()
    r = -(r - r.mean()) / r.std().clamp_min(1e-12)         # high r <=> desirable (low qual_raw)
    q = torch.exp(float(beta) * r)                          # [P] quality weights
    L = (q.unsqueeze(1) * q.unsqueeze(0)) * (Z @ Z.t()) + 1e-4 * torch.eye(P, dtype=torch.float64)

    cis = torch.zeros(n, P, dtype=torch.float64)
    di2 = torch.diagonal(L).clone()
    selected = [int(torch.argmax(di2))]
    for it in range(1, n):
        j = selected[-1]
        if it == 1:
            ei = L[j] / torch.sqrt(di2[j].clamp_min(1e-12))
        else:
            ei = (L[j] - cis[: it - 1].t() @ cis[: it - 1, j]) / torch.sqrt(di2[j].clamp_min(1e-12))
        cis[it - 1] = ei
        di2 = di2 - ei * ei
        di2[selected] = -float("inf")
        selected.append(int(torch.argmax(di2)))
    return torch.tensor(selected[:n], dtype=torch.long)


def _select_relmatch(n, cand_probs, target_row, gen):
    """v4 (Formalization 1: class-relation matrix): greedy mean-matching in teacher *soft-label*
    space. Where momentmatch matches the pool's mean+covariance in penultimate *feature* space,
    relmatch matches each class's mean teacher softmax vector to its full-pool row

        R[i, :] = average soft-label mass class-i samples place on every class j,

    the relational geometry the feature-space selectors are blind to. The selected subset induces
    R̂[i, :] = mean_{s in S} p_s; over the full per-class pool (|S| = n) we minimize the full-row
    squared error

        J(S) = || mean_{s in S} p_s  -  r_i ||^2 ,

    with p_s the teacher softmax (K-dim simplex) and r_i = target_row the full-pool mean (the literal
    full row R[i, :], self-class confidence included). No L2 normalization (probabilities are already
    on the simplex).

    Greedy forward selection: maintain s1 = sum_{s in S} p_s; each step adds the candidate
    minimizing ||(s1 + p_j)/k - r_i||^2, an O(P*K) scan, so the whole select is O(n*P*K).
    Deterministic; `gen` is unused (kept for a uniform variant signature).

    cand_probs [P, K]: the full per-class candidates' soft-labels.
    target_row [K]: full per-class pool mean soft-label (= empirical R[i, :]).
    Returns n distinct LOCAL indices into cand_probs.
    """
    n = min(n, cand_probs.shape[0])
    Pc = cand_probs.double()
    P, K = Pc.shape
    r = target_row.double()

    s1 = torch.zeros(K, dtype=torch.float64)               # running sum of selected probs
    avail = torch.ones(P, dtype=torch.bool)
    selected = []
    for step in range(1, n + 1):
        k = float(step)
        mean = (s1.unsqueeze(0) + Pc) / k                  # [P, K] candidate-augmented mean
        diff = mean - r.unsqueeze(0)                       # [P, K]
        J = (diff * diff).sum(dim=1)                       # [P] full-row squared error
        J[~avail] = float("inf")
        j = int(torch.argmin(J))
        selected.append(j)
        avail[j] = False
        s1 += Pc[j]
    return torch.tensor(selected[:n], dtype=torch.long)


def _select_reldist(n, cand_probs, target_probs, gen):
    """v5 (Formalization 2: per-class relational *distribution*): match the whole per-class soft-label
    distribution, not just its mean. Where relmatch matches each class's mean teacher softmax to its
    full-pool row R[i, :] (a K-vector of means), reldist matches, on every coordinate d, the full 1-D
    *distribution* of mass-on-d the class's crops place there, to the pool's:

        minimize  J(S) = sum_d  W2^2( {p_s[d]}_{s in S},  {p_t[d]}_{t in pool} )

    the per-coordinate 1-D Wasserstein-2 (sliced OT) between the selected set and the full per-class
    pool (the full row, self-class axis included). Matching the distribution rather than the mean fixes
    relmatch's identifiability gap (many subsets share a mean) and keeps each class's intra-class
    sub-modes and its boundary tail. The mean match is the degenerate limit: collapse each coordinate's
    distribution to its mean and W2^2(delta_muS, delta_mut) = (muS - mut)^2 recovers relmatch's
    per-coordinate objective, so reldist contains Formalization 1.

    Algorithm: the W2-optimal n-point quantizer of a 1-D distribution sits at its quantile levels
    tau_k = (k - 0.5)/n, so we build the target quantile profile t_k[d] = quantile_d(tau_k) over the pool
    (active axes only: the columns with non-trivial spread) and greedily assign each rank k to the
    nearest unused candidate. This spreads by construction; a from-scratch greedy-W would instead pick
    the distribution's center first and collapse. O(n * P * A) on the A active axes. Deterministic;
    `gen` is unused (kept for a uniform variant signature).

    cand_probs [P, K]: the full per-class candidates' soft-labels.
    target_probs [M, K]: the full per-class pool whose per-axis quantiles are the target (M >= P).
    Returns n distinct LOCAL indices into cand_probs.
    """
    n = min(n, cand_probs.shape[0])
    Pc = cand_probs.double()                               # [P, K] candidates
    Pt = target_probs.double()                             # [M, K] pool (defines per-axis quantiles)
    P, K = Pc.shape

    # active coordinates: the columns with non-trivial pool spread, so the near-constant columns of a
    # large-K problem cost nothing and never drive the assignment.
    active = Pt.var(dim=0, unbiased=False) > 1e-8
    if not bool(active.any()):
        active = torch.ones(K, dtype=torch.bool)           # degenerate fallback
    cols = active.nonzero(as_tuple=True)[0]
    Pc_a = Pc[:, cols]                                     # [P, A]
    Pt_a = Pt[:, cols]                                     # [M, A]

    taus = (torch.arange(1, n + 1, dtype=torch.float64) - 0.5) / n      # [n] quantile levels
    T = torch.quantile(Pt_a, taus, dim=0)                  # [n, A] target quantile profiles

    avail = torch.ones(P, dtype=torch.bool)
    selected = []
    for kk in range(n):
        diff = Pc_a - T[kk].unsqueeze(0)                   # [P, A]
        d2 = (diff * diff).sum(dim=1)                      # [P] squared distance to t_k
        d2[~avail] = float("inf")
        j = int(torch.argmin(d2))
        selected.append(j)
        avail[j] = False
    return torch.tensor(selected[:n], dtype=torch.long)


def _select_facloc(n, cand_emb, target_emb, gen):
    """v6 (Formalization 3: submodular / set-level complementarity): greedy facility-location
    coverage. Where covmatch maximizes the selected set's *volume* (log-det / DPP) and relmatch /
    reldist match the pool's mean / per-coordinate distribution, facility-location maximizes how well
    the selected set *covers* the full per-class pool:

        F(S) = sum_{i in pool} max_{s in S} cos(e_i, e_s)

    on L2-normalized embeddings (same cosine convention as _select_covmatch). F is monotone
    submodular, so the standard greedy (add the candidate with the largest marginal coverage gain at
    each step) carries the (1 - 1/e) guarantee. Complementarity is set-level *by construction*: a
    candidate's value is its marginal gain

        Delta(j | S) = sum_{i in pool} max(0,  cos(e_i, e_j) - max_{s in S} cos(e_i, e_s)),

    i.e. it counts only what j adds over what S already covers, not any standalone score. This is the
    formalization that literally "earns the word complementary" (thought.md, Formalization 3).

    cand_emb [P, D]: the full per-class candidates' embedding (teacher soft-labels or features).
    target_emb [M, D]: the full per-class pool to cover (M >= P; here the same pool).
    Returns n distinct LOCAL indices into cand_emb. Deterministic; `gen` is unused (kept for a
    uniform variant signature).
    """
    n = min(n, cand_emb.shape[0])
    Zc = F.normalize(cand_emb.double(), dim=1)            # [P, D] candidates
    Zt = F.normalize(target_emb.double(), dim=1)          # [M, D] pool to cover
    Smat = Zt @ Zc.t()                                    # [M, P] cos(pool_i, cand_j)

    covered = torch.full((Zt.shape[0],), -1.0, dtype=torch.float64)  # per-pool best cov (min cosine)
    avail = torch.ones(Zc.shape[0], dtype=torch.bool)
    selected = []
    for _ in range(n):
        gain = torch.clamp(Smat - covered.unsqueeze(1), min=0).sum(dim=0)     # [P] marginal coverage
        gain[~avail] = float("-inf")
        j = int(torch.argmax(gain))
        selected.append(j)
        avail[j] = False
        covered = torch.maximum(covered, Smat[:, j])      # absorb j into the coverage frontier
    return torch.tensor(selected[:n], dtype=torch.long)


def _select_variant(method, n, dist, feats, k, gen, mean_weight=1.0,
                    beta=0.0, qual_raw=None, probs=None, facloc_space="softlabel"):
    """Pick n indices (into the per-class pool) by a variance-aware rule, on CPU tensors.

    `dist` [P] is the per-candidate teacher CE loss (realism); `feats` [P, D] the teacher
    penultimate features (None for relmatch/reldist, which read soft-labels instead). All variants
    select over the full per-class pool (the same candidates stock ranks); covmatch and
    momentmatch read the pool's feature geometry as their target. `qual_raw` [P] is the qddpp
    quality score ("lower is better"; defaults to `dist`, i.e. teacher confidence) and `beta` its
    quality<->diversity knob. `probs` [P, K] are the teacher soft-labels read by relmatch (whose
    full-pool mean is the target row R[i, :]) and reldist (whose full-pool per-axis *distribution* is
    the target). facloc covers the full pool by greedy facility-location over either the soft-labels
    (`facloc_space="softlabel"`) or the features (`facloc_space="feature"`).
    """
    eligible = torch.argsort(dist, descending=False)
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
    if method == "momentmatch":
        local = _select_momentmatch(n, feats[eligible], feats, mean_weight, gen)
        return eligible[local]
    if method == "qddpp":
        qr = dist if qual_raw is None else qual_raw
        local = _select_qddpp(n, feats[eligible], qr[eligible], beta, gen)
        return eligible[local]
    if method == "relmatch":
        local = _select_relmatch(n, probs[eligible], probs.mean(0), gen)
        return eligible[local]
    if method == "reldist":
        local = _select_reldist(n, probs[eligible], probs, gen)
        return eligible[local]
    if method == "facloc":
        if facloc_space == "feature":
            local = _select_facloc(n, feats[eligible], feats, gen)
        else:
            local = _select_facloc(n, probs[eligible], probs, gen)
        return eligible[local]
    raise ValueError(f"unknown --select-method: {method}")


def selector(n, model, images, labels, size, m=5, method="stock",
             k=8, rng_seed=0, mean_weight=1.0,
             beta=0.0, quality="confidence", facloc_space="softlabel"):
    """Select n crops per class from the m-crop candidates.

    method="stock" reproduces RDED exactly: keep the most teacher-confident crop per image,
    then take the n globally-most-confident. The variance-aware methods (random/stratified/
    covmatch/momentmatch/qddpp) instead spread the selection across the class's within-class
    feature spread, over the full per-class pool, at the same n (= fixed IPC); momentmatch
    matches the full pool's mean+covariance (mean_weight scales the mean term); qddpp trades
    feature volume against a quality score via `beta` (`quality`: "confidence" = teacher CE,
    "margin" = top1-top2 logit gap, a boundary-seeking lever). relmatch matches the class's mean
    teacher *soft-label* to its full-pool row R[i, :] (the class-relation matrix) in probability
    space; reldist matches the full per-coordinate soft-label *distribution* to the pool's. facloc
    greedily covers the full pool by facility-location over the soft-labels (facloc_space="softlabel",
    the default) or the features (facloc_space="feature") — set-level complementarity.
    """
    facloc_softlabel = method == "facloc" and facloc_space == "softlabel"
    need_feats = (method not in ("stock", "relmatch", "reldist")) and not facloc_softlabel
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

        # The best crop's logits feed qddpp's "margin" quality (top1-top2 gap) and the soft-label
        # selectors' teacher softmax (relmatch/reldist, and facloc in soft-label space); everything
        # else uses teacher confidence (best_dist) as the quality score.
        need_probs = method in ("relmatch", "reldist") or facloc_softlabel
        best_logits = None
        if (method == "qddpp" and quality == "margin") or need_probs:
            best_logits = preds.reshape(m, s[0], preds.shape[1])[index, torch.arange(s[0])]

        qual_raw = best_dist
        if method == "qddpp" and quality == "margin":
            top2 = best_logits.topk(2, dim=1).values
            qual_raw = top2[:, 0] - top2[:, 1]

        probs = F.softmax(best_logits, dim=1) if need_probs else None

    if method == "stock":
        sel = torch.argsort(best_dist, descending=False)[:n]
    else:
        gen = torch.Generator().manual_seed(int(rng_seed))
        sel = _select_variant(
            method, n, best_dist.cpu(),
            feats.cpu() if feats is not None else None,
            k, gen, mean_weight, beta=beta, qual_raw=qual_raw.cpu(),
            probs=probs.cpu() if probs is not None else None,
            facloc_space=facloc_space,
        )

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
