# The selectors, with code — how each selection rule actually works

This is the **code-level companion** to [`EXPERIMENTS.md`](EXPERIMENTS.md). That document
explains the metrics and the *math* of every selector (no code); this one focuses **solely on the
selectors** and walks through the **actual implementation** so you can see how each rule turns a
class's candidate crops into a chosen subset.

Everything below lives in [`synthesize/utils.py`](../synthesize/utils.py); the CLI flags are in
[`argument.py`](../argument.py) and the call site is in [`synthesize/main.py`](../synthesize/main.py).
Code snippets are quoted verbatim from those files (trimmed to the relevant lines).

> The selection step is the **only** part of the pipeline the intervention changes. Teacher
> relabeling and KD training are byte-identical to stock RDED. So everything here is "given a
> class's crops, which `n` do we keep?"

---

## 0. The one-paragraph mental model

For each class we have `mipc` source images, each randomly cropped `m` times → `mipc * m` candidate
crops. The teacher scores every crop. We must keep exactly `n = ipc * factor²` crops (the **fixed-IPC
budget** — same for every method). **Stock** keeps the `n` most teacher-confident crops, which are
all near the class mean → over-collapsed. The **variance-aware** methods instead choose `n` from the
**full per-class pool** (the same candidates stock ranks) to **spread out** in feature space —
recovering the within-class variance stock discards, at no change to the budget. (Two variants,
**`relmatch`** (§8) and **`reldist`** (§9), are the exception: they work in
*soft-label* space, preserving *between*-class structure — the class-relation matrix, and for `reldist`
its full per-class distribution — instead of within-class spread.)

---

## 1. The entry point: `selector(...)`

Every method comes through one function. It does the teacher forward pass, reduces each source image
to its single best crop, then dispatches to either the stock path or a variance-aware variant.

```python
def selector(n, model, images, labels, size, m=5, method="stock",
             k=8, rng_seed=0, mean_weight=1.0,
             beta=0.0, quality="confidence", facloc_space="softlabel"):
    facloc_softlabel = method == "facloc" and facloc_space == "softlabel"
    need_feats = (method not in ("stock", "relmatch", "reldist")) and not facloc_softlabel  # logits/soft-labels only
    with torch.no_grad():
        images = images.cuda()
        s = images.shape                       # [mipc, m, C, H, W]

        # flatten crops: [mipc*m, C, H, W], crop-major
        images = images.permute(1, 0, 2, 3, 4).reshape(s[0]*s[1], s[2], s[3], s[4])
        labels_rep = labels.repeat(m).cuda()

        # teacher forward → logits (+ penultimate feats only when a variant needs them)
        preds, feats = _forward_logits_feats(model, pad(images, size).cuda(), s[0], need_feats)

        # per-crop teacher CE loss ("realism"): low loss ⇔ high confidence
        dist = cross_entropy(preds, labels_rep).reshape(m, s[0])     # [m, mipc]

        # keep each source image's *single most-confident crop* → [mipc]
        index = torch.argmin(dist, 0)
        best_dist = dist[index, torch.arange(s[0])]
        images = images.reshape(m, s[0], ...)[index, torch.arange(s[0])]
        if need_feats:
            feats = feats.reshape(m, s[0], feats.shape[1])[index, torch.arange(s[0])]
```

Two things to notice:

- **`need_feats = method not in ("stock", "relmatch", "reldist")`** — the stock path never extracts penultimate
  features, so it reproduces RDED *exactly*. Features are only captured (via a forward hook on the last
  linear layer, see `_forward_logits_feats`) when a variance-aware method asks for them. `relmatch` is
  the lone exception among the variants: it works from the teacher's **soft labels** (already in the
  logits), so it too skips the feature hook.
- **The inner `argmin`** — before *any* selection, each source image is collapsed to its single
  most-confident crop (`index = torch.argmin(dist, 0)`). So selection always operates on one
  representative crop per source image; `best_dist` is that crop's teacher CE loss.

Then the dispatch:

```python
    if method == "stock":
        sel = torch.argsort(best_dist, descending=False)[:n]        # n lowest-loss crops
    else:
        gen = torch.Generator().manual_seed(int(rng_seed))          # deterministic
        sel = _select_variant(method, n, best_dist.cpu(), feats.cpu(), k, gen,
                              mean_weight, beta=beta, qual_raw=qual_raw.cpu(),
                              probs=probs.cpu(), facloc_space=facloc_space)
    return images[sel.to(images.device)].detach()
```

**Stock is literally one line of sorting.** Everything else routes through `_select_variant`, which
runs on **CPU tensors** (the matrices are tiny: pool size × feature dim).

---

## 2. The shared entry: the full per-class pool

Every variance-aware method selects over the **full per-class pool** — the same candidates `stock`
ranks — so all methods see an identical budget and candidate set; only the *rule* for picking `n`
differs. The dispatcher just orders the pool by teacher CE loss (low ⇔ confident) for stable,
deterministic local indexing, and short-circuits when the pool is already at the budget:

```python
def _select_variant(method, n, dist, feats, k, gen, mean_weight=1.0,
                    beta=0.0, qual_raw=None, probs=None, facloc_space="softlabel"):
    eligible = torch.argsort(dist, descending=False)   # full pool, ranked by CE
    if eligible.numel() <= n:
        return eligible[:n]                    # pool == budget → no freedom to spread
    ...
```

Note the indirection: variants return **local** indices into `eligible`, and the dispatcher maps them
back with `return eligible[local]`. Keep that in mind reading each method.

---

## 3. `random` — the dose-response anchor

Inject variance without being clever. Uniformly sample `n` from the floored pool:

```python
    if method == "random":
        perm = torch.randperm(eligible.numel(), generator=gen)[:n]
        return eligible[perm]
```

This is the **weak lever**. It isolates "more within-class spread" from "*smartly chosen* spread" —
if `covmatch` beats `random` beats `stock`, the gain is partly from being clever, not just from
variance. The `gen` makes it deterministic given the seed.

---

## 4. `stratified` — span the visual modes (`_select_stratified`)

Cluster the pool's features with a tiny hand-rolled k-means, then take crops **round-robin across
clusters**, most-confident-first within each. This guarantees the selection touches every mode of the
class instead of piling onto the dominant one.

```python
def _kmeans(x, k, gen, iters=10):
    P = x.shape[0]
    centers = x[torch.randperm(P, generator=gen)[:k]].clone()      # random-point init
    assign = torch.zeros(P, dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(dim=1)             # assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                centers[c] = x[mask].mean(dim=0)                  # update
    return assign

def _select_stratified(n, dist, feats, k, gen):
    k = max(1, min(k, feats.shape[0]))
    assign = _kmeans(feats.float(), k, gen)
    buckets = []
    for c in range(k):
        idx = (assign == c).nonzero(as_tuple=True)[0]
        if idx.numel():
            buckets.append(idx[torch.argsort(dist[idx])].tolist()) # ascending loss within cluster
    sel, depth = [], 0
    while len(sel) < n:                                            # round-robin across clusters
        progressed = False
        for b in buckets:
            if depth < len(b):
                sel.append(b[depth]); progressed = True
                if len(sel) >= n: break
        if not progressed: break
        depth += 1
    return torch.tensor(sel[:n], dtype=torch.long)
```

The **round-robin** is the key: `depth=0` takes the best crop of every cluster, `depth=1` the second-
best of every cluster, etc. So coverage of modes comes first, and confidence breaks ties within a
mode. `k` is `--select-k` (default 8).

---

## 5. `covmatch` — maximize feature volume (`_select_covmatch`)

Pick the subset whose features are most **spread out**, by greedily maximizing the log-determinant of
the cosine kernel (a DPP-MAP / determinantal point process objective). `log det K_S` is the log-volume
of the parallelotope the selected feature vectors span — maximizing it forces them to be mutually
dissimilar, i.e. **increases within-class scatter** → pushes `NC1_distilled` up toward real.

```python
def _select_covmatch(n, feats, gen):
    n = min(n, feats.shape[0])
    Z = F.normalize(feats.double(), dim=1)                   # L2-normalize → cosine space
    P = Z.shape[0]
    K = Z @ Z.t() + 1e-4 * torch.eye(P, dtype=torch.float64) # cosine kernel + ridge

    cis = torch.zeros(n, P, dtype=torch.float64)
    di2 = torch.diagonal(K).clone()                          # conditional variances
    selected = [int(torch.argmax(di2))]                      # start: highest variance
    for it in range(1, n):
        j = selected[-1]
        if it == 1:
            ei = K[j] / torch.sqrt(di2[j].clamp_min(1e-12))
        else:
            ei = (K[j] - cis[:it-1].t() @ cis[:it-1, j]) / torch.sqrt(di2[j].clamp_min(1e-12))
        cis[it-1] = ei
        di2 = di2 - ei * ei                                  # downdate remaining variances
        di2[selected] = -float("inf")                       # never re-pick
        selected.append(int(torch.argmax(di2)))             # add max conditional variance
    return torch.tensor(selected[:n], dtype=torch.long)
```

This is the **fast greedy of Chen et al. (2018)**. The intuition for the loop: `di2[i]` holds the
*conditional variance* of candidate `i` given everything already selected — how much new volume it
would add. Each step adds the `argmax`, then `di2 = di2 - ei*ei` downdates the rest (Cholesky-style
rank-1 update). It's **optimization-free and deterministic** (`gen` is unused).

> **Why L2-normalize?** Working in cosine space removes the feature-*norm* confound — the same reason
> the Mahalanobis OOD score normalizes (see `EXPERIMENTS.md §4.3`). `covmatch` then restores
> *directional* spread, complementary to the magnitude that normalization discards.

**The one nuance ("match, don't maximize"):** because `covmatch` *maximizes* volume, it can
**overshoot** real variance (`NC1_distilled / NC1_real > 1`). Empirically the best trust sits near a
ratio of 1, which is what `momentmatch` and `qddpp` target instead.

---

## 6. `momentmatch` — match the real covariance, don't overshoot (`_select_momentmatch`)

Where `covmatch` maximizes volume regardless of the target's shape, `momentmatch` greedily picks the
subset whose feature **mean + second moment** best match the *full pool's* moments. Matching both the
mean `μ` and uncentered second moment `M` is equivalent to matching the centered covariance
`Σ = M - μμᵀ` — the exact statistic the tied-covariance Mahalanobis score reads.

```python
def _select_momentmatch(n, cand_feats, target_feats, mean_weight, gen):
    n = min(n, cand_feats.shape[0])
    Zc = F.normalize(cand_feats.double(), dim=1)            # [P, D] candidates (floored pool)
    Zt = F.normalize(target_feats.double(), dim=1)         # [M, D] target = FULL per-class pool
    P, D = Zc.shape

    mu_t = Zt.mean(dim=0)                                   # target mean
    M_t  = (Zt.t() @ Zt) / Zt.shape[0]                     # target uncentered 2nd moment
    quad_Mt = ((Zc @ M_t) * Zc).sum(dim=1)                 # z_jᵀ M_t z_j  (constant per cand)

    s1  = torch.zeros(D, dtype=torch.float64)              # running sum of selected z
    ZcG = torch.zeros(P, D, dtype=torch.float64)          # Zc @ G,  G = Σ_{i∈S} z_i z_iᵀ
    avail = torch.ones(P, dtype=torch.bool)
    selected = []
    for step in range(1, n + 1):
        k = float(step)
        mu = (s1.unsqueeze(0) + Zc) / k                    # mean if we added each candidate
        mean_term = ((mu - mu_t.unsqueeze(0)) ** 2).sum(dim=1)
        quad_G = (ZcG * Zc).sum(dim=1)                     # z_jᵀ G z_j
        sm_term = (2.0/k**2)*quad_G + 1.0/k**2 - (2.0/k)*quad_Mt
        J = sm_term + mean_weight * mean_term              # objective (shared consts dropped)
        J[~avail] = float("inf")
        j = int(torch.argmin(J))                           # pick the best-matching addition
        selected.append(j); avail[j] = False
        zj = Zc[j]
        ZcG += (Zc @ zj).unsqueeze(1) * zj.unsqueeze(0)    # rank-1 update: G += z_j z_jᵀ
        s1  += zj
    return torch.tensor(selected[:n], dtype=torch.long)
```

The objective is `J(S) = ‖M_S − M_t‖_F² + mean_weight · ‖μ_S − μ_t‖²`. The trick that keeps each step
cheap is maintaining `ZcG = Zc @ G` and `s1` with **rank-1 updates**, so adding a crop is `O(P·D)`
rather than recomputing from scratch. The target is the **full** per-class pool (`target_feats`), not
the floored candidates — so the selected set reproduces the class's *anisotropic* spread instead of
merely spanning it. `mean_weight` (`--momentmatch-mean-weight`, default 1.0) trades the mean term
against the covariance term. Note this is the one variant where the dispatcher passes the **full pool**
as the target:

```python
    if method == "momentmatch":
        local = _select_momentmatch(n, feats[eligible], feats, mean_weight, gen)
        return eligible[local]
```

---

## 7. `qddpp` — one knob from `covmatch` to `stock` (`_select_qddpp`)

`stock` (pure quality) and `covmatch` (pure diversity) are the two endpoints of a **single** objective:
a quality-weighted DPP (an *L-ensemble*, Kulesza & Taskar 2012). Put a quality weight `q_i` on each
crop and maximize `log det L_S` where `L = diag(q) (ZZᵀ) diag(q)`. Because that determinant separates
into `2·Σ log q_j + log det K_S`, it is literally **a quality term plus `covmatch`'s volume term**, and
one temperature `β` dials between them.

```python
def _select_qddpp(n, feats, qual_raw, beta, gen):
    n = min(n, feats.shape[0])
    Z = F.normalize(feats.double(), dim=1)
    P = Z.shape[0]
    r = qual_raw.double()
    r = -(r - r.mean()) / r.std().clamp_min(1e-12)         # high r ⇔ desirable (low qual_raw)
    q = torch.exp(float(beta) * r)                         # quality weights
    L = (q.unsqueeze(1) * q.unsqueeze(0)) * (Z @ Z.t()) + 1e-4 * torch.eye(P, dtype=torch.float64)
    # ... identical Chen-et-al. greedy log-det as _select_covmatch, but on L instead of K ...
```

The greedy loop is *byte-for-byte* `covmatch`'s — only the kernel changes (`L` instead of `K`). The
knob:

- **`beta → 0`**: `q → 1`, `L → ZZᵀ` ⇒ **exactly `covmatch`** (pure feature volume).
- **`beta → ∞`**: the quality term dominates ⇒ collapses onto the top-quality crops ⇒ **exactly
  `stock`** (when quality is the teacher CE loss).

So intermediate `β` **targets** real within-class variance rather than overshooting it — confidence
enters as a *soft* quality weight instead of a hard cutoff. Set with `--select-beta`.

### The near-OOD lever: `--select-quality margin`

By default the quality score is teacher confidence (`best_dist`). Switching `quality="margin"` makes it
the teacher's **top1−top2 logit gap**, up-weighting *low-margin* (decision-boundary) crops. This is
computed back in `selector(...)`:

```python
    qual_raw = best_dist                                   # default: confidence (far-OOD lever)
    if method == "qddpp" and quality == "margin":
        best_logits = preds.reshape(m, s[0], preds.shape[1])[index, torch.arange(s[0])]
        top2 = best_logits.topk(2, dim=1).values
        qual_raw = top2[:, 0] - top2[:, 1]                 # margin → boundary-seeking (near-OOD lever)
```

Confidence-quality + volume restores within-class *spread* (a **far**-OOD lever); margin-quality seeds
the selection with crops near the class boundary — the geometry **near**-OOD detection reads, the one
shift type variance restoration misses.

---

## 8. `relmatch` — match the class-relation matrix in soft-label space (`_select_relmatch`)

Every method above works in **feature** space and targets *within-class* geometry (how spread out a
class is). `relmatch` is the odd one out on both counts: it works in **soft-label** space and targets
*between-class* geometry — **how each class relates to all the others**.

The teacher's softmax over all `K` classes is a crop's *soft label*; its **off-diagonal** mass is dark
knowledge ("this cat also looks 22% like a dog"). Average the soft labels over a whole class and you
get one row of the **class-relation matrix**, `R[i,:]` — the dataset's inter-class fingerprint. `stock`
keeps the most-confident crops and flattens that fingerprint; `relmatch` keeps the subset whose **mean
soft label** reproduces it, i.e. it drives the distilled set's `R̂[i,:]` toward `R[i,:]`.

Mechanically it is `momentmatch`'s *mean* term moved from feature space into probability space (and
with no L2-normalization — soft labels already live on the simplex):

```python
def _select_relmatch(n, cand_probs, target_row, gen):
    n  = min(n, cand_probs.shape[0])
    Pc = cand_probs.double()                               # [P, K] candidates' soft labels (full pool)
    P, K = Pc.shape
    r = target_row.double()                                # [K] target row  r = R[i,:]

    s1 = torch.zeros(K, dtype=torch.float64)               # running SUM of selected soft labels
    avail = torch.ones(P, dtype=torch.bool)
    selected = []
    for step in range(1, n + 1):
        k    = float(step)
        mean = (s1.unsqueeze(0) + Pc) / k                  # [P, K] mean if we added each candidate
        diff = mean - r.unsqueeze(0)
        J    = (diff * diff).sum(dim=1)                    # full-row squared error  ‖r̂ − r‖²
        J[~avail] = float("inf")
        j = int(torch.argmin(J))                           # add the crop that pulls the mean toward r
        selected.append(j); avail[j] = False
        s1 += Pc[j]                                        # rank-1 update: just add its soft label
    return torch.tensor(selected[:n], dtype=torch.long)
```

The objective is simply `J(S) = ‖r̂(S) − r‖²` with `r̂(S) = (1/|S|) Σ_{s∈S} p_s` the selected crops'
mean soft label — the same greedy "add the candidate that pulls the running mean toward the target" as
`momentmatch`, but mean-only, and matching the **full row** `R[i,:]` (self-confidence and inter-class
lean together). The dispatcher feeds it the soft labels and the **full pool's** mean as the target (like
`momentmatch`, the target is the full pool):

```python
    if method == "relmatch":                               # soft-label space, not feats
        return eligible[_select_relmatch(n, probs[eligible], probs.mean(0), gen)]
```

**Where `probs` comes from.** `relmatch` is the one variant that needs **soft labels, not penultimate
features** — so `need_feats` is false for it (§1) and `selector(...)` just softmaxes the already-gathered
best-crop logits:

```python
    probs = F.softmax(best_logits, dim=1) if method in ("relmatch", "reldist") else None
```

**Why it matters.** The row `R[i,:]` *is* the class's relational geometry — its self-confidence and which
classes it sits near — and that is exactly what calibration, OOD detection, and robustness read, and
exactly what confidence-greedy `stock` flattens. A from-scratch, beginner-level walkthrough with a worked
3-class example lives in
[`RELMATCH_EXPLAINED.md`](RELMATCH_EXPLAINED.md); the matrix is measured by `class_relation_matrix` /
`relation_divergence` in [`validation/diagnostics.py`](../validation/diagnostics.py) and reported by
`tools/diagnose_geometry.py` (`rel_frob`, `rel_frob_offdiag`, `rel_row_cos_offdiag`, `rel_eig_overlap`).

---

## 9. `reldist` — match the class's soft-label *distribution*, not just its mean (`_select_reldist`)

`relmatch` (§8) matches each class's **mean** soft-label — one row `R[i,:]`. But a class is not its
average: many subsets share a mean while describing very different classes. `reldist` matches the **whole
per-class distribution** of soft labels — its modes, sub-modes and boundary tail — by making the selected
set's per-coordinate distribution close to the full pool's in **Wasserstein** (optimal-transport)
distance. Collapse every distribution to its mean and the objective becomes `relmatch`'s exactly, so
`reldist` **contains** Formalization 1 and adds the higher quantiles the mean discards.

It avoids the center-collapse a from-scratch greedy would hit by using the textbook 1-D quantizer: target
the pool's evenly-spaced quantiles `τ_k = (k−½)/n` per active coordinate and assign each rank to its
nearest still-unused crop. Deterministic, `O(n·P·A)` over the `A` active axes.

```python
def _select_reldist(n, cand_probs, target_probs, gen):
    active = target_probs.var(0) > 1e-8                    # only the columns that actually move
    taus = (torch.arange(1, n + 1) - 0.5) / n              # quantile levels (k-1/2)/n
    T = torch.quantile(target_probs[:, cols], taus, dim=0) # [n, A] target quantile profiles t_k
    for kk in range(n):                                    # assign each rank to its nearest unused crop
        d2 = ((cand - T[kk]) ** 2).sum(1); d2[~avail] = inf
        pick = int(d2.argmin())
```

The dispatcher feeds it the soft labels and the **full pool** as the distribution target (the analogue of
`momentmatch` passing the full-pool features):

```python
    if method == "reldist":                                # soft-label distribution, not feats
        return eligible[_select_reldist(n, probs[eligible], probs, gen)]
```

Like `relmatch`, `reldist` reads **soft labels, not features** (so it joins
`need_feats = method not in ("stock", "relmatch", "reldist")`), and matches the **full row** — every
moving coordinate, self-confidence and inter-class lean alike — no knob.

**Why it matters.** `relmatch` reproduces each row's *mean* but can **overshoot its shape** (it recruits
too many extreme boundary crops to hit the average); `reldist` keeps the shape right — and gets the best
mean as a by-product (lowest `rel_frob` at matched IPC). Measured by `relation_distribution_divergence`
in [`validation/diagnostics.py`](../validation/diagnostics.py) (`reldist_w_full`, `reldist_w_off`,
`reldist_sw`, `reldist_tail_cov`); a from-scratch walkthrough with a worked toy lives in
[`RELDIST_EXPLAINED.md`](RELDIST_EXPLAINED.md).

---

## 10. `facloc` — greedy facility-location coverage (`_select_facloc`)

`covmatch` (§5) maximizes feature **volume** (log-det / DPP); `relmatch`/`reldist` (§8–9) match a
soft-label **statistic**. `facloc` is the **coverage** selector (Formalization 3: set-level
complementarity): pick the `n` crops that best *cover* the full per-class pool, where the value of
each pick is its **marginal** coverage gain given what's already chosen. It maximizes the
facility-location objective `F(S) = Σ_i max_{s∈S} cos(e_i, e_s)` — monotone submodular, so the greedy
below is `(1−1/e)`-optimal. The math (submodularity, the greedy guarantee, the
coverage-vs-volume contrast with `covmatch`, and a worked toy) lives in
[`FACLOC_EXPLAINED.md`](FACLOC_EXPLAINED.md).

```python
def _select_facloc(n, cand_emb, target_emb, gen):
    n = min(n, cand_emb.shape[0])
    Zc = F.normalize(cand_emb.double(), dim=1)            # [P, D] candidates
    Zt = F.normalize(target_emb.double(), dim=1)          # [M, D] pool to cover
    Smat = Zt @ Zc.t()                                    # [M, P] cos(pool_i, cand_j)

    covered = torch.full((Zt.shape[0],), -1.0, dtype=torch.float64)   # coverage frontier (min cosine)
    avail = torch.ones(Zc.shape[0], dtype=torch.bool)
    selected = []
    for _ in range(n):
        gain = torch.clamp(Smat - covered.unsqueeze(1), min=0).sum(dim=0)       # [P] marginal gain
        gain[~avail] = float("-inf")
        j = int(torch.argmax(gain))
        selected.append(j); avail[j] = False
        covered = torch.maximum(covered, Smat[:, j])      # absorb j into the frontier
    return torch.tensor(selected[:n], dtype=torch.long)
```

The only state is `covered[i] = max_{s∈S} cos(pool_i, s)` (the best similarity each pool point has to
anything chosen so far), seeded at `−1` (the minimum cosine, so an empty set covers nothing — this keeps
the first pick's gain monotone in the total similarity `Σ_i Smat[i,j]`, i.e. the coverage medoid). Each
round scores every candidate by `Σ_i max(0, Smat[i,j] − covered[i])` (how much it would *improve* the
worst-covered points), takes the argmax, and updates the frontier — a rank-1 `max`. `O(n·M·P)`,
deterministic (`gen` unused).

**The `--facloc-space` knob.** `facloc` is the one selector that runs in *either* space:

- **`softlabel`** (default): the embedding is the teacher **soft label** (the relational manifold, the
  same `probs` `relmatch`/`reldist` use) — so `need_feats` is false and `selector(...)` softmaxes the
  best-crop logits, exactly like `relmatch`/`reldist`. Coverage here targets the inter-class geometry
  of `R`.
- **`feature`**: the embedding is the penultimate **feature** (`need_feats` true) — the
  *coverage* sibling of `covmatch`'s *volume* objective, in the same feature space.

The dispatcher feeds it the chosen embedding and the **full pool** as the cover target (like
`momentmatch`/`relmatch`/`reldist`):

```python
    if method == "facloc":
        if facloc_space == "feature":
            return eligible[_select_facloc(n, feats[eligible], feats, gen)]
        return eligible[_select_facloc(n, probs[eligible], probs, gen)]
```

**Why it matters.** Coverage cannot leave a class sub-mode unrepresented, so it keeps the dog-ish-cat
boundary crops `stock` discards — the off-diagonal `R` mass calibration/OOD read. Measured by the same
`class_relation_matrix` / `relation_divergence` diagnostics as `relmatch` (expect lower
`rel_frob_offdiag` vs `stock` at matched IPC).

---

## 11. The dispatch table, in one place

```python
def _select_variant(method, n, dist, feats, k, gen,
                    mean_weight=1.0, beta=0.0, qual_raw=None,
                    probs=None, facloc_space="softlabel"):
    eligible = torch.argsort(dist, descending=False)       # full pool, ranked by CE
    if eligible.numel() <= n:
        return eligible[:n]

    if method == "random":
        perm = torch.randperm(eligible.numel(), generator=gen)[:n]
        return eligible[perm]
    if method == "stratified":
        return eligible[_select_stratified(n, dist[eligible], feats[eligible], k, gen)]
    if method == "covmatch":
        return eligible[_select_covmatch(n, feats[eligible], gen)]
    if method == "momentmatch":
        return eligible[_select_momentmatch(n, feats[eligible], feats, mean_weight, gen)]
    if method == "qddpp":
        qr = dist if qual_raw is None else qual_raw
        return eligible[_select_qddpp(n, feats[eligible], qr[eligible], beta, gen)]
    if method == "relmatch":                               # soft labels, not feats
        return eligible[_select_relmatch(n, probs[eligible], probs.mean(0), gen)]
    if method == "reldist":                                # soft-label distribution, not feats
        return eligible[_select_reldist(n, probs[eligible], probs, gen)]
    if method == "facloc":                                 # coverage over soft-labels or feats
        if facloc_space == "feature":
            return eligible[_select_facloc(n, feats[eligible], feats, gen)]
        return eligible[_select_facloc(n, probs[eligible], probs, gen)]
    raise ValueError(f"unknown --select-method: {method}")
```

| method | needs feats? | needs full pool? | extra knobs | one-line behavior |
|---|---|---|---|---|
| `stock` | no | — | — | `n` lowest-loss crops (RDED) |
| `random` | yes | no | — | uniform sample of the floored pool |
| `stratified` | yes | no | `--select-k` | round-robin across k-means clusters |
| `covmatch` | yes | no | — | greedy max log-det (max feature volume) |
| `momentmatch` | yes | **yes** | `--momentmatch-mean-weight` | match full-pool mean+covariance |
| `qddpp` | yes | no | `--select-beta`, `--select-quality` | quality↔volume DPP; β interpolates stock↔covmatch |
| `relmatch` | no — **soft labels** | **yes** | — | match full-pool mean soft label = class-relation row `R[i,:]` (full row) |
| `reldist` | no — **soft labels** | **yes** | — | match full-pool soft-label *distribution* per coordinate (Wasserstein / quantile match; full row) |
| `facloc` | only if `--facloc-space feature` | **yes** | `--facloc-space` | greedy facility-location **coverage** of the full pool (set-level complementarity; `(1−1/e)` greedy), over soft-labels (default) or features |

---

## 12. How it's wired into synthesis

The call site, once per class batch, in [`synthesize/main.py`](../synthesize/main.py):

```python
for c, (images, labels) in enumerate(tqdm(train_loader)):
    images = selector(
        args.ipc * args.factor**2,          # n = the fixed-IPC budget
        model, images, labels, args.input_size,
        m=args.num_crop,
        method=getattr(args, "select_method", "stock"),
        k=getattr(args, "select_k", 8),
        rng_seed=args.seed * 100003 + c,    # per-class deterministic seed
        mean_weight=getattr(args, "momentmatch_mean_weight", 1.0),
        beta=getattr(args, "select_beta", 0.0),
        quality=getattr(args, "select_quality", "confidence"),
        facloc_space=getattr(args, "facloc_space", "softlabel"),
    )
    images = mix_images(images, args.input_size, args.factor, args.ipc)
    save_images(args, denormalize(images), c)
```

After selection, `mix_images` stitches the chosen crops into the distilled images — **unchanged from
stock RDED**, as is the downstream teacher relabeling and KD training.

### Path-keying (so variants never clobber each other)

The distilled-set output path is keyed by the selection method (and each method's sub-knobs) in
[`argument.py`](../argument.py) via the single source of truth `selector_suffix(args)` in
[`validation/run_key.py`](../validation/run_key.py), so every variant caches to its own directory:

```python
# argument.py — the same suffix used by experiment.sh (resume) and tools/diagnose_geometry.py
args.exp_name += selector_suffix(args)      # "" for stock; e.g. "_selqddpp_b0.5", "_selfacloc_spfeature"
```

`selector_suffix` appends `_sel<method>` plus only the non-default keyed sub-knobs (`qddpp` beta/quality,
`stratified` k, `momentmatch` mean-weight, `facloc` space). This matters because runs are deterministic
and cached — a `covmatch` set and a `qddpp_b0.5` set live at distinct paths and are never confused.

---

## 13. Try one

```bash
# stock vs covmatch on one cell / one seed
CUDA_VISIBLE_DEVICES=0 METHODS="stock covmatch" CELLS="cifar100:conv3" \
  IPCS="10" SEEDS="42" bash scripts/run_select_variants.sh

# qddpp at an intermediate beta (between covmatch β=0 and stock β→∞)
python main.py --select-method qddpp --select-beta 0.5 ...

# qddpp boundary (near-OOD) lever
python main.py --select-method qddpp --select-beta 0.5 --select-quality margin ...

# read the dose-response table
python tools/analyze_select.py

# confirm the H1 lever moved (NC1 of the distilled set vs equal real subset)
python tools/diagnose_geometry.py --subset cifar100 --arch-name conv3 \
  --ipc 10 --select-method covmatch --syn-leaf syn_data_seed42

# measure the inter-class soft-label distribution match (reldist vs stock/relmatch)
python tools/diagnose_geometry.py --subset cifar100 --arch-name conv3 \
  --ipc 10 --select-method reldist --syn-leaf syn_data_seed42
```

For the metrics these selectors move and the full math, see [`EXPERIMENTS.md`](EXPERIMENTS.md).
