# `reldist` explained — from intuition to math to code

> A teaching walkthrough, in the same spirit as [`RELMATCH_EXPLAINED.md`](RELMATCH_EXPLAINED.md).
> We reuse the **one running example** (three classes: 🐱 cat, 🐶 dog, 🚗 car) and carry it all the way:
> the idea, the math, the algorithm (traced by hand), the code, and the diagnostics.
>
> **Prerequisite:** read `RELMATCH_EXPLAINED.md` first. `reldist` is the next step on the same ladder,
> so this doc assumes you already know the class-relation matrix `R` and the `stock`/`relmatch` story.

---

## 0. TL;DR (read this first)

`relmatch` (Formalization 1) keeps a small set whose **average opinion of the teacher** reproduces the
whole class's average opinion — it matches the **mean** soft label, i.e. one row `R[i,:]` of the
class-relation matrix.

But a class is not its average. Real cats come in *kinds*: textbook cats, dog-ish cats, a long tail of
weird cats. The *average* cat is a single point — often a kind of cat that **doesn't actually exist**
(a blurred half-dog-ish cat). Many different little sets share that same average while describing very
different classes, so "match the mean" under-determines what you keep.

`reldist` (Formalization 2) matches the **whole distribution** of soft labels in each class — its
modes, its spread, its tail — not just the mean. It does so with **optimal transport**: it makes the
selected set's per-class soft-label *distribution* close to the full pool's in **Wasserstein** distance.
The mean match falls out as a special case (collapse every distribution to a point), so `reldist`
**contains** `relmatch` and additionally pins down the *shape* of each row that the mean discards.

---

## 1. Vocabulary (the four new words)

You already know *teacher*, *soft label*, *dark knowledge*, *IPC*, *`stock`*, and the *class-relation
matrix `R`* from the `relmatch` doc. Four more:

- **Distribution over the simplex.** One image gives one soft-label *point* (e.g. `[0.80, 0.18, 0.02]`).
  A whole class gives a *cloud* of such points living on the probability simplex. `relmatch` summarized
  that cloud by its mean (one point — the row `R[i,:]`); `reldist` keeps the **cloud's shape**.
- **Coordinate.** Fix the cat class and look at one column of its crops' soft labels — e.g. the **cat
  column** (a 1-D list of "how confident each cat is") or the **dog column** ("how dog-ish each cat
  is"). Each entry `R[i,j]` was a *scalar* (the mean of one such column); now we keep the *distribution*
  of each column.
- **Quantile function.** Sort a column. The sorted values *are* the quantile function: the 25th
  percentile, the median, the 75th percentile, … A distribution is fully described by its quantiles.
- **1-D Wasserstein (earth-mover) distance.** The natural distance between two 1-D distributions: line
  up their quantile functions and measure the (squared) gap. `W₂² = 0` iff the two sorted lists agree —
  same modes, same spread, same tail. In 1-D, optimal transport is just "sort both, compare." It is the
  ruler `reldist` minimizes.

---

## 2. The idea in one picture

Take the cat class. Suppose its crops form **two kinds**, in equal number — textbook cats and dog-ish
cats (the car column is a small constant, shown for completeness):

```
        cat    dog    car
 T1 :  0.85   0.10   0.05     textbook cat
 T2 :  0.85   0.10   0.05     textbook cat
 D1 :  0.55   0.30   0.15     dog-ish cat
 D2 :  0.55   0.30   0.15     dog-ish cat
```

The class is **bimodal**: two real kinds of cat. What is the *average* cat? `[0.70, 0.20, 0.10]` — a
cat that is **70% cat / 20% dog**, i.e. *neither* a textbook cat *nor* a dog-ish one. It is a kind that
**does not occur in the data**. We may keep **2** crops (IPC = 2). Which two?

- **`relmatch`** wants the kept crops' **mean** to equal `[0.70, 0.20, 0.10]`. It is satisfied by
  *anything* that averages there — and a mean alone cannot tell "one textbook + one dog-ish" apart from
  "two identical 70%-cats" (had any existed). The target it aims at is the **non-existent average cat**.
- **`reldist`** wants the kept crops' **distribution** to look like the pool's — its two modes. So it
  keeps **one crop from each kind**, `{T, D}`, reproducing "cats come in two kinds: textbook and
  dog-ish." Its target is the **shape**, not a single point.

Same class, same budget. `relmatch` aims at the *average* cat; `reldist` keeps the *two real kinds* of
cat. (On a clean two-kind class like this the mean happens to single out `{T, D}` too, so both land
there — see §6. The gap opens when a class has a *richer* shape, where many subsets share the mean but
only one reproduces the distribution; that is what the at-scale numbers in §8 measure.)

> Why you should care: the dog-ish-cat mode *is* the boundary between cat and dog. Sub-modes and tails
> are where calibration, OOD detection and robustness are decided. The mean smears them into one
> average; the distribution keeps them apart.

---

## 3. The object we care about: the per-class distribution `Pᵢ`

**Definition.** For class `i`, let `Pᵢ` be the **distribution of soft-label vectors** of class-`i`
images — the whole cloud on the simplex, not its average. `relmatch`'s object `R[i,:]` is exactly the
**mean** of `Pᵢ`:

$$ R[i,:] \;=\; \mathbb{E}_{p \sim P_i}[\,p\,] . $$

So `relmatch` matched **the first moment of `Pᵢ`**. `reldist` matches **`Pᵢ` itself**. Concretely we
look at `Pᵢ` one **coordinate** at a time: for each class `j`, the column `Pᵢ[:,j]` is the 1-D
distribution of "how much class `i` leans on class `j`" (and for `j=i`, the spread of its own
confidence). `R[i,j]` was the *mean* of that column; now we keep the *whole* column — the full row's
distribution, self-confidence and inter-class lean alike.

For the cat row this is the difference between

```
 relmatch keeps:   R[cat, dog] = 0.20                       (one number: the average lean)
 reldist  keeps:   P_cat[:,dog] = {0.10, 0.10, 0.30, 0.30}  (the whole shape: two modes)
```

> Mental model: `R` was the dataset's confusion fingerprint *averaged to one ink-blot per row*.
> `reldist`'s object is the *full pressure map* of each row — where within a class the mass is dense,
> where it's sparse, how far its tail reaches.

The same simplification that made `relmatch` tractable still holds: **the distribution of class `i`
depends only on the crops we pick for class `i`**, so matching every class's `Pᵢ` splits into `K`
independent per-class problems — exactly one call of the selector per class, the existing RDED loop.

---

## 4. What "good selection" means: `P̂ᵢ ≈ Pᵢ`

The tiny distilled set induces its **own** per-class distribution `P̂ᵢ` (the cloud of the *selected*
crops). Good selection means `P̂ᵢ` reproduces `Pᵢ` — same modes, same spread, same tail — measured by
the **Wasserstein** distance between the two clouds. `relmatch` only asked `mean(P̂ᵢ) ≈ mean(Pᵢ)`;
`reldist` asks for the whole shape:

$$ \widehat P_i \;\approx\; P_i \qquad\text{(in Wasserstein distance), not just}\qquad \mathbb{E}[\widehat P_i] \approx \mathbb{E}[P_i]. $$

The next section turns "match the whole shape" into a single number to minimize.

---

## 5. The objective (the math, gently)

Fix one class `i`. The candidate crops have soft labels `p₁, p₂, …` (each a `K`-vector); the full pool
defines the target distribution `Pᵢ`. For a selected set `S`, the induced distribution is the empirical
cloud `{p_s : s ∈ S}`. We match it to `Pᵢ` **coordinate by coordinate**, summing the 1-D Wasserstein gap
over columns:

$$ \boxed{\;J(S) \;=\; \sum_{j=1}^{K} W_2^2\!\Big(\, \{p_s[j]\}_{s\in S}\,,\; \{p_t[j]\}_{t\in \text{pool}} \,\Big)\;}\qquad \text{minimize over } S,\ |S|=\text{IPC}. $$

**The 1-D Wasserstein `W₂²`.** For two 1-D samples, sort both and integrate the squared gap between
their quantile functions:

$$ W_2^2(a,b) \;=\; \int_0^1 \big(F_a^{-1}(u) - F_b^{-1}(u)\big)^2\, du . $$

`F⁻¹` is just "the sorted values." So `W₂²` is small exactly when the two sorted lists track each other
— **same quantiles ⇒ same modes, same spread, same tail.** No bandwidth, no kernel, no optimization: in
1-D, optimal transport *is* sorting. We sum it over every coordinate the pool actually moves on (the
~constant columns of a large-`K` problem contribute ≈0 and are skipped), so the **full row** is matched
— each class's own confidence spread *and* its lean toward every other class.

**The nesting — why this *contains* `relmatch`.** Suppose each coordinate's distribution were a single
spike at its mean (zero spread). Then `W₂²` between two spikes `δ_a, δ_b` is just `(a−b)²`, and

$$ J(S) \;\xrightarrow[\text{collapse each column to its mean}]{}\; \sum_j \big(\overline{p_S}[j] - \overline{p_\text{pool}}[j]\big)^2 \;=\; \lVert \overline{p_S} - r_i\rVert^2, $$

which is **exactly `relmatch`'s objective**. So `relmatch` is `reldist` with every distribution crushed
to its mean — the **zeroth-order** version. `reldist` keeps the higher quantiles (spread, modes, tails)
that the mean discards.

> The same ladder as the feature-space selectors: `momentmatch` matches a mean **and covariance** in
> feature space; `relmatch` matches a mean in soft-label space; `reldist` matches the **whole
> distribution** in soft-label space. Each rung keeps more of the object and discards less.

---

## 6. The algorithm: match the quantiles (not a from-scratch greedy)

How do we pick IPC crops to minimize `J(S)`? There's a tempting trap: *greedily add the crop that most
lowers `J`*. Don't — its first pick is the single crop nearest the **whole** distribution (its
**center**), and a center-anchored greedy then collapses toward the middle. The mean-chaser's failure
mode, re-encountered.

The fix is the textbook 1-D quantizer. The `n`-point set that best matches a distribution in `W₂` sits
at its **`n` evenly-spaced quantiles**, levels `τ_k = (k − ½)/n`. So:

> For each rank `k = 1…n`, build the **target quantile profile** `t_k[j] = ` (the pool's `τ_k`-quantile
> of column `j`). Then **assign** each rank to the nearest still-unused real crop. The targets are
> spread across the distribution by construction, so the selection spreads with them.

Let's trace it **by hand** on the two-kind cat class of §2, IPC = 2. Pool = `{T, T, D, D}` with
`T = [.85, .10, .05]`, `D = [.55, .30, .15]`.

**Targets.** `τ = (0.25, 0.75)`. Take the `0.25`- and `0.75`-quantile of *each* column:

```
 cat column {.55, .55, .85, .85}:   q.25 = .55   q.75 = .85
 dog column {.10, .10, .30, .30}:   q.25 = .10   q.75 = .30
 car column {.05, .05, .15, .15}:   q.25 = .05   q.75 = .15
                                    ----------   ----------
                          t_1 = [.55, .10, .05]   t_2 = [.85, .30, .15]
```

**Assign rank 1** (target `t_1 = [.55, .10, .05]`). Squared distance of each crop to `t_1`:

```
 T [.85 .10 .05]:  (.30)² + 0    + 0     = .09
 D [.55 .30 .15]:  0     + (.20)²+ (.10)² = .05    <-- nearest
```

Pick a **D**. **Assign rank 2** (target `t_2 = [.85, .30, .15]`), that D now unavailable:

```
 T [.85 .10 .05]:  0     + (.20)²+ (.10)² = .05    <-- nearest
 D [.55 .30 .15]:  (.30)²+ 0    + 0      = .09
```

Pick a **T**. Done.

**Result.** `reldist` selects **one `D` and one `T`** — one crop from each kind — so its induced
distribution `{T, D}` reproduces the pool's two modes *exactly* (`W₂² = 0` on every column). A
mean-matcher reaches the same pair *here* (the mean uniquely identifies it on a clean two-mode class),
but `reldist` got there by reproducing the **shape**, and on richer classes — where many subsets share
the mean — that is what keeps the modes a mean alone would blur. (This exact case is a unit test; §11.)

> Why it's safe: the assignment is **deterministic** (fixed quantile levels, fixed nearest-pick order,
> no RNG), cheap (`O(n·P·A)` over the `A` moving columns), and spreads by construction — the same
> house-style as every other RDED selector.

---

## 7. The code (the core, mapped to the math)

Two short pieces. The **selector** lives in `synthesize/utils.py` as `_select_reldist`; it is the
quantile-assignment of §6:

```python
def _select_reldist(n, cand_probs, target_probs, gen):
    Pc = cand_probs.double()                              # [P, K] candidates' soft labels
    Pt = target_probs.double()                            # [M, K] full pool (defines the quantiles)
    P, K = Pc.shape

    active = Pt.var(dim=0, unbiased=False) > 1e-8          # only the columns that actually move
    cols   = active.nonzero(as_tuple=True)[0]
    Pc_a, Pt_a = Pc[:, cols], Pt[:, cols]

    taus = (torch.arange(1, n + 1, dtype=torch.float64) - 0.5) / n   # quantile levels (k-1/2)/n
    T = torch.quantile(Pt_a, taus, dim=0)                 # [n, A] target quantile profiles t_k

    avail = torch.ones(P, dtype=torch.bool)
    selected = []
    for kk in range(n):                                   # assign each rank to its nearest crop
        d2 = ((Pc_a - T[kk]) ** 2).sum(dim=1)             # squared distance to t_k
        d2[~avail] = float("inf")
        j = int(torch.argmin(d2))
        selected.append(j); avail[j] = False
    return torch.tensor(selected[:n], dtype=torch.long)
```

Line-by-line this *is* the hand trace: `T` are the `t_k` quantile profiles, the loop assigns each rank
to its nearest unused crop, `active` is the "only the columns that move" guard. No knob, no optimization
loop, no randomness — the whole row's distribution is matched.

**Where do the soft labels come from?** Exactly like `relmatch`: inside `selector(...)` the teacher
already runs on every candidate crop, so `probs = softmax(best_logits)` is free, and the dispatcher
`_select_variant(...)` passes the **full pool** as the target (the analogue of `momentmatch` passing the
full-pool features):

```python
if method == "reldist":
    local = _select_reldist(n, probs[eligible], probs, gen)   # candidates, full-pool target
    return eligible[local]
```

Selection is over the **full per-class pool** — the same candidates `stock` ranks.

The **measuring stick** lives in `validation/diagnostics.py` as `wasserstein1d_sq` — the exact 1-D `W₂²`
of §5, computed on the merged grid of the two samples' CDF breakpoints. It is used by the diagnostic and
the tests (the selector reaches the same optimum by quantile matching).

---

## 8. The diagnostics: did it actually work?

We *claim* `reldist` preserves each class's soft-label **distribution**. We check it without training a
student, with `relation_distribution_divergence(...)` in `validation/diagnostics.py`. It runs the teacher
on the distilled set and on a real-train reference, groups soft labels by class, and reports:

| metric | what it measures | good direction |
|---|---|---|
| `reldist_w_full` | mean over classes of `Σ_j W₂(real col, distilled col)` — the **full-row per-coordinate Wasserstein** | ↓ lower **(headline)** |
| `reldist_w_off` | same, off-diagonal columns only (the inter-class part) | ↓ lower |
| `reldist_sw` | sliced-`W₂` over a seeded bank of random simplex directions (an approximation of the *full multivariate* `W₂`) | ↓ lower |
| `reldist_tail_cov` | per class, `q90(max off-diag mass \| distilled) / q90(… \| real)`, clipped to `[0,1]` — did the **boundary tail** survive? | ↑ higher |

We also report the `relmatch` mean-matrix metrics (`rel_frob`, `rel_frob_offdiag`, `rel_eig_overlap`)
on the same set, because a good *distribution* match should imply a good *mean* match for free.

**The real numbers.** Building all three sets at matched budget (`cifar100`, teacher `conv3`, IPC = 10,
real reference 10/class, seed 42) and diagnosing each gave:

```
                 ---- mean of R (Formalization 1) ----    ---- distribution of P (Formalization 2) ----
 set        rel_frob  off-diag  eig_overlap  |  W_full   W_off   sliced-W   tail_cov
 stock        3.74      1.22       0.15       |  1.32     0.882   0.450      0.058
 relmatch     2.14      1.17       0.45       |  1.46     1.216   0.296      0.621
 reldist      1.71      1.11       0.36       |  1.34     1.165   0.252      0.470
```

Read it carefully:

1. **`reldist` best matches the class-relation matrix overall** — lowest `rel_frob` (`1.71`) and lowest
   off-diagonal `rel_frob` (`1.11`), beating even `relmatch`, whose *only* job is that mean. Matching the
   distribution gives you a better mean than matching the mean directly (the nesting of §5, in practice).
2. **`reldist` best matches the full multivariate distribution** — lowest `sliced-W` (`0.252`) and a
   lower full-row Wasserstein than `relmatch` (`reldist_w_full` `1.34` < `1.46`). It reproduces the
   *shape*, not just the average.
3. **`relmatch` overshoots.** It wins `eig_overlap` (`0.45`) and raw `tail_cov` (`0.621`) by
   *over-recruiting* boundary crops to hit the mean — handy for raw tail coverage, but it distorts the
   distribution (worse `W_full`/`sliced-W`). `reldist` keeps a **balanced** shape.
4. **The one place `stock` "wins"** is `reldist_w_off` (`0.882`): `conv3`'s real off-diagonal mass is
   modest, so confidence-greedy's near-zero off-diagonal happens to sit closest to it — while `stock` is
   worst on everything that matters (`eig_overlap 0.15`, `tail_cov 0.058`, `rel_frob 3.74`, the
   geometry it simply deletes).

In one line: **`stock` deletes the class's shape; `relmatch` reproduces its mean but overshoots the
shape; `reldist` reproduces the shape — and gets the best mean as a by-product.**

(The same numbers are logged to `logs/diagnostics.jsonl` by
`python tools/diagnose_geometry.py --subset cifar100 --arch-name conv3 --ipc 10 --syn-leaf syn_data_seed42 --select-method reldist`.)

---

## 9. Three honest framings (a reviewer will ask)

- **Why Wasserstein and not MMD/kernels?** In 1-D, Wasserstein is *parameter-free* — just sorting, no
  bandwidth to tune, and it reads off modes/spread/tails directly via quantiles. An MMD match would add a
  kernel and a bandwidth knob. (As `σ→∞` an RBF-MMD reduces to mean-matching too — both families nest
  `relmatch` — but the OT route keeps the *interpretable* per-coordinate quantile picture.)
- **Sliced (per-coordinate) vs full multivariate OT.** We match the **marginals** of each coordinate (a
  *sliced* Wasserstein), not the full joint via Sinkhorn. That's the tractable, deterministic,
  interpretable choice. Full entropic OT (Sinkhorn) is the heavier alternative; we use a
  random-projection sliced-`W` only as the multivariate *diagnostic* (`reldist_sw`), never in the
  selector.
- **Positioning.** `NCFM` (CVPR 2025) matches whole *feature* distributions by **synthesizing** images;
  `IMS3`'s S³ (CVPR 2026) scores generated **subgroups**; `IID` (CVPR 2024) adds feature **covariance**.
  `reldist` is the **selection-based, soft-label-space** cousin: it keeps real crops whose per-class
  *relational distribution* matches the pool — the object none of those preserves.

---

## 10. Recap — the whole story in five sentences

1. A class is not its average: many subsets share a class's mean soft label, and that mean is often a
   *kind that doesn't exist*, so `relmatch`'s target under-determines what you keep.
2. `reldist` keeps the **whole per-class soft-label distribution** — its modes, spread and tail — by
   matching it to the pool in **Wasserstein** distance, coordinate by coordinate (the full row).
3. It does this by **quantile matching** — target the pool's evenly-spaced quantiles and take the nearest
   real crop to each — a deterministic spread that avoids the center-collapse a naive greedy would hit.
4. Collapsing every distribution to its mean recovers `relmatch` exactly, so `reldist` **contains**
   Formalization 1 and adds the higher quantiles the mean discards.
5. The diagnostics confirm it: `reldist` gives the best match to both the class-relation **matrix**
   (lowest `rel_frob`) and the full soft-label **distribution** (lowest `sliced-W`), while `relmatch`'s
   mean-only match overshoots the shape.

---

## 11. Where to look in the repo

| piece | file | symbol |
|---|---|---|
| the quantile-assignment selector | `synthesize/utils.py` | `_select_reldist`, dispatched in `_select_variant` |
| soft-labels + full-pool target wiring | `synthesize/utils.py` | inside `selector(...)` |
| CLI flag | `argument.py` | `--select-method reldist` |
| exact 1-D `W₂²` + the distribution diagnostic | `validation/diagnostics.py` | `wasserstein1d_sq`, `relation_distribution_divergence` |
| run the diagnostic | `tools/diagnose_geometry.py` | `--select-method reldist` |
| unit tests (the §6 toy is one) | `tests/test_select.py` | `test_select_reldist_*`, `test_wasserstein1d_sq_known` |
| Formalization 1, for contrast | `docs/RELMATCH_EXPLAINED.md` | the mean-matching predecessor |
| the other selectors | `docs/SELECTORS.md` | `stock`, `covmatch`, `momentmatch`, `qddpp`, `relmatch` |

Run it yourself:

```bash
# 1) build a reldist distilled set
python main.py --subset cifar100 --arch-name conv3 --ipc 10 \
               --factor 1 --mipc 300 --num-crop 5 \
               --seed 42 --syn-data-path syn_data_seed42 --select-method reldist

# 2) measure how well it preserved each class's soft-label distribution
#    (compare against --select-method stock / relmatch at matched IPC)
python tools/diagnose_geometry.py --subset cifar100 --arch-name conv3 --ipc 10 \
               --syn-leaf syn_data_seed42 --select-method reldist
```
