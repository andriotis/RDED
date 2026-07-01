# The `analyze_select` table — every metric, from intuition to math

> A teaching walkthrough in the same spirit as [`FACLOC_EXPLAINED.md`](FACLOC_EXPLAINED.md),
> [`RELMATCH_EXPLAINED.md`](RELMATCH_EXPLAINED.md) and [`RELDIST_EXPLAINED.md`](RELDIST_EXPLAINED.md).
> Those documents explain *selectors*; this one explains the **scoreboard** they are judged by —
> the per-metric columns printed by [`tools/analyze_select.py`](../tools/analyze_select.py). This is
> **math + intuition**; the metric *implementations* live in
> [`validation/diagnostics.py`](../validation/diagnostics.py) and
> [`validation/nc_metrics.py`](../validation/nc_metrics.py).

---

## 0. TL;DR (read this first)

`analyze_select.py` is the **H2 causal readout**: it answers *"did swapping stock confidence-greedy
selection for a variance-aware selector (`covmatch`/`relmatch`/`reldist`/`facloc`/…) make the trained
**student** more trustworthy?"* Each row is a metric; each block of columns is a selector variant
(recovered from the `_sel<method>` tag in `exp_name`, so `relmatch` and `relmatch_dw1` are separate
columns). After every non-stock variant sits a **Δ vs stk** column. The colouring is **direction-aware**:

- **green** = better than stock, **red** = worse, on *this* metric's known good direction;
- **bold** = best value across all variants for that metric;
- metrics with **no natural direction** (only `nc1` here) are never coloured — they are diagnostic,
  not a verdict.

Everything reduces to one question per metric: *holding IPC and architecture fixed, did the selector
move this number the right way relative to stock?* The rest of this document is what "the right way"
means, metric by metric.

The metrics fall into four families:

| family | metrics | reads |
|---|---|---|
| **accuracy** | `top1` | did we keep the task working? |
| **representation geometry** | `nc1` | how collapsed is the student's feature space? (diagnostic, no direction) |
| **soft-label content** | `sl_entropy`, `sl_ent_std`, `sl_conf`, `sl_conf_std`, `sl_eff_n`, `sl_eff_n_std` | how much *inter-class / relational* signal does the distilled set carry? (the **lever** — the cause we intervene on) |
| **trustworthiness** | `ece`, `ece_ts`, `auroc_{msp,energy,maha,featnorm}`, `oscr_msp`, `fpr95_msp` | calibration + open-set/OOD detection (the **outcome** we hope to move) |

The causal logic of the table: the soft-label family is the **lever** the selector pulls directly;
calibration + OOD are the **downstream outcome**; `top1` is the **guardrail** (the gains must not cost
accuracy); `nc1` is the **mechanism witness**.

---

## 1. Accuracy — `top1`

The plain one. After training the student only on the distilled set, evaluate on the real validation
set and report top-1 accuracy (%):

$$
\texttt{top1} \;=\; 100 \cdot \frac{1}{N}\sum_{n=1}^{N}\mathbf{1}\big[\hat y_n = y_n\big],
\qquad \hat y_n = \arg\max_k \mathrm{logit}_k(x_n).
$$

**Direction: higher is better.** Its role here is a **guardrail**, not the headline. The whole
trustworthy-distillation thesis is that you can buy calibration/OOD gains *without* paying accuracy.
So the bar `top1` must clear is not "beat stock" but "**don't lose to stock**" — a selector that lifts
AUROC by 3 points while dropping `top1` by 4 has not made a free trustworthiness gain, it has just
traded one axis for another. Watch the Δ here as a veto.

---

## 2. Representation geometry — `nc1` (`nc1(stu/val)`)

`nc1` is the first Papyan–Han–Donoho **neural-collapse** metric, computed on the **student's**
penultimate features over the **real validation set** (hence the `stu/val` label — it is the H2,
trained-model quantity, not a property of the data). Let `Σ_W` be the within-class scatter and `Σ_B`
the between-class scatter of the features:

$$
\Sigma_W = \frac{1}{N}\sum_{n}(z_n - \mu_{y_n})(z_n - \mu_{y_n})^\top,
\qquad
\Sigma_B = \frac{1}{K}\sum_{k}(\mu_k - \mu_G)(\mu_k - \mu_G)^\top,
$$

with `μ_k` the class-`k` feature mean, `μ_G` the global mean. Then

$$
\texttt{nc1} \;=\; \frac{1}{K}\,\operatorname{tr}\!\big(\Sigma_W \,\Sigma_B^{+}\big),
$$

where `Σ_B^+` is the pseudo-inverse. Read it as a **signal-to-noise ratio of the feature space,
inverted**: it is large when within-class spread `Σ_W` is big relative to the between-class
separation `Σ_B` (classes smeared together, *under-collapsed*), and small when each class collapses to
a tight, well-separated cluster (*over-collapsed*).

**Direction: none** — this is the one metric the table refuses to colour. The reason is the crux of
the whole study. Confidence-greedy `stock` drives the student toward *strong* collapse (tight textbook
clusters), which **lowers** `nc1` — and on a naïve "collapse = good representation" reading that looks
like a win. But strong collapse is exactly what *flattens the inter-class boundary structure* that
calibration and OOD detection depend on. So a lower `nc1` is **not** automatically better here, and a
higher one is not automatically worse. `nc1` is printed as a **mechanism witness**: you read it
*alongside* the trustworthiness columns to see *how* a selector moved the geometry, not to score it.
(At low IPC `nc1` is generally high — the student is under-collapsed on held-out data regardless of
selector; see [`EXPERIMENTS.md`](EXPERIMENTS.md).)

---

## 3. Soft-label content — the lever

These six come from `soft_label_stats` and summarise the **teacher soft-labels** $$p_n =
\mathrm{softmax}(\text{teacher logits}(x_n))$$ over the **selected crops**. They describe *the data the
student will be trained on*, before any training happens — so they are the variable the selector
controls **directly**. Every one is a statistic of the per-crop entropy / confidence.

For each selected crop `n` define

$$
H_n = -\sum_{k} p_{n,k}\log p_{n,k}\ \text{(nats)},
\qquad
\mathrm{conf}_n = \max_k p_{n,k},
\qquad
\mathrm{eff\_n}_n = e^{H_n}.
$$

`H_n` is the Shannon entropy of the soft label; `eff_n` is its **perplexity** — the *effective number
of classes* the teacher spreads mass over for that crop (`1.0` = perfectly one-hot, `K` = uniform).
The table reports the mean and standard deviation (over crops) of each:

| column | symbol | direction | reads |
|---|---|---|---|
| `sl_entropy` | `mean_n H_n` | **higher** | how *diffuse* the soft labels are — more entropy = more off-diagonal (inter-class) mass retained |
| `sl_ent_std` | `std_n H_n` | **higher** | *heterogeneity* of the selection — a mix of sharp and soft crops, not one monotone regime |
| `sl_conf` | `mean_n conf_n` | **lower** | average peak mass; lower confidence ⇔ more mass left for other classes |
| `sl_conf_std` | `std_n conf_n` | **higher** | spread in crop confidence — again, a *varied* set, not all textbook exemplars |
| `sl_eff_n` | `mean_n e^{H_n}` | **higher** | mean effective #classes per crop — the richness of the inter-class signal |
| `sl_eff_n_std` | `std_n e^{H_n}` | **higher** | spread in per-crop effective-#classes |

### Two-step reduction (what "mean of `H_n`" actually means)

Every column above is a *double* reduction — it helps to keep the two steps separate:

1. **Per crop, over classes (`K → 1`).** Each selected crop `n` has a soft label `p_n`, a *vector* of
   length `K` (one probability per class). `H_n = -Σ_k p_{n,k} log p_{n,k}` collapses that whole
   `K`-vector into a **single scalar** measuring how spread-out *that one crop's* label is (`conf_n` and
   `eff_n` likewise collapse `p_n` to one scalar). After this step the set is just a list of `N`
   numbers `{H_1, …, H_N}`, one per crop.

2. **Over crops (`N → 2`).** The `_mean`/`_std` columns are the mean and standard deviation **of that
   list of scalars**: `sl_entropy = (1/N) Σ_n H_n`, `sl_ent_std = std_n(H_n)`.

So `sl_entropy` is *not* one entropy of the whole set — it is the **average of the per-crop
entropies**, and `sl_ent_std` is how much those per-crop entropies differ from each other.

The two summaries answer different questions about the *same* list:

- **mean** → "how diffuse is a *typical* crop's soft label?"
- **std** → "do the crops *differ* from one another in how diffuse they are?"

The std is the easy one to misread. A set of all-textbook crops (all low `H_n`) and a set of
all-maximally-ambiguous crops (all high `H_n`) have *very different means* but **both have near-zero
std**, because within each set the crops resemble each other. High `sl_ent_std` specifically means the
selection *mixes* sharp anchors with soft boundary crops — the heterogeneity the variance-aware
selectors are after. The same `K→1` then `N→2` structure applies verbatim to `sl_conf*` (via `conf_n`)
and `sl_eff_n*` (via `eff_n`).

The unifying intuition: **stock keeps only the most confident crops** (textbook cats), which drives
`sl_entropy`, `sl_eff_n` down and `sl_conf` up, and — because every kept crop looks the same — drives
all three `*_std` columns *down* too. The soft-label cloud collapses onto its diagonal. A
variance-aware selector that deliberately keeps boundary / sub-mode crops (the dog-ish cats of
[`FACLOC_EXPLAINED.md`](FACLOC_EXPLAINED.md)) **raises entropy and its spread, lowers confidence**, and
keeps the off-diagonal mass alive. So green on this family is the **direct evidence the intervention
fired** — the cause, before we ask whether the effect followed.

Two subtleties worth stating:

- The `*_std` columns are marked "higher = better" because the thesis prizes a *heterogeneous*
  selection (some sharp anchors, some boundary crops), not because more variance is good in the
  abstract. They are a **diversity** read, paired with their `_mean` partner.
- `sl_entropy` and `sl_eff_n` are monotone transforms of each other per crop (`eff_n = e^H`), so their
  *means* move together but are not redundant: `sl_eff_n` is in *interpretable class-count units*,
  which is why both are kept.

---

## 4. Calibration — `ece`, `ece_ts`

A model is **calibrated** when its confidence matches its accuracy: of all predictions made at
confidence 0.8, about 80 % should be correct. **Expected Calibration Error** measures the gap. Bin the
test predictions into `B = 15` equal-width confidence bins; in bin `b` let `acc_b` be the empirical
accuracy and `conf_b` the mean confidence. Then

$$
\texttt{ece} \;=\; \sum_{b=1}^{B} \frac{|\mathcal B_b|}{N}\,\big|\,\mathrm{acc}_b - \mathrm{conf}_b\,\big|,
$$

the sample-weighted average bin-wise gap. **Direction: lower is better** (0 = perfectly calibrated).
Distilled-data students are typically *overconfident* (`conf_b > acc_b`), so a selector that keeps the
relational manifold — leaving the student less sure on genuinely ambiguous inputs — should **lower**
ECE.

`ece_ts` is the **same metric after temperature scaling**: a single scalar `T > 0` is fit on a
held-out real-train slice by minimising NLL of `softmax(logits / T)`, then

$$
\texttt{ece\_ts} \;=\; \mathrm{ECE}\big(\mathrm{logits}/T,\ \text{labels}\big).
$$

Temperature scaling is the standard cheap post-hoc fix for global over/under-confidence. The contrast
between `ece` and `ece_ts` is informative: if `ece` is high but `ece_ts ≈ 0`, the miscalibration was
just a global temperature (easily fixed, not deeply concerning). If `ece_ts` *stays* high, the
miscalibration is **structural** — confidence is wrong in a way no single temperature repairs — and a
selector that lowers `ece_ts` has fixed something temperature scaling cannot. That makes `ece_ts` the
more demanding, more interesting calibration column.

---

## 5. Open-set / OOD detection — the AUROC, OSCR, FPR@95 columns

These ask the trustworthiness question stock most clearly fails: *can the student tell when an input
is **not** one of its classes?* Every method produces a scalar **score** per input, with the universal
convention **higher = more in-distribution (ID)**. Each metric then measures how well that score
separates a real-val ID set (positives) from an OOD negative set.

### 5.1 The four scores (what is being thresholded)

| score | definition | reads |
|---|---|---|
| **MSP** | `max_k softmax(logits)_k` | maximum softmax probability — the model's raw confidence |
| **energy** | `T·logsumexp(logits/T)` | negative free energy (Liu et al. 2020); uses *all* logits, not just the max |
| **maha** | `max_k −d²_k(z)` with `d²_k` the tied-covariance Mahalanobis distance to class mean `μ_k` | a **geometry** score in feature space — how well `z` fits the class-conditional Gaussians |
| **feat_norm** | `‖z‖₂` | penultimate feature magnitude — ID inputs activate strongly, OOD collapse toward 0 |

The Mahalanobis score deserves a note because it is the one tied most tightly to the variance thesis.
Features are **L2-normalised** before fitting a single tied covariance `Σ` (shrunk toward a scaled
identity so it inverts), and the score is the max over classes of the negative squared Mahalanobis
distance:

$$
d_k^2(z) = (z-\mu_k)^\top \Sigma^{-1} (z-\mu_k),
\qquad
\texttt{maha}(z) = -\min_k d_k^2(z).
$$

Normalisation makes the score read **directional** (cosine-space) consistency with the class structure
— precisely the directional variance the variance-aware selectors are built to preserve — while the
discarded magnitude is captured separately by `feat_norm`, making the two **complementary** reads
([`rded-mahalanobis-normalization`] in memory; this is non-optional — a raw Mahalanobis inverts and
ranks OOD as "closer"). The within-class covariance `Σ` it reads is the **mediator** the study
predicts links variance-preserving selection to the AUROC gain.

### 5.2 AUROC (`auroc_msp`, `auroc_energy`, `auroc_maha`, `auroc_featnorm`)

Area under the ROC curve for ID-vs-OOD, computed **exactly** via the Mann–Whitney U statistic (no
sweep, tie-aware). With `n⁺` ID and `n⁻` OOD scores, ranking all `n⁺+n⁻` scores ascending and summing
the ID ranks `R⁺`:

$$
\texttt{auroc} \;=\; \frac{R^{+} - \tfrac{1}{2}n^{+}(n^{+}+1)}{n^{+}\,n^{-}}
\;=\; \Pr\big[\text{score}(\text{ID}) > \text{score}(\text{OOD})\big].
$$

So AUROC is literally the probability a random ID input scores higher than a random OOD input. **0.5 =
random, 1.0 = perfect; higher is better.** Four scores ⇒ four AUROC columns, because a selector can
help one score family and not another — e.g. lift `auroc_maha` (it fixed the *feature geometry*) while
barely moving `auroc_msp` (it didn't change the *softmax* confidence). Reading them together localises
*where* the trustworthiness gain lives.

### 5.3 OSCR (`oscr_msp`)

**Open-Set Classification Rate** folds accuracy and OOD-rejection into one curve. Sweeping a threshold
`δ` downward over the MSP score, define the **correct classification rate** and **false-positive rate**

$$
\mathrm{CCR}(\delta) = \frac{\#\{\text{ID correctly classified} \ \wedge\ \text{score}\ge\delta\}}{N_\text{id}},
\qquad
\mathrm{FPR}(\delta) = \frac{\#\{\text{OOD}\ \wedge\ \text{score}\ge\delta\}}{N_\text{ood}},
$$

and report the area under the CCR-vs-FPR curve (trapezoid). **Direction: higher is better.** The
difference from AUROC: OSCR demands the admitted ID inputs be **not just admitted but correctly
classified** — it is unforgiving of a model that confidently lets in an OOD input *or* misclassifies an
ID one. It is the most "end-to-end" open-set number in the table.

### 5.4 FPR@95 (`fpr95_msp`)

The operational complaint metric. Fix the threshold that **admits 95 % of ID** inputs (the 5th
percentile of ID scores); report the fraction of OOD that sneaks past it:

$$
\texttt{fpr95} \;=\; \frac{1}{N_\text{ood}}\sum_j \mathbf{1}\big[\text{score}(\text{OOD}_j) \ge \tau_{95}\big],
\qquad \tau_{95} = \text{5th percentile of ID scores}.
$$

**Direction: lower is better** (0 = no OOD leaks at 95 % ID recall). Where AUROC is threshold-free and
averaged over all operating points, FPR@95 pins one *realistic* operating point ("I insist on keeping
95 % of my real traffic — how much garbage do I let in?"). A selector can improve average AUROC yet
worsen this specific corner, so it is reported on its own.

---

## 6. How to read a column at a glance

Putting the direction conventions in one place — this is exactly what drives the green/red colouring:

| metric | direction | family | role |
|---|---|---|---|
| `top1` | ↑ higher | accuracy | guardrail — must not regress |
| `nc1` | — none | geometry | mechanism witness (uncoloured) |
| `sl_entropy`, `sl_ent_std` | ↑ higher | soft-label | lever |
| `sl_conf` | ↓ lower | soft-label | lever |
| `sl_conf_std` | ↑ higher | soft-label | lever |
| `sl_eff_n`, `sl_eff_n_std` | ↑ higher | soft-label | lever |
| `ece`, `ece_ts` | ↓ lower | calibration | outcome |
| `auroc_msp`, `auroc_energy`, `auroc_maha`, `auroc_featnorm` | ↑ higher | OOD | outcome |
| `oscr_msp` | ↑ higher | open-set | outcome |
| `fpr95_msp` | ↓ lower | OOD | outcome |

The **story a good selector tells in one table**: green across the soft-label lever (entropy up,
confidence down, spreads up) → green across the calibration + OOD outcome (ECE down, AUROC/OSCR up,
FPR@95 down) → **no red** on the `top1` guardrail, with `nc1` shifting toward *less* collapse as the
uncoloured corroboration. That is the H2 trustworthiness claim, made visible: a variance-aware
intervention on the *data* (the lever) caused a trustworthiness gain in the *trained student* (the
outcome), for free on accuracy.

---

## 7. One line each

- `top1` — accuracy; the gain must be free of it (guardrail).
- `nc1` — student feature collapse; diagnostic, deliberately uncoloured (more collapse ≠ better).
- `sl_entropy` / `sl_eff_n` (+ their `_std`) — how much diffuse inter-class signal the selected crops carry; the lever, ↑.
- `sl_conf` (+ `sl_conf_std`) — peak soft-label mass; ↓ mean, ↑ spread = relational structure kept.
- `ece` / `ece_ts` — calibration gap, raw and after temperature scaling; ↓. `ece_ts` staying low ⇒ structural, not global.
- `auroc_{msp,energy,maha,featnorm}` — ID-vs-OOD separability of four different scores; ↑; reading them apart localises the gain.
- `oscr_msp` — open-set rate (admit *and* classify ID, reject OOD); ↑, the strictest.
- `fpr95_msp` — OOD leakage at 95 % ID recall; ↓, the operating-point complaint metric.
