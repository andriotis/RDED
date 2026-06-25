# Trustworthiness experiments: metrics, selectors, and how to read the results

This is a self-contained guide to the experiments layered on top of RDED. It covers
**(1)** the commands that print the result tables, **(2)** every metric we report, and
**(3)** the selection methods and how they differ from stock RDED. Metrics and selectors are
given as **intuition + the precise mathematics** — no code walkthroughs.

> Formulas use LaTeX and render on GitHub (and any Markdown viewer with MathJax/KaTeX). In a
> plain terminal they appear as raw `$` source.

---

## 1. The big picture

**Stock RDED** is optimization-free distillation: for each class it crops real images, scores
the crops with a frozen **teacher**, keeps the most confident ones, stitches them into a tiny
**distilled set**, and **relabels** them with the teacher's soft outputs. A **student** is then
trained *only* on that set by knowledge distillation. RDED reports **accuracy and nothing else**.

This project adds two things:

- **A trustworthiness diagnostic.** Beyond accuracy we measure **calibration**, **open-set / OOD
  detection**, and the **neural-collapse geometry** of features. Two questions:
  - **H1 — synthesis geometry:** does the *teacher* see the **distilled set** as *more collapsed*
    (less within-class spread) than an equal slice of real data?
  - **H2 — student trustworthiness:** is the *student* trained on the distilled set
    **overconfident** and **open-set-fragile** relative to the real-data teacher?
- **A causal intervention.** We change **only the selection step** to deliberately put back the
  within-class variance stock RDED discards — at the **same images-per-class (IPC)** — and watch
  whether trust improves. Relabeling and training are untouched.

The thesis in one line: stock RDED **over-collapses** the distilled set, and that over-collapse
*causes* the student's overconfidence/open-set fragility; restoring within-class variance at
fixed IPC fixes it.

---

## 2. How to view the results

All four are read-only and run from the repo root.

| Command | What it shows | Reads |
|---|---|---|
| `python tools/show_diagnostics.py [--dataset cifar100] [--ipc 10] [--per-seed]` | The **stock baseline grid**, organized per cell `(dataset, arch, ipc)`: the H1 rows (distilled set vs equal real subset, under the teacher) and the H2 rows (student vs teacher reference). | `logs/results.jsonl`, `logs/diagnostics.jsonl` |
| `python tools/export_diagnostics.py [--out-dir exports]` | Regenerates the shareable report: `diagnostics_report.md` + CSVs, with **mean ± std** over seeds and a `h2_per_seed.csv`. | `logs/results.jsonl`, `logs/diagnostics.jsonl` |
| `python tools/analyze_select.py [files…]` | **The intervention table**: per cell, `stock` vs `random`/`stratified`/`covmatch` as mean ± std, with a dose-response arrow on each metric. Defaults to globbing `logs/results_select*.jsonl`. | `logs/results_select*.jsonl` |
| `python tools/diagnose_geometry.py --subset S --arch-name A --ipc N [--select-method M] [--syn-leaf syn_data_seedX]` | **H1 readout** for one cell/selector: the teacher-induced $\mathrm{NC1}$ of the distilled set vs an equal real subset (and the teacher's real-val reference). | distilled set on disk |

**Where the numbers live.** A training run logs one JSON row per seed to `logs/results.jsonl`
(the trust panel is the nested `diag` field, written only when the run is passed `--diagnostics`).
The selection experiment writes to `logs/results_select*.jsonl`. Teacher-geometry rows go to
`logs/diagnostics.jsonl`. The rendered report lives in `exports/`.

The single most useful table for "how is the intervention different from stock" is
`tools/analyze_select.py`.

---

## 3. Notation

- $\phi(x)\in\mathbb R^{D}$ — **penultimate features** (the input to the final linear layer).
- $z(x)\in\mathbb R^{K}$ — **logits**; $K$ classes; $\operatorname{softmax}(z)_k = e^{z_k}/\sum_j e^{z_j}$.
- A model is either the **teacher** (pretrained on all real data) or the **student** (trained only on the distilled set).
- **Open-set setup:** in-distribution (ID) inputs are the real validation set; OOD negatives come from a **far/near panel** — *far*: SVHN, MNIST, FashionMNIST (Places365 optional); *near*: CIFAR-10, DTD (STL-10 optional). Per-set metrics are logged under `diag.ood.<name>.*` (pass `--ood-sets`). An **OOD score** $s(x)\in\mathbb R$ follows the convention **higher $\Rightarrow$ more in-distribution**.

---

## 4. The metrics

### 4.1 Accuracy and calibration

**Top-1 accuracy** $\mathrm{acc}$ — fraction correctly classified. The only thing stock RDED reports.

**Expected Calibration Error (ECE).** *Intuition:* of all predictions made with ~$p$ confidence,
a calibrated model should be right ~$p$ of the time; ECE is the average confidence–accuracy gap.
Let $\mathrm{conf}(x)=\max_k\operatorname{softmax}(z)_k$. Partition the $N$ test points into
$M=15$ equal-width confidence bins $B_m=\big(\tfrac{m-1}{M},\tfrac{m}{M}\big]$ (the first bin is
also closed on the left), and

$$
\mathrm{ECE} \;=\; \sum_{m=1}^{M} \frac{|B_m|}{N}\,\Big|\,\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\,\Big|,
$$

where $\mathrm{acc}(B_m)$ and $\mathrm{conf}(B_m)$ are the mean accuracy and mean confidence in
the bin. **Lower is better**; $0$ = perfectly calibrated.

**Overconfidence gap.** $\;\mathrm{overconf\_gap} = \overline{\mathrm{conf}} - \mathrm{acc}$,
the mean confidence minus accuracy. **Positive $\Rightarrow$ overconfident.** When *all*
miscalibration is overconfidence (every bin has $\mathrm{conf}\ge\mathrm{acc}$), the absolute
values in ECE can be dropped and $\mathrm{ECE}=\sum_m \tfrac{|B_m|}{N}(\mathrm{conf}(B_m)-\mathrm{acc}(B_m)) = \overline{\mathrm{conf}}-\mathrm{acc}=\mathrm{overconf\_gap}$. Empirically the two are
nearly equal here, which is *why* we can say the error is single-signed (hence correctable).

**Temperature scaling (TS).** *Intuition:* a single knob $T$ that softens (or sharpens) all
predictions; it can fix confidence *level* but not the *ranking* of inputs. Fit $T$ by minimizing
negative log-likelihood on a **held-out real-train split** (images the student never saw):

$$
T^\* = \arg\min_{T>0}\; -\sum_{i} \log \operatorname{softmax}\!\big(z_i / T\big)_{y_i},
$$

then recompute calibration and open-set on $\operatorname{softmax}(z/T^\*)$, reported as
`ece_ts` and `auroc_msp_ts`. $T^\*>1$ means the model was overconfident. The **gap** between a
metric and its `_ts` version is the share of the problem that is *post-hoc fixable*; what TS
cannot fix (it is a monotone, per-sample-rank-preserving-on-MSP transform of confidence) is the
real open-set deficit.

### 4.2 Open-set operating metrics

Given ID scores $\{s_i^{+}\}_{i=1}^{n_+}$ and OOD scores $\{s_j^{-}\}_{j=1}^{n_-}$, threshold at
$\delta$: accept as ID if $s\ge\delta$. Then $\mathrm{TPR}(\delta)$ is the ID accept-rate and
$\mathrm{FPR}(\delta)$ the OOD accept-rate.

**AUROC.** *Intuition:* the probability the model scores a random ID input above a random OOD
input. Equivalently the area under the $\mathrm{TPR}$-vs-$\mathrm{FPR}$ curve, computed via the
Mann–Whitney statistic:

$$
\mathrm{AUROC} \;=\; \mathbb P\big(s^{+} > s^{-}\big) \;=\; \frac{R_{+} - \tfrac{1}{2}n_+(n_++1)}{n_+ \, n_-},
$$

where $R_{+}$ is the sum of ranks of the ID scores among all $n_++n_-$ scores (ties at half).
**Higher is better;** $0.5$ = chance. **$\mathrm{AUROC}<0.5$ is "inverted"** — the model is on
average *more* confident on OOD than on real inputs.

**FPR\@95.** The OOD accept-rate at the threshold that accepts 95% of ID:
$\;\mathrm{FPR95}=\mathrm{FPR}(\delta_{95})$ with $\delta_{95}$ the 5th percentile of ID scores.
**Lower is better.**

**OSCR (Open-Set Classification Rate).** *Intuition:* AUROC ignores whether accepted ID inputs
are *classified correctly*; OSCR rewards rejecting OOD **and** getting accepted ID right. With
the correct-classification rate

$$
\mathrm{CCR}(\delta) = \frac{\#\{x\in\mathrm{ID}: \hat y(x)=y \text{ and } s(x)\ge\delta\}}{n_+},
\qquad
\mathrm{FPR}(\delta) = \frac{\#\{x\in\mathrm{OOD}: s(x)\ge\delta\}}{n_-},
$$

OSCR is the area under the $\mathrm{CCR}$-vs-$\mathrm{FPR}$ curve as $\delta$ sweeps. **Higher is better.**

### 4.3 OOD scores $s(x)$

The metrics above depend on *which* score ranks inputs. We report several because they probe
different parts of the model.

**Maximum softmax probability (MSP)** — closest to a "stock" confidence:

$$
s_{\mathrm{MSP}}(x) = \max_k \operatorname{softmax}(z)_k .
$$

**Energy** — a temperature-controlled log-sum-exp of the logits (negative free energy); smoother
and less saturating than MSP:

$$
s_{\mathrm{energy}}(x) = T\,\log\sum_{k} e^{z_k/T}, \qquad T=1 .
$$

**Feature norm** — pure magnitude of the representation (no class structure):

$$
s_{\mathrm{featnorm}}(x) = \lVert \phi(x)\rVert_2 .
$$

**Mahalanobis (geometry-aware).** *Intuition:* score by how well $x$ fits the *class-conditional
Gaussian* of the features — directly reads the cluster geometry. Estimate class means $\mu_k$ and
a single **tied, shrunk** covariance on a held-out real-train split,

$$
\Sigma \;=\; (1-\alpha)\,\frac1N\sum_i (\tilde\phi_i-\mu_{y_i})(\tilde\phi_i-\mu_{y_i})^\top
\; +\; \alpha\,\frac{\operatorname{tr}\Sigma_0}{D}\,I ,
$$

and

$$
s_{\mathrm{maha}}(x) = -\min_{k}\; (\tilde\phi - \mu_k)^\top \Sigma^{-1} (\tilde\phi - \mu_k) .
$$

**Why $\tilde\phi$ is L2-normalized.** Expanding the quadratic,

$$
(\tilde\phi-\mu_k)^\top\Sigma^{-1}(\tilde\phi-\mu_k)
= \underbrace{\tilde\phi^\top\Sigma^{-1}\tilde\phi}_{\text{same for all }k}
\;-\;2\,\tilde\phi^\top\Sigma^{-1}\mu_k \;+\; \mu_k^\top\Sigma^{-1}\mu_k ,
$$

so $s_{\mathrm{maha}}= -\tilde\phi^\top\Sigma^{-1}\tilde\phi + \max_k(2\tilde\phi^\top\Sigma^{-1}\mu_k-\mu_k^\top\Sigma^{-1}\mu_k)$.
With raw features the first term grows like $\lVert\phi\rVert^2$, and since ID inputs have
*larger* feature norm than OOD, it makes ID look *farther* — **inverting** the score (we measured
AUROC $\approx 0.07$ on raw conv features). Normalizing $\tilde\phi=\phi/\lVert\phi\rVert$ removes
the norm confound, so Mahalanobis reads **directional** (cosine-space) class consistency — and it
becomes complementary to $s_{\mathrm{featnorm}}$, which captures the magnitude that normalization
discards. *(`max_logit` $=\max_k z_k$ is defined in the writeup but not currently computed.)*

### 4.4 Neural-collapse geometry (Papyan–Han–Donoho)

*Intuition:* well-trained features collapse to tight class clusters arranged as a maximally-spread,
symmetric simplex. These four numbers quantify how far a set of features is from that ideal. Let
$\mu_G=\tfrac1N\sum_i\phi_i$ be the global mean, $\mu_k$ the class-$k$ mean, and define the
within- and between-class scatter

$$
\Sigma_W = \frac1N\sum_i (\phi_i-\mu_{y_i})(\phi_i-\mu_{y_i})^\top,
\qquad
\Sigma_B = \frac1K\sum_k (\mu_k-\mu_G)(\mu_k-\mu_G)^\top .
$$

- **NC1 (within-class collapse)** — within-class scatter relative to between-class scatter:
  $$\mathrm{NC1} = \frac1K\operatorname{tr}\!\big(\Sigma_W\,\Sigma_B^{+}\big),$$
  with $\Sigma_B^{+}$ the pseudoinverse. **Lower = tighter, more separable clusters.** This is the
  central statistic of the project (see §5).
- **NC2 (equinorm)** — coefficient of variation of the centered class-mean norms
  $a_k=\lVert\mu_k-\mu_G\rVert$:
  $\;\mathrm{NC2}=\operatorname{std}(a)/\operatorname{mean}(a)$. **Lower = more equal norms.**
- **NC3 (equiangularity)** — mean deviation of pairwise class-mean cosines from the simplex-ETF
  ideal $-\tfrac{1}{K-1}$:
  $$\mathrm{NC3}=\operatorname{mean}_{k\ne k'}\Big|\,\cos(\mu_k-\mu_G,\;\mu_{k'}-\mu_G)+\tfrac{1}{K-1}\,\Big|.$$
  **Lower = closer to the ideal frame.**
- **NC4 (nearest-class-mean agreement)** — fraction of inputs where the nearest class mean
  $\arg\min_k\lVert\phi-\mu_k\rVert$ equals the classifier's prediction.

> **Which model, which split — read this.** The *formula* is identical everywhere, but the
> *inputs* differ and the numbers move oppositely:
> - **H1 NC** = the **teacher's** features on a **set** (the distilled set, or an equal real subset). This measures the synthesis geometry. The distilled set's $\mathrm{NC1}$ comes out **below** the real subset's — it is *over-collapsed*.
> - **H2 NC** = the **student's** features on the **real validation set**. This measures the trained model. At low IPC it is *high* (under-collapsed / poorly separated on held-out data).

---

## 5. The selectors — and how they differ from stock RDED

This is the one part of the pipeline the intervention changes. For one class, let the candidate
crops be $\{x_j\}$ with teacher logits $z_j$ and **penultimate features** $\phi_j$. Define the
**realism score** as the teacher cross-entropy

$$
\ell_j = -\log \operatorname{softmax}(z_j)_{[y]} \quad(\text{low }\ell_j \Leftrightarrow \text{high teacher confidence}).
$$

Each method chooses a size-$n$ index set $S$ (with $n=\mathrm{IPC}\cdot\text{factor}^2$). (When
crops are stitched, an inner step first keeps each source image's most-confident crop,
$\arg\min$ over its $m$ random crops, before selection.)

### 5.1 Stock RDED — top-confidence

$$
S_{\mathrm{stock}} = \big\{ \text{the } n \text{ indices with the smallest } \ell_j \big\}.
$$

*Intuition / why it over-collapses.* The teacher is most confident exactly where a crop sits
**near its class mean** in feature space (small $\lVert\phi_j-\mu_y\rVert$). Selecting the
lowest-$\ell$ crops therefore systematically keeps low-deviation, prototypical points and
discards the tails — so the selected within-class scatter $\Sigma_W^{(S)}$ is biased small and
$\mathrm{NC1}_{\text{distilled}}\ll \mathrm{NC1}_{\text{real}}$ (we measure a ratio $\approx 0.5$).
That missing within-class spread is the quantity the interventions restore.

### 5.2 The realism floor (shared guard)

To avoid trading confidence for *garbage*, the variance-seeking methods select **within an
eligible pool** of the most-realistic candidates:

$$
P = \big\{ \text{the } \lceil r\, n\rceil \text{ indices with smallest } \ell_j \big\},
\qquad r = \texttt{--select-realism-floor}\ (\text{default } 3).
$$

All methods below choose $S\subseteq P$, $|S|=n$.

### 5.3 `random` — the weak-lever control

$$
S \sim \mathrm{Uniform}\Big(\{\,S\subseteq P : |S|=n\,\}\Big).
$$

*Intuition.* Inject variance without being clever about it. It isolates the effect of "more
within-class spread" from "*smartly chosen* spread", and anchors the dose-response.

### 5.4 `stratified` — span the modes

Cluster the pool's features with $k$-means ($k=\texttt{--select-k}$),

$$
\min_{C_1,\dots,C_k}\;\sum_{c=1}^{k}\sum_{j\in C_c}\big\lVert \phi_j-\bar\phi_c\big\rVert^2 ,
$$

then fill $S$ **round-robin across clusters**, taking the lowest-$\ell$ (most realistic) member of
each in turn. *Intuition.* Guarantees the selection covers the distinct visual modes of the class
rather than piling onto the single dominant one.

### 5.5 `covmatch` — maximize feature volume (match real covariance)

Select the subset whose features are most **spread out**, via greedy log-determinant (DPP-MAP)
maximization on the cosine kernel $K_{ij}=\langle\tilde\phi_i,\tilde\phi_j\rangle$ (with
$\tilde\phi$ unit-normalized, $\epsilon$ a small ridge):

$$
S = \arg\max_{S\subseteq P,\,|S|=n}\; \log\det\!\big(K_S + \epsilon I\big),
$$

solved by the standard greedy that, at each step, adds the candidate of **maximum conditional
variance** given those already chosen. *Intuition.* $\log\det K_S$ is (twice) the log-volume of
the parallelotope spanned by the selected feature vectors — maximizing it pushes them to be
mutually dissimilar, which is exactly **increasing $\Sigma_W^{(S)}$**, i.e. driving
$\mathrm{NC1}_{\text{distilled}}$ up toward the real value. Equivalently: *compose the per-class
set to match the real within-class covariance.*

### 5.6 `qddpp` — one knob from `covmatch` to `stock` (and a near-OOD quality)

`stock` and `covmatch` are the two extremes of a single objective. Put a **quality** weight
$q_j>0$ on each candidate and maximize the log-determinant of the **quality-weighted** kernel — an
**L-ensemble DPP** (Kulesza & Taskar 2012) — with the same greedy as §5.5:

$$
L_{ij} = q_i\,\langle\tilde\phi_i,\tilde\phi_j\rangle\,q_j,
\qquad
S = \arg\max_{S\subseteq P,\,|S|=n}\;\log\det\!\big(L_S + \epsilon I\big).
$$

Because $L_S = \operatorname{diag}(q_S)\,K_S\,\operatorname{diag}(q_S)$, the objective **separates**:

$$
\log\det L_S \;=\; 2\!\sum_{j\in S}\log q_j \;+\; \log\det K_S ,
$$

a **quality** term (favor high-$q$ crops) plus the **diversity / feature-volume** term `covmatch`
maximizes (§5.5). Set the quality from a standardized per-crop score and a temperature $\beta$:

$$
q_j = \exp(\beta\, r_j),\qquad
r_j = -\,\frac{\ell_j - \bar\ell}{\operatorname{std}(\ell)}\quad(\text{confidence}).
$$

- $\beta\to 0$: $q_j\to 1$, $L\to K$ $\Rightarrow$ **exactly `covmatch`** (pure volume).
- $\beta\to\infty$: the quality term dominates $\Rightarrow$ the $n$ highest-$q$ (lowest-$\ell$) crops
  $\Rightarrow$ **exactly `stock`**.

So $\beta$ is a **single continuous dial** between the two endpoints, and the hard realism floor of
§5.2 becomes a *soft* quality: intermediate $\beta$ can be tuned to land
$\mathrm{NC1}_{\text{distilled}}/\mathrm{NC1}_{\text{real}}\approx 1$ — the "match, don't overshoot"
target of §5.7 — rather than covmatch's overshoot. Selected with `--select-method qddpp --select-beta`.

**Boundary quality (the near-OOD lever).** Replace the quality score with the teacher **margin**
$m_j = z_{(1)} - z_{(2)}$ (top-1 minus top-2 logit), using $r_j=-\operatorname{standardize}(m_j)$ so
*low-margin* (decision-boundary) crops are up-weighted (`--select-quality margin`). Where
confidence-quality $+$ volume restores within-class *spread* (a **far**-OOD lever — over-dispersion
blurs near-OOD), boundary-quality instead seeds the selection with crops near the class margin, the
geometry **near**-OOD detection reads — targeting the one shift type variance restoration misses.

### 5.7 What changes vs. stock, and the one nuance

Only the index rule $S$ above changes; the teacher relabeling and the KD training are identical to
stock (selected with `--select-method {stock,random,stratified,covmatch,momentmatch,qddpp}`).
**"Match, don't maximize":** `covmatch` maximizes volume, so it can **overshoot** real variance
($\mathrm{NC1}_{\text{distilled}}/\mathrm{NC1}_{\text{real}}>1$); empirically the best trust sits
near a ratio of $1$, so $r$, $k$, and `qddpp`'s $\beta$ are knobs controlling *how much* variance is
injected, and the target is to *match* real data, not to maximize spread.

---

## 6. Why it matters, and what the experiments show

The diagnostic establishes the correlation (over-collapsed sets ↔ untrustworthy students), but
correlation alone is confounded by IPC. The selectors give the **causal test**: at **fixed IPC**,
the chain $\texttt{stock}<\texttt{random}<\texttt{covmatch}$ is a *dose* of within-class variance
(it raises $\mathrm{NC1}_{\text{distilled}}$ monotonically toward real). If trust tracks that dose,
over-collapse is causal.

It does. Across both architectures (Conv, ResNet-18) and both datasets, at IPC 10/50, the trust
metrics improve monotonically with the dose **at no accuracy cost** — ECE down, AUROC/OSCR up,
FPR\@95 down — and the **geometry-aware Mahalanobis** moves the most, exactly as the mechanism
predicts. Two honest boundaries: **IPC 1** cannot benefit (a single image per class has no
within-class variance to preserve), and **calibration** is largely fixable post-hoc by temperature
scaling — so the genuine contribution is the **open-set / geometry** improvement, which TS cannot
provide. Run `python tools/analyze_select.py` to see the full dose-response table.

The dose `stock`<`random`<`covmatch` is three points on a continuum: `qddpp` (§5.6) makes the
quality↔diversity trade-off a **single continuous knob** $\beta$ with `stock` ($\beta\to\infty$) and
`covmatch` ($\beta=0$) as its endpoints, so the dose-response becomes a tunable frontier rather than
a fixed ladder. The far/near split of the gain — variance restoration helps **far**-OOD (SVHN,
MNIST, FashionMNIST) but not **near**-OOD (CIFAR-10, DTD) — motivates `qddpp`'s margin quality as the
complementary near-OOD lever.

---

## 7. Quickstart — reproduce one comparison

```bash
# 1. View the stock baseline (H1 + H2) for the CIFAR-100 / ipc-10 cells
python tools/show_diagnostics.py --dataset cifar100 --ipc 10

# 2. Run stock vs covmatch on one cell, one seed (pick a free GPU)
CUDA_VISIBLE_DEVICES=0 METHODS="stock covmatch" CELLS="cifar100:conv3" \
  IPCS="10" SEEDS="42" bash scripts/run_select_variants.sh

# 3. Compare them (the intervention dose-response table)
python tools/analyze_select.py

# 4. Confirm the H1 lever moved: nc1_distilled vs the real subset under the teacher
python tools/diagnose_geometry.py --subset cifar100 --arch-name conv3 \
  --ipc 10 --select-method covmatch --syn-leaf syn_data_seed42
```

`scripts/run_select_variants.sh` is configured by environment variables
(`METHODS`, `CELLS`, `IPCS`, `SEEDS`, `RESULTS_FILE`, `--select-*` knobs); a student run records
the full trust panel only when it is launched with `--diagnostics` (the driver does this).
