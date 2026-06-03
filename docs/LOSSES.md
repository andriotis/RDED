# Student-loss terms in `validation/losses.py`

Mathematical reference for each loss term registered in `LOSS_REGISTRY`, so the
differences between them are easy to compare side by side. Each term returns a
batchmean-reduced scalar and is composed additively with the per-term weights in
the registry (e.g. `--w-rce`, `weights: {socce: 0.5}`).

> Formulas below use LaTeX math and render on GitHub (and any Markdown viewer
> with MathJax/KaTeX). In a plain terminal they appear as raw `$$` source.

## Notation (shared by all terms)

- $z^S$ — student logits on the (cutmix/mixup) **mixed** image, shape $[B, N]$
- $z^T$ — teacher logits on the **mixed** image, shape $[B, N]$
- $T$ — distillation temperature; $N$ — number of classes; $B$ — batch size; $i$ — batch index, $c,k$ — class index
- $p^S = \operatorname{softmax}(z^S / T)$ — student tempered probabilities
- $p^T = \operatorname{softmax}(z^T / T)$ — teacher tempered probabilities
- $\operatorname{KL}(a \,\|\, b) = \sum_c a_c\,(\log a_c - \log b_c)$, with batchmean = average over $i$

CutMix/MixUp metadata (in `MixInfo`): $\lambda$ (`lam`) is the mixing weight of
the host image, $y_i$ the host hard label, $y_{\text{partner}(i)}$ the partner
hard label (`rand_index`).

Feature-relational terms (`pkt`, `spkt`) additionally use the **penultimate-layer
features** on the mixed image: $h^S_i$ (student) and $h^T_i$ (teacher), shapes
$[B, D_S]$ and $[B, D_T]$. They operate on *pairs of samples in a batch* (not on
classes), need no matching feature dimensionality, and use **no temperature**.

---

## 1. `kl` — standard KD (stock RDED)

$$
L_{\mathrm{KL}} = \frac{1}{B}\sum_i \operatorname{KL}\!\left(p^T_i \,\|\, p^S_i\right)
= \frac{1}{B}\sum_i \sum_c p^T_{i,c}\left(\log p^T_{i,c} - \log p^S_{i,c}\right)
$$

Pull the student's tempered distribution toward the teacher's tempered
distribution on the same mixed image. This is the baseline every other term is
measured against.

---

## 2. `mxce` — mix-aware KL (target = convex mix of teacher preds)

$$
p^T_{\text{host}} = \operatorname{softmax}\!\big(z^T(\text{images})/T\big), \qquad
p^T_{\text{partner}} = \operatorname{softmax}\!\big(z^T(\text{images}[\text{rand}])/T\big)
$$

$$
p^T_{\text{target}} = \lambda\, p^T_{\text{host}} + (1-\lambda)\, p^T_{\text{partner}}, \qquad
L_{\mathrm{mxce}} = \frac{1}{B}\sum_i \operatorname{KL}\!\left(p^T_{\text{target},\,i} \,\|\, p^S_i\right)
$$

Identical to `kl` on the student side. **Only the target changes**: instead of
the teacher's prediction on the blended pixels, the target is the convex mix of
the teacher's predictions on the two *original* images. Requires the un-mixed
teacher pass (`mix_info.teacher_logits_unmixed`). With no mixing
($\lambda \ge 1$ or no `rand_index`) it collapses to `kl`.

---

## 3. `ockl` — one-cold KL (inverted teacher target, negated student) — DEPRECATED

$$
P^{\text{inv}}_c = \frac{1 - p^T_c}{N - 1}, \qquad
Q^S = \operatorname{softmax}(-z^S / T), \qquad
L_{\mathrm{OCKL}} = \frac{1}{B}\sum_i \operatorname{KL}\!\left(P^{\text{inv}}_i \,\|\, Q^S_i\right)
$$

Both distributions are "flipped". The target $P^{\text{inv}}$ spreads mass over
the classes the teacher *rejects*; the student is scored by its negated logits.
Imposes anti-class pressure on the **shape** of the logits (does not share
`kl`'s minimum).

---

## 4. `ockl_logitneg` — negate teacher *logits* (not probabilities)

$$
P^{T,\text{neg}} = \operatorname{softmax}(-z^T / T)\ \ \big(\text{not } (1-p^T)/(N-1)\big), \qquad
Q^S = \operatorname{softmax}(-z^S / T)
$$

$$
L = \frac{1}{B}\sum_i \operatorname{KL}\!\left(P^{T,\text{neg}}_i \,\|\, Q^S_i\right)
$$

Like `ockl` it negates both sides, but the target is the softmax of the
**negated teacher logits**. Because both sides are negated symmetrically, it
shares `kl`'s global minimum ($z^S = z^T + \text{const}$) — so it distills
"rejection knowledge" (emphasizing the teacher's most-rejected classes)
**without** distorting logit shape the way `ockl` does.

---

## 5. `gce` — Generalized Cross-Entropy (soft label), $q$ via `--gce-q`

$$
L_{\mathrm{GCE}} = \frac{1}{B}\sum_i \sum_c y_{i,c}\,\frac{1 - \left(p^S_{i,c}\right)^q}{q},
\qquad y = \operatorname{softmax}(z^T / T),\ \ q \in (0,1]\ (\text{default } 0.7)
$$

Noise-tolerant interpolation: $q \to 0$ recovers cross-entropy, $q = 1$ gives
MAE. The logit gradient is bounded by $\left(p^S\right)^q$, which damps
over-confident wrong targets (Zhang & Sabuncu 2018).

---

## 6. `rce` — Reverse Cross-Entropy (soft label), floor via `--sce-log-floor`

$$
L_{\mathrm{RCE}} = -\frac{1}{B}\sum_i \sum_c p^S_{i,c}\,\log \max\!\left(y_{i,c},\, e^{A}\right),
\qquad y = \operatorname{softmax}(z^T / T),\ \ A = \text{log floor (default } -4)
$$

Cross-entropy with the **roles swapped**: the student distribution weights the
log of the (clamped) teacher distribution. The clamp $e^{A}$ replaces
$\log(0)$. Compose with `kl` to build Symmetric CE: `--w-kl α --w-rce β`.

---

## 7. `scce` — Soft Complementary Cross-Entropy

$$
L_{\mathrm{sCCE}} = -\frac{1}{B}\sum_i \sum_c \left(1 - y_{i,c}\right)\,\log\!\left(1 - p^S_{i,c}\right),
\qquad y = \operatorname{softmax}(z^T / T)
$$

($p^S$ clamped below $1$ for stability.) Rejection-side mirror of CE: instead of
rewarding mass on the teacher's likely classes, it penalizes student mass on the
classes the teacher finds *unlikely* (weight $1 - y_c$). Estimates the same
V-information as KL from the rejection side.

---

## 8. `socce` — Soft One-Cold CE (anti-class structural regularizer) — DEPRECATED

$$
y^{\text{mix}}_c = \lambda\,\mathbf{1}[c = y_i] + (1-\lambda)\,\mathbf{1}[c = y_{\text{partner}(i)}], \qquad
\bar{y}_c = \frac{1 - y^{\text{mix}}_c}{N - 1}
$$

$$
L_{\mathrm{sOCCE}} = -\frac{1}{B}\sum_i \sum_c \bar{y}_{i,c}\,\log \operatorname{softmax}(-z^S_i)_c
$$

Two things make this different from everything above:
- **Target source = ground-truth cutmix labels**, not the teacher. $\bar{y}$ puts
  equal mass on every class *except* the true (mixed) ones.
- **No temperature** — operates on raw student logits $z^S$, through the negated
  softmax $\operatorname{softmax}(-z^S)$.

With no mixing it reduces to the one-hot one-cold encoding from the OCCE paper.

---

## 9. `pkt` — Probabilistic Knowledge Transfer (feature-space relational KD)

Cosine-similarity kernel, then row-normalized conditional probabilities, on
**penultimate features** (Passalis & Tefas, ECCV 2018):

$$
K(a,b) = \tfrac{1}{2}\!\left(\frac{a^\top b}{\|a\|_2\,\|b\|_2} + 1\right) \in [0,1], \qquad
P^T_{ij} = \frac{K(h^T_i, h^T_j)}{\sum_k K(h^T_i, h^T_k)}, \quad
Q^S_{ij} = \frac{K(h^S_i, h^S_j)}{\sum_k K(h^S_i, h^S_k)}
$$

$$
L_{\mathrm{PKT}} = \frac{1}{B}\sum_i \operatorname{KL}\!\left(P^T_i \,\|\, Q^S_i\right)
$$

Matches the **geometry** of the teacher's feature space (which samples are near
which) rather than the teacher's output — the signal `kl` throws away. Proven to
preserve the **Quadratic Mutual Information** between features and labels; needs no
temperature/bandwidth and no matching dimensionality. Gradients flow to the student
features only (teacher detached).

> **Diagonal convention.** The paper text normalizes over $k \ne j$ (excludes
> self-similarity); the authors' released code keeps the diagonal ($K(h_i,h_i)=1$).
> We follow the **released code** (it produced the paper's numbers) — see
> `_cosine_cond_prob` in `validation/losses.py`.

> **Mix-aware PKT (planned ablation, not yet implemented).** `pkt` matches teacher
> and student geometry both on the *mixed* image. The mix-aware variant would build
> the teacher relational target from the *un-mixed* features (analogous to how
> `mxce` mixes un-mixed teacher logits), answering whether relational structure is
> better matched pre- or post-mix. The `MixInfo` plumbing leaves room for a
> `teacher_feats_unmixed` field when this lands.

---

## 10. `spkt` — Supervised PKT (feature-space class structure)

Same student side as `pkt`, but the relational target is the **class structure**
instead of the teacher's geometry:

$$
A_{ij} = \mathbf{1}[\,y_i = y_j\,], \qquad
P_{ij} = \frac{A_{ij}}{\sum_k A_{ik}}, \qquad
L_{\mathrm{sPKT}} = \frac{1}{B}\sum_i \operatorname{KL}\!\left(P_i \,\|\, Q^S_i\right)
$$

Pulls same-class samples together and pushes different-class apart **directly in
feature space** — a *relational* neural-collapse target (within-class collapse +
between-class separation, cf. NC1/NC3). Uses the host hard label $y_i$ under cutmix
(like `socce`); the $\lambda$-weighted soft-affinity form is a further ablation. Note
this differs from the deprecated logit-space anti-class terms (`ockl`, `socce`): it
shapes the *features*, where neural collapse actually lives, rather than the logits.

---

## Diagnosis verdicts (cifar100/conv3, weight-aware; see `tools/diagnose_losses.py`)

Reproduce with `python tools/diagnose_losses.py`. A single `w=1.0` point is **not**
a verdict; a loss is deprecated only when its best delta across *every weight
tested* stays inside KL's seed-noise band.

| term | status | evidence |
|------|--------|----------|
| `kl` | **anchor** | 48.55 ± 0.21 (ipc=10) |
| `ockl` | **DEPRECATED** | 6-weight sweep ×3 ipc; best Δ ≈ 0 at w→0, monotone harm above w≈0.1 |
| `socce` | **DEPRECATED** | 8-weight sweep ×3 ipc ×2 arch; only the no-op 1e-5 is safe; NC2 worsens |
| `rce`, `gce`, `scce`, `ockl_logitneg`, `mxce` | **sweep-first** | only run at w=1.0 (gce also only q=0.7) → confirmatory grid in `scripts/sweeps/confirm_weights.yaml` before any verdict |
| `pkt`, `spkt` | **new, sweep-first** | feature-relational terms; weight-sweep against KL (low IPC first) before any keep/deprecate verdict |

Deprecated terms stay importable (negative results must stay reproducible) and are
excluded from the active sweeps.

---

## Comparison at a glance

| term            | target distribution                                            | target source        | student transform                       | uses $T$ |
|-----------------|----------------------------------------------------------------|----------------------|-----------------------------------------|----------|
| `kl`            | $p^T$                                                          | teacher (mixed img)  | $\operatorname{softmax}(z^S/T)$         | yes      |
| `mxce`          | $\lambda\,p^T_{\text{host}} + (1-\lambda)\,p^T_{\text{partner}}$ | teacher (orig imgs)  | $\operatorname{softmax}(z^S/T)$         | yes      |
| `ockl`          | $(1 - p^T)/(N-1)$                                              | teacher (inverted)   | $\operatorname{softmax}(-z^S/T)$        | yes      |
| `ockl_logitneg` | $\operatorname{softmax}(-z^T/T)$                              | teacher (neg logits) | $\operatorname{softmax}(-z^S/T)$        | yes      |
| `gce`           | $p^T$ (robust CE weights)                                     | teacher              | $\operatorname{softmax}(z^S/T)$, power $q$ | yes   |
| `rce`           | $p^T$ (clamped, log argument)                                 | teacher              | $\operatorname{softmax}(z^S/T)$ as weight | yes    |
| `scce`          | $1 - p^T$ (rejection weights)                                 | teacher (rejection)  | $\log(1 - \operatorname{softmax}(z^S/T))$ | yes    |
| `socce`         | $(1 - y^{\text{mix}})/(N-1)$                                  | hard cutmix labels   | $\operatorname{softmax}(-z^S)$          | **no**   |
| `pkt`           | $P^T$ (teacher-feature cosine cond-prob)                      | teacher (feature geometry) | cosine cond-prob of $h^S$         | **no**   |
| `spkt`          | $A/\!\sum A$ (same-class affinity, row-norm)                  | hard labels (feature space) | cosine cond-prob of $h^S$        | **no**   |

**Families:**
- *Standard / mix-aware KD*: `kl`, `mxce` — match the teacher's confident classes.
- *Anti-class / one-cold (logit space, DEPRECATED)*: `ockl`, `socce` — never beat KL (see verdicts); `ockl_logitneg` shares KL's minimum.
- *Robust soft-label CE*: `gce`, `rce`, `scce` — noise-tolerant or rejection-side reformulations of CE.
- *Feature-relational (PKT)*: `pkt`, `spkt` — match the **geometry** of the teacher's feature space (`pkt`) or the class structure (`spkt`) over batch sample-pairs, instead of per-sample logits. The only terms that read penultimate features.
