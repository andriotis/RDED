# `facloc` explained — from intuition to math

> A teaching walkthrough in the same spirit as [`RELMATCH_EXPLAINED.md`](RELMATCH_EXPLAINED.md) and
> [`RELDIST_EXPLAINED.md`](RELDIST_EXPLAINED.md). We reuse the same running example (three classes:
> 🐱 cat, 🐶 dog, 🚗 car) and carry it from the idea to the math. This document is **math + intuition**;
> the code-level walk-through lives in [`SELECTORS.md`](SELECTORS.md).

---

## 0. TL;DR (read this first)

`relmatch` (Formalization 1) and `reldist` (Formalization 2) ask *"does the selected set's statistic
match the pool's statistic?"* — the mean row `R[i,:]`, or the whole per-coordinate distribution. They
score the set by a **distance to a target**.

`facloc` (Formalization 3) asks a different question: *"does every kind of crop in the class have a
chosen representative nearby?"* It scores the set by **how well it covers the pool**. The value of
adding one more crop is its **marginal contribution to that coverage given what is already selected** —
so a crop is worth keeping only for what it *adds*, never for any standalone score. That is the precise
sense in which this is the formalization that *earns the word "complementary"* (thought.md,
Formalization 3): complementarity here is a property of the **set**, defined by construction.

---

## 1. The object: facility location

Let each crop `i` have an embedding `e_i` — here the teacher **soft label** `p_i` (the relational
manifold, `--facloc-space softlabel`, the default), or the penultimate **feature** `z_i`
(`--facloc-space feature`). Fix a class and let its full per-class pool be `U = {1, …, M}`. For a
selected subset `S ⊆ U` define the **facility-location value**

$$
F(S) \;=\; \sum_{i \in U} \; \max_{s \in S} \; \mathrm{sim}(e_i, e_s),
\qquad \mathrm{sim}(a,b) = \cos(a,b) = \frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}.
$$

Read it literally: for **every** pool point `i`, find its **best** representative in `S` (the most
similar selected crop) and add up those best-similarities. `F(S)` is large exactly when *no* pool
point is left far from all chosen crops — i.e. when `S` **covers** the class. The cosine kernel is the
same L2-normalized convention `covmatch` uses, so "far" means *different in direction* (a dog-ish cat
vs. a textbook cat), with the norm confound removed.

This is the classic **facility-location / k-medoid coverage** objective. We *maximize* it under the
fixed budget `|S| = n` (the IPC budget `n = ipc · factor²`).

---

## 2. Why coverage ≠ the other selectors

Three statistics, three different geometries — it helps to see exactly how `facloc` differs:

| selector | object | what it rewards |
|---|---|---|
| `covmatch` | `log det K_S` (DPP / log-determinant) | **volume / repulsion** — picks mutually *dissimilar* crops; an outlier pair can win even if it leaves the bulk of the class unrepresented |
| `relmatch` | `‖μ_S − R[i,:]‖²` | the selected **mean** equals the pool mean — silent about the shape |
| `reldist` | per-coordinate `W₂²` | the selected **distribution** equals the pool's, coordinate by coordinate |
| **`facloc`** | `F(S) = Σ_i max_{s∈S} cos(e_i,e_s)` | **coverage** — every kind of crop has a nearby representative; rewards *representativeness*, penalizes leaving a sub-mode uncovered |

The sharp contrast is with `covmatch`. Log-determinant is a *repulsion* objective: it can happily
spend two of its `n` picks on a pair of mutually-distant rare crops, because that pair spans a large
volume — even if a populous sub-mode of the class then has **no** representative at all. Facility
location cannot do that: leaving a populous region uncovered costs a large `Σ_i max…` deficit, so it
spreads picks toward the **modes** of the class, not its extremes. *Coverage, not volume.*

---

## 3. Submodularity and the greedy guarantee (the reason this is principled, not a heuristic)

`F` has two structural properties, both elementary to check from the definition:

- **Monotone:** adding a crop can only raise each inner `max`, so `F(S ∪ {j}) ≥ F(S)`.
- **Submodular (diminishing returns):** the marginal gain of a crop shrinks as `S` grows. Define

$$
\Delta(j \mid S) \;=\; F(S \cup \{j\}) - F(S)
\;=\; \sum_{i \in U} \max\!\Big(0,\; \cos(e_i,e_j) - \max_{s \in S}\cos(e_i,e_s)\Big).
$$

Each pool point `i` contributes only the amount by which `j` **improves** its current best coverage,
clamped at 0. As `S` grows the running `max_{s∈S}` only rises, so every term of `Δ(j∣S)` can only
fall: `Δ(j∣S) ≥ Δ(j∣T)` whenever `S ⊆ T`. That is submodularity verbatim.

For a monotone submodular `F`, the **greedy** algorithm — repeatedly add the crop of maximum marginal
gain — is guaranteed to reach at least `(1 − 1/e) ≈ 0.632` of the optimal `F` over *all* size-`n`
subsets (Nemhauser–Wolsey–Fisher, 1978). Exact maximization is NP-hard; greedy is the textbook,
near-optimal, fully deterministic answer. So `facloc`'s "pick the most-complementary crop next" is not
an ad-hoc heuristic — it is the optimal-up-to-`(1−1/e)` solver for the coverage objective.

---

## 4. The algorithm (greedy, with the running coverage frontier)

The only state greedy needs is the **coverage frontier** `c_i = max_{s∈S} cos(e_i, e_s)` — pool point
`i`'s best similarity to anything chosen so far. Start it at `−1` (the minimum a cosine can take: an
empty set covers nothing). This init matters: on the first round `gain_j = Σ_i (cos(e_i,e_j) − (−1)) =
Σ_i (cos(e_i,e_j)+1)` is monotone in the total similarity `F({j}) = Σ_i cos(e_i,e_j)`, so the first
pick is the true coverage **medoid** — not whatever happens to sit at index 0. Then each round is one
vectorized scan:

1. `gain_j = Σ_i max(0, S_{ij} − c_i)` for every still-available candidate `j`, where
   `S_{ij} = cos(e_i, e_j)`;
2. pick `j★ = argmax_j gain_j`, add it to `S`;
3. absorb it into the frontier: `c_i ← max(c_i, S_{ij★})`.

Repeat `n` times. With the precomputed `M×P` similarity matrix this is `O(n·M·P)`; on a per-class pool
of a few hundred crops it is instant. It is deterministic (no RNG — `gen` is unused), so it slots into
the same deterministic-and-cached run regime as every other selector.

### Worked micro-example

Six cat crops; cosine of their soft labels gives two tight clusters — **A** = {a₁,a₂,a₃} (textbook
cats) and **B** = {b₁,b₂,b₃} (dog-ish cats), with cross-cluster similarity ≈ 0 and within-cluster
≈ 1. Budget `n = 2`.

- **Round 1.** Every crop covers its own cluster (≈ 3 points rising from `−1` to ≈ 1) and barely the
  other, so by symmetry all six have an equal gain; greedy takes the first by tie-break — say `a₁`.
  Frontier: cluster A covered, cluster B still at `−1`.
- **Round 2.** Another A crop adds almost nothing (A is already covered → `gain ≈ 0`); **any B crop**
  adds ≈ 3 (it rescues the three uncovered dog-ish cats). Greedy takes a B crop.

Result: one representative from **each** cluster — coverage straddles both sub-modes. A volume
objective (`covmatch`) might instead pick the two *most extreme* crops; a confidence-greedy `stock`
would pick two textbook A cats and leave the dog-ish tail with **no** representative. This straddle is
exactly what `tests/test_select.py::test_select_facloc_straddles_two_clusters` pins down.

---

## 5. Why it matters for trustworthiness

The dog-ish-cat sub-mode is precisely the **inter-class boundary** structure — the off-diagonal mass
of the class-relation matrix `R`, the geometry that calibration, OOD detection, and adversarial
robustness read, and the geometry confidence-greedy `stock` flattens (it keeps only textbook cats).
By construction `facloc` refuses to leave that sub-mode uncovered, so its distilled set should
**preserve the relational manifold**. Concretely, in soft-label space we expect it to lower the
relation-matrix divergences `rel_frob_offdiag` / `rel_row_cos_offdiag` versus `stock` at matched IPC —
the same `class_relation_matrix` / `relation_divergence` diagnostics
([`validation/diagnostics.py`](../validation/diagnostics.py)) that `relmatch`/`reldist` are measured
by. The `--facloc-space feature` variant runs the same coverage in penultimate-feature space, giving a
clean **coverage-vs-volume** ablation against `covmatch` in the space `covmatch` already lives in.

---

## 6. One line each, vs. its neighbors

- `stock`: keep the `n` most confident crops → all textbook cats, boundary tail discarded.
- `covmatch`: maximize feature **volume** → mutually distant crops, can ignore the bulk.
- `relmatch` / `reldist`: match the pool's **mean** / **distribution** statistic.
- **`facloc`**: maximize **coverage** of the pool — every sub-mode gets a nearby representative; value
  of a crop = its *marginal* coverage gain (set-level complementarity, `(1−1/e)`-optimal greedy).
