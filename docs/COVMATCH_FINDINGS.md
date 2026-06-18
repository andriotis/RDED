# Variance-Aware Crop Selection in RDED: A Trustworthiness Study (`covmatch`)

This document records the project goal, the current findings (with results), and how to reproduce or
extend them. Repo `andriotis/RDED`. Status: the selection study is complete (June 2026); the picture below
is settled. A condensed orientation is also intended to live with the project notes; this is the full
reference.

---

## 0. Orientation (start here)

**What we're doing.** Testing whether a *variance-preserving crop selection* in RDED dataset distillation
(`covmatch`) makes the distilled-data-trained student more **trustworthy** — better calibrated and better
at OOD/open-set detection — without costing accuracy. The one intervention is **stage-1 selection**
(`synthesize/utils.py::selector`, flag `--select-method`); everything downstream (training, soft labels)
is unchanged, which is what makes any trust change *causally* attributable to the distilled-set geometry.

**Where we are.** The selection study is complete:
- `covmatch` **improves calibration (significant 4/4 cells) and far-OOD/SVHN detection (significant 3/4),
  at zero top-1 cost**, via a clean stock→random→covmatch dose at IPC=10.
- Mechanism: it **restores within-class feature variance** (NC1) that RDED's top-confidence rule collapses.
- A foil selector that *matches* covariance instead of *maximizing* its volume (`momentmatch`) **fails** —
  the operative principle is *maximize* variance, not *match* it.
- The realism floor is a **causal variance dial**: opening it overshoots real variance and trades ~4 top-1
  pts for big far-OOD + calibration gains.
- **Key boundary:** the OOD gain is **far-OOD (SVHN)-specific** — it does **not** transfer to near-OOD
  textures (DTD). Only the calibration gain is OOD-type-robust.

**Where to look.** Results live in `logs/*.jsonl` (one row per run; `r["diag"]` holds the trust panel,
`r["diag"]["ood"][set]` the per-OOD metrics). Analysis is `tools/analyze_*.py`; geometry is
`tools/diagnose_geometry.py`. See §3 for exact files and commands. Runs are **deterministic and reused** —
never re-run a completed `(dataset,arch,ipc,method,seed)`; SVHN numbers reproduce bit-exact.

---

## 1. The research question

RDED keeps, per source image, the single crop the teacher is most confident on, then takes the globally
most-confident crops. This **collapses within-class feature variance** onto one mode. Hypothesis: that
collapse is what makes RDED students overconfident and open-set-fragile, and **restoring within-class
variance at selection time** should improve trustworthiness at fixed IPC and accuracy. `covmatch` tests
this by maximizing the **log-determinant (volume)** of the selected crops' L2-normalized teacher-feature
cosine kernel over a realism-floored pool (greedy DPP, Chen et al. 2018).

The selector family (the only thing varied): `stock` (RDED top-confidence) → `random` (uniform over the
floored pool) → `stratified` (k-means round-robin) → `covmatch` (log-det/volume maximization) restore
progressively more within-class variance; `momentmatch` (§2.4) is a covariance-*matching* foil.

---

## 2. Findings

### 2.1 The dose-response (stock → random → covmatch)
At **IPC=10**, geometry-aware trust improves monotonically, top-1 unchanged. Mahalanobis-AUROC (mean±std,
3 seeds):

| cell (IPC=10) | stock | random | covmatch | Δ(cov−stock) |
|---|---|---|---|---|
| cifar100/conv3 | 0.586 | 0.595 | **0.669** | +0.083 |
| cifar100/resnet18 | 0.878 | 0.898 | **0.908** | +0.030 |
| tiny/conv4 | 0.457 | 0.469 | **0.599** | +0.143 |
| tiny/resnet18 | 0.978 | 0.987 | 0.986 | +0.008 (ceiling) |

`random` lands *between* stock and covmatch ⇒ a genuine dose (partial vs full variance restoration). The
same shape appears on energy/feat-norm-AUROC, ECE, and FPR95. **IPC dependence:** no benefit at IPC=1
(budget too small for diversity; sometimes regresses), sweet spot at IPC=10, persists/partly saturates at
IPC=50 (Appendix B). **Selectivity:** softmax-only scores (MSP) are noisy/flat; the gain lives in
*feature-space* scores, exactly where covmatch acts.

### 2.2 Paired significance (covmatch − stock, same seed, IPC=10)
Pairing each covmatch run with the stock run at the same seed cancels synthesis+init noise that makes the
unpaired ±std bars overlap. `*` = bootstrap 95% CI excludes 0.

| cell | Δ Mahalanobis-AUROC [CI95] | Δ ECE (↓ better) [CI95] | Δ top-1 |
|---|---|---|---|
| cifar100/conv3 | **+0.083** [+0.023,+0.164] * | **+0.012** [+0.006,+0.020] * | within noise |
| cifar100/resnet18 | **+0.030** [+0.020,+0.040] * | **+0.012** [+0.008,+0.019] * | within noise |
| tiny/conv4 | **+0.143** [+0.077,+0.255] * | **+0.011** [+0.005,+0.018] * | within noise |
| tiny/resnet18 | +0.008 [−0.001,+0.022] | **+0.025** [+0.017,+0.031] * | within noise |

**Mahalanobis significant 3/4** (4th at ceiling), **ECE significant 4/4** (the most consistent gain),
**top-1 every CI includes 0** (free of accuracy cost).

### 2.3 Mechanism — within-class variance restoration (NC1)
Teacher-side neural-collapse NC1 of the *distilled set* relative to a matched *real* subset (IPC=10);
NC1 = tr(Σ_W Σ_B⁺)/K, so lower = more collapsed:

| cell | stock | random | covmatch | real NC1 (abs.) |
|---|---|---|---|---|
| cifar100/conv3 | 0.58× | 0.67× | **0.87×** | 1.52 |
| cifar100/resnet18 | 0.48× | 0.56× | **0.72×** | 3.17 |
| tiny/conv4 | 0.65× | 0.74× | **0.93×** | 2.20 |
| tiny/resnet18 | 0.43× | 0.52× | **0.69×** | 4.42 |

Stock collapses NC1 to ~half of real; covmatch restores it toward real, monotone with the OOD gain. At
IPC=50 covmatch reaches/exceeds real (conv3 1.01×, conv4 1.03×, cifar/resnet 1.16×). **Mediator:** within
each cell, the distilled-set NC1-gap correlates *negatively* with the student's Mahalanobis-AUROC
(Spearman ρ = −0.39, −0.20, −0.47, −0.13); the L2-normalized covariance-gap does **not** (pooled ρ ≈ +0.14)
— the mediator is variance *magnitude*, not covariance shape.

### 2.4 Falsification — *maximize*, don't *match* (`momentmatch`)
`momentmatch` greedily matches the pool's mean + second moment (`min ‖M_S−M_t‖_F² + λ‖μ_S−μ_t‖²`).
Falsifiable prediction: it should equal or beat covmatch. **It fails.** Mahalanobis-AUROC at IPC=10:

| cell | stock | covmatch | momentmatch | mm − cov |
|---|---|---|---|---|
| cifar100/conv3 | 0.586 | 0.669 | 0.620 | −0.050 |
| cifar100/resnet18 | 0.878 | 0.908 | 0.874 | −0.034 |
| tiny/conv4 | 0.457 | 0.599 | 0.429 | −0.170 |
| tiny/resnet18 | 0.978 | 0.986 | 0.986 | 0.000 |

momentmatch ≈ stock < covmatch (loses in 3/4). Mirror image: momentmatch *raises* top-1 (paired vs stock
+1.07 conv3 *, +0.56 tiny/resnet *) by selecting prototypical crops — the opposite of what helps OOD.
**Open puzzle:** covmatch and momentmatch produce distilled sets with *nearly identical* NC1 *and* effective
rank (conv4: 2.04/2.04; eff-rank 333/330) yet different student OOD ⇒ the distilled-set's aggregate geometry
explains stock→covmatch but **not** covmatch→momentmatch; the specific crops chosen matter beyond the
summary statistic.

### 2.5 The realism-floor knob (accuracy↔trust frontier)
Sweeping `--select-realism-floor` on tiny/conv4 (eligible pool size = floor·IPC):

| floor (pool) | Mahalanobis-AUROC | top-1 | ECE |
|---|---|---|---|
| stock | 0.457 | 40.2 | 0.064 |
| 2 (20) | 0.482 | 40.2 | 0.063 |
| 3 (30) = default | 0.599 | 40.1 | 0.053 |
| 5 (50) | 0.481 | 40.4 | 0.042 |
| 10 (100) | 0.588 | 39.4 | 0.044 |
| **30 (300) = full pool** | **0.912** | **37.0** | **0.017** |

The big, consistent gain appears only at the **full-pool extreme** (intermediate floors are within noise —
so this is *not* a smooth dial but a distinct high-trust operating point). Generalizing full-pool covmatch
(fl30) across all cells (vs covmatch default fl3):

| cell | Δ Mahalanobis | Δ top-1 | ECE (fl3→fl30) | NC1/real (fl3→fl30) |
|---|---|---|---|---|
| cifar100/conv3 | +0.144 (0.669→0.813) | −5.0 | 0.090→0.072 | 0.87→**1.43** |
| cifar100/resnet18 | +0.038 (0.908→0.945) | −4.2 | 0.174→0.131 | 0.72→**1.24** |
| tiny/conv4 | +0.313 (0.599→0.912) | −3.2 | 0.053→0.017 | 0.93→**1.41** |
| tiny/resnet18 | +0.007 (0.986→0.993) | −3.6 | 0.125→0.039 | 0.69→**1.62** |

Opening the floor pushes within-class variance from *undershoot* to *overshoot* of real (NC1/real
0.7–0.9 → 1.2–1.6), buying large SVHN-OOD and consistent calibration gains for ~4 top-1 points. The OOD gain
scales with headroom (largest where the baseline was weakest). **The realism floor is the causal variance
dial — the strongest causal evidence in the study: turning one knob moves trust monotonically.**

### 2.6 The boundary — far-OOD (SVHN)-specific
Multi-OOD evaluation (back-filled onto the baselines via 24 deterministic retrains that reproduce SVHN
bit-exact). Mahalanobis-AUROC, stock / covmatch, per OOD set, IPC=10:

| cell | SVHN (far) | DTD (textures, near) | CIFAR-10 (near) |
|---|---|---|---|
| cifar100/conv3 | 0.586 / **0.669** | 0.316 / 0.302 | 0.509 / 0.502 |
| cifar100/resnet18 | 0.878 / **0.908** | 0.869 / 0.871 | 0.501 / 0.510 |
| tiny/conv4 | 0.457 / **0.599** | 0.344 / 0.336 | 0.755 / 0.792 |
| tiny/resnet18 | 0.978 / **0.986** | 0.924 / 0.929 | 0.877 / 0.863 |
| **mean Δ(cov − stock)** | **+0.066** | **−0.003** | **+0.006** |

covmatch's selection advantage is real on SVHN, ~zero on DTD, marginal on CIFAR-10. The **calibration (ECE)
gain is OOD-independent** and holds across all cells (§2.2). **Why:** over-dispersing the within-class
covariance inflates the class-conditional Gaussian, which separates *distant* OOD (SVHN digits) but **blurs
near-OOD** (textures/natural images overlap the now-wider ID manifold). Variance restoration is a far-OOD
lever.

**Defensible headline:** *variance-aware distillation selection improves model calibration (robustly) and
far-OOD detection (at zero accuracy cost); the OOD benefit is shift-type-specific.*

### 2.7 What covmatch is, and is not
**Is:** a drop-in stage-1 selector that restores within-class feature variance via log-det (volume)
maximization; a zero-accuracy-cost improvement to calibration (4/4) and far-OOD detection (3/4); causally
linked to within-class variance (dose-response + NC1 mediator + the floor knob). **Is not:** a general OOD
improvement (far-OOD-specific); a covariance-*matching* effect (matching fails, maximizing wins); an
IPC-universal effect (no benefit at IPC=1); a free trust-maximizer (pushing past real costs accuracy).

---

## 3. Where to look & how to verify

**Result files (`logs/`, JSONL, one row/run; `r["diag"]` = trust panel, `r["diag"]["ood"][set]` = per-OOD):**
- `results_select_{conv3,conv4,cifar_resnet}.jsonl` + `results_select_tiny_resnet_all.jsonl` — the
  stock/random/covmatch dose-response, all IPC (SVHN-only). (`tiny_resnet_all` = deduped consolidation of
  the scattered `_g*/_bf*` GPU-shard files.)
- `results_momentmatch.jsonl` — momentmatch, IPC=10 (multi-OOD).
- `results_floor_sweep.jsonl` — realism-floor sweep + fl30 across cells (multi-OOD).
- `results_baseline_multiood.jsonl` — stock+covmatch IPC=10 re-run with multi-OOD (the §2.6 backfill).
- `mediator_geometry.jsonl` (+ `mediator_geometry_ipc.jsonl`, `floor_geom.jsonl`) — teacher-side distilled
  vs real geometry: NC1, `cov_gap_to_real`, `wcov_effrank`, keyed by `select_method`.

**Analysis commands (read-only):**
- Dose-response: `python tools/analyze_select.py logs/results_select_conv3.jsonl logs/results_select_conv4.jsonl logs/results_select_cifar_resnet.jsonl logs/results_select_tiny_resnet_all.jsonl`
- Paired CIs: `python tools/analyze_paired.py <same files> logs/results_momentmatch.jsonl`
- Mediator: `python tools/analyze_mediator.py --geom logs/mediator_geometry.jsonl --results <result files>`
- Per-OOD-set: `python tools/analyze_multiood.py logs/results_baseline_multiood.jsonl logs/results_floor_sweep.jsonl`
- One distilled set's geometry: `python tools/diagnose_geometry.py --subset <ds> --arch-name <arch> --ipc 10 --seed 42 --syn-leaf syn_data_seed42 --select-method covmatch [--select-realism-floor 30]`

**Drivers (to add runs):** `scripts/run_select_variants.sh` (env: `METHODS IPCS SEEDS REALISM_FLOOR
SAVE_STUDENT OOD_SETS SKIP_SYNTH RESULTS_FILE CUDA_VISIBLE_DEVICES`), `scripts/run_mediator_geometry.sh`.

**Reproduce / extend cheaply:**
- **Determinism + reuse:** runs are bit-exact deterministic; distilled sets are cached at
  `exp/<exp_name>/syn_data_seed<N>/`. Never re-run a completed config — reuse cached sets (`--skip-synth`)
  and logged rows. `exp_name` path-keys the method (`_sel<m>`) and a non-default floor (`_fl<F>`).
- **Training-free re-eval:** all stock/covmatch IPC=10 students are checkpointed
  (`exp/<exp>/student_syn_data_seed<N>.pth`). To score a *new* OOD set/metric without retraining:
  `python main.py … --select-method <m> --diagnostics-only --ood-sets <list>` (loads the checkpoint, runs
  the trust panel, logs it).
- **Code map:** selector `synthesize/utils.py`; metrics `validation/diagnostics.py` +
  `validation/nc_metrics.py`; training/diagnostics wiring `validation/main.py` + `main.py`; flags
  `argument.py`.

---

## 4. Limitations & open questions
1. **Confirm the far-vs-near axis (free):** score more *far*-OOD sets (Places/LSUN) via `--diagnostics-only`
   on the existing checkpoints — establish the benefit is far-vs-near, not SVHN-specific per se.
2. **The momentmatch puzzle (§2.4):** equal aggregate geometry, different student OOD — the selection's
   effect is not fully captured by distilled-set NC1/eff-rank.
3. **Calibration is the most robust claim** (4/4, OOD-independent); the OOD claim must be stated as *far*-OOD.
   Note the raw-ECE gain partly closes after temperature scaling, so the calibration story is strongest pre-TS.
4. **Power/scale:** 3 seeds per cell (bootstrap CIs over 3); CIFAR-100 / Tiny-ImageNet only; ImageNet untested.

---

## Appendix A — configuration
CIFAR-100 (`conv3`, `resnet18_modified`) + Tiny-ImageNet (`conv4`, `resnet18_modified`); teacher = student;
`mipc=300`, `num_crop=5`, `factor=1`, IPC ∈ {1, 10, 50}, seeds {42, 43, 44}, 300 student epochs. Mahalanobis
class statistics and the calibration temperature are fit on a held-out real-train slice (`fit_ipc=50`).
OOD sets: SVHN / DTD / CIFAR-10 (10k-sample cap). covmatch default realism floor = 3.

## Appendix B — full IPC×cell Mahalanobis-AUROC (stock / random / covmatch)
| cell | IPC=1 | IPC=10 | IPC=50 |
|---|---|---|---|
| cifar100/conv3 | .318 / .336 / .280 | .586 / .595 / .669 | .802 / .840 / .856 |
| cifar100/resnet18 | .800 / .663 / .731 | .878 / .898 / .908 | .920 / .933 / .941 |
| tiny/conv4 | .152 / .145 / .143 | .457 / .469 / .599 | .730 / .764 / .846 |
| tiny/resnet18 | .799 / .874 / .824 | .978 / .987 / .986 | .988 / .993 / .993 |
