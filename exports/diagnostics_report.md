# RDED — trustworthiness & geometry diagnostics


**Project.** RDED is an optimization-free dataset-distillation paradigm: instead of optimizing synthetic pixels, it selects realistic real-image crops and relabels them with a pretrained **teacher** via knowledge distillation. A **student** is then trained *only* on this tiny distilled set. RDED (and the dataset-distillation literature generally) reports **accuracy and nothing else**.

**What this adds.** Two diagnostics RDED never measures, to ask whether distilled-data training produces *trustworthy* models, and why:

- **H2 — student trustworthiness.** Train the student on stock RDED, then measure calibration (ECE) and open-set/OOD robustness (OSCR, AUROC, FPR95) in addition to accuracy. *Mean over 3 seeds.*
- **H1 — synthesis geometry.** With the teacher alone (no training), compare the neural-collapse geometry the teacher induces on the **distilled set** vs. an equally-sized **real** subset, against the teacher's own real-data reference. Asks whether the distilled set is geometrically *unlike* real data.

**Grid.** datasets {CIFAR-100, Tiny-ImageNet} × archs {Conv3/Conv4, ResNet-18 (modified)} × IPC {1, 10, 50}; OOD = SVHN.


## How to read the metrics

Every model is a **teacher** pretrained on the full real dataset, or a **student**
trained *only* on the small distilled set via knowledge distillation (KD) from
that teacher. Open-set / OOD negatives are the **SVHN** test split; in-distribution
(ID) is the real validation set. Accuracy/top-1 aside, none of these metrics are
something RDED (or dataset distillation in general) normally reports.

**Accuracy & calibration**
- `best_top1` / `top1` — top-1 accuracy (%). Higher = better.
- `ece` — Expected Calibration Error (15 bins). Gap between confidence and
  accuracy. **Lower = better** (0 = perfectly calibrated).
- `overconf_gap` — mean confidence minus accuracy. **Positive = overconfident.**
  (Note: across every student row it equals ECE almost exactly — i.e. *all* the
  miscalibration is overconfidence, never under-.)

**Open-set / OOD robustness (ID = real val, OOD = SVHN)**
- `oscr_msp` — Open-Set Classification Rate: area under the
  correct-classification-rate vs false-positive-rate curve. **Higher = better.**
- `auroc_msp` — AUROC of separating ID from OOD by max-softmax-prob.
  **Higher = better;** 0.5 = chance; **< 0.5 = the model is *more* confident on
  OOD than on real data** (inverted, worse than a coin flip).
- `fpr95_msp` — false-positive rate at 95% true-positive rate. **Lower = better.**
- `*_energy`, `*_maha`, `*_feat_norm` — the same OSCR/AUROC/FPR95 under alternative OOD
  scores: free energy (from logits), Mahalanobis distance to class means, and penultimate
  feature norm. `maha`/`feat_norm` are *geometry-aware* — they read the feature cluster
  structure directly, so they expose over-collapse more than MSP does.
- `ece_ts`, `auroc_msp_ts` — ECE and MSP-AUROC after **temperature scaling** (one scalar T
  fit by NLL on a held-out real-train slice). The gap to the un-scaled value is the share of
  miscalibration that is post-hoc-fixable.

**Feature geometry (Papyan–Han–Donoho neural collapse)**
Computed on penultimate features. *Which model and which split differ by table:* in **H2**
these are the trained **student** on **real validation**; in **H1** they are the **teacher**
on the distilled set / real subset. Tells us how "collapsed"/separable a set looks.
- `nc1` — within-class variance relative to between-class variance.
  **Lower = tighter, more separable clusters.**
- `nc2` — variation of class-mean norms (equinorm). **Lower = more equal norms**
  (closer to the ideal simplex equiangular tight frame, ETF).
- `nc3` — deviation of class-mean angles from the ideal ETF (equiangularity).
  **Lower = closer to ETF.**


## H2 — student trained on stock RDED (mean±std over seeds)

Higher OSCR/AUROC and lower ECE/FPR95 = more trustworthy. Each cell is mean±std across seeds (per-seed rows in `h2_per_seed.csv`). **NC1–NC3 here are the trained *student's* geometry on real validation** — contrast H1, whose NC is the *teacher's* geometry on the distilled/real set.

| dataset | arch | ipc | n_seeds | best_top1 | ece | overconf_gap | oscr_msp | auroc_msp | fpr95_msp | oscr_energy | auroc_energy | fpr95_energy | nc1 | nc2 | nc3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar100 | conv3 | 1 | 3 | 21.6133±0.3988 | 0.2025±0.009 | 0.2025±0.009 | 0.1758±0.0031 | 0.6741±0.0415 | 0.8744±0.0268 | 0.1694±0.0033 | 0.6469±0.0393 | 0.9339±0.0213 | 15.5787±0.2407 | 0.2963±0.0056 | 0.2161±0.0043 |
| cifar100 | conv3 | 10 | 3 | 48.5567±0.3113 | 0.0974±0.0095 | 0.0972±0.0095 | 0.4375±0.0088 | 0.8083±0.0287 | 0.6663±0.0557 | 0.4569±0.0045 | 0.9033±0.0176 | 0.3809±0.0511 | 9.2012±0.0343 | 0.2561±0.0024 | 0.1846±0.0049 |
| cifar100 | conv3 | 50 | 3 | 57.54±0.18 | 0.0971±0.0068 | 0.0971±0.0068 | 0.5383±0.0033 | 0.8721±0.0026 | 0.5082±0.0201 | 0.5492±0.0055 | 0.9284±0.0091 | 0.2867±0.0393 | 8.3858±0.0902 | 0.2514±0.0004 | 0.1703±0.0006 |
| cifar100 | resnet18_modified | 1 | 3 | 11.1533±0.2768 | 0.2739±0.0156 | 0.2738±0.0155 | 0.0679±0.0081 | 0.4758±0.0693 | 0.9701±0.0167 | 0.0589±0.0081 | 0.3791±0.0648 | 0.9911±0.0055 | 34.8044±0.715 | 0.357±0.0097 | 0.2737±0.0053 |
| cifar100 | resnet18_modified | 10 | 3 | 44.71±0.2081 | 0.1901±0.003 | 0.1901±0.003 | 0.3731±0.0032 | 0.6949±0.0074 | 0.8623±0.0053 | 0.3905±0.0041 | 0.7649±0.0079 | 0.82±0.0262 | 12.7254±0.0299 | 0.26±0.0021 | 0.1998±0.0076 |
| cifar100 | resnet18_modified | 50 | 3 | 63.2933±0.1498 | 0.1322±0.0021 | 0.1322±0.0021 | 0.5704±0.005 | 0.8177±0.012 | 0.7132±0.0324 | 0.5945±0.007 | 0.8942±0.0153 | 0.5307±0.056 | 7.8008±0.0809 | 0.2106±0.0044 | 0.1765±0.002 |
| tinyimagenet | conv4 | 1 | 3 | 13.31±0.1652 | 0.1772±0.0039 | 0.1772±0.0039 | 0.0934±0.0025 | 0.5184±0.0233 | 0.9652±0.0041 | 0.0801±0.0031 | 0.423±0.0126 | 0.9942±0.0021 | 17.2361±0.0194 | 0.2653±0.0077 | 0.2074±0.0021 |
| tinyimagenet | conv4 | 10 | 3 | 39.9467±0.1563 | 0.0643±0.0074 | 0.0642±0.0074 | 0.3282±0.0117 | 0.6583±0.0368 | 0.8803±0.0404 | 0.3053±0.0221 | 0.6256±0.0652 | 0.9392±0.028 | 12.3415±0.0628 | 0.2218±0.0035 | 0.1977±0.0007 |
| tinyimagenet | conv4 | 50 | 3 | 48.3433±0.4761 | 0.0557±0.0075 | 0.0553±0.0078 | 0.4618±0.0025 | 0.8869±0.0147 | 0.4992±0.0546 | 0.4637±0.0045 | 0.9154±0.0075 | 0.4518±0.035 | 9.7301±0.1231 | 0.2159±0.0024 | 0.1756±0.003 |
| tinyimagenet | resnet18_modified | 1 | 3 | 8.8733±0.2219 | 0.333±0.0245 | 0.3329±0.0244 | 0.035±0.0048 | 0.2825±0.037 | 0.9972±0.0015 | 0.0245±0.0051 | 0.1929±0.035 | 0.9998±0.0002 | 45.7487±0.3246 | 0.3115±0.0046 | 0.2431±0.0035 |
| tinyimagenet | resnet18_modified | 10 | 3 | 43.6967±0.8208 | 0.1538±0.0127 | 0.1537±0.0126 | 0.3696±0.0046 | 0.6886±0.0287 | 0.9066±0.0736 | 0.3734±0.0129 | 0.7059±0.0347 | 0.9936±0.0076 | 20.9542±0.1198 | 0.2466±0.0042 | 0.2065±0.002 |
| tinyimagenet | resnet18_modified | 50 | 3 | 58.1967±0.1222 | 0.115±0.0034 | 0.1149±0.0032 | 0.5493±0.0112 | 0.8659±0.0287 | 0.5938±0.0385 | 0.5589±0.0015 | 0.9046±0.0086 | 0.5874±0.1119 | 14.2708±0.1398 | 0.2188±0.0017 | 0.1847±0.0022 |

## H1 — teacher-induced geometry: distilled set vs real subset vs teacher reference

Three subjects per (dataset, arch, IPC): the distilled set, a real subset of the same size, and the teacher's full real-val reference (the only subject with OOD columns). Lower NC1/NC2/NC3 = more collapsed.

| subset | arch | ipc | subject | top1 | nc1 | nc2 | nc3 | ece | oscr_msp | auroc_msp | fpr95_msp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cifar100 | conv3 | 1 | distilled-set geometry (teacher feats) | 100 | 0 | 0.1973 | 0.0595 | 0.0159 |  |  |  |
| cifar100 | conv3 | 1 | real-train subset geometry (n/cls=1) | 74 | 0 | 0.1928 | 0.0578 | 0.0822 |  |  |  |
| cifar100 | conv3 | 1 | teacher reference (real val vs SVHN) | 59.67 | 8.1319 | 0.2495 | 0.1717 | 0.1035 | 0.5633 | 0.8801 | 0.4855 |
| cifar100 | conv3 | 10 | distilled-set geometry (teacher feats) | 99.9 | 0.8944 | 0.2117 | 0.0979 | 0.0259 |  |  |  |
| cifar100 | conv3 | 10 | real-train subset geometry (n/cls=10) | 75.2 | 1.521 | 0.173 | 0.1063 | 0.044 |  |  |  |
| cifar100 | conv3 | 10 | teacher reference (real val vs SVHN) | 59.67 | 8.1319 | 0.2495 | 0.1717 | 0.1035 | 0.5633 | 0.8801 | 0.4855 |
| cifar100 | conv3 | 50 | distilled-set geometry (teacher feats) | 99.02 | 2.697 | 0.2374 | 0.1297 | 0.0619 |  |  |  |
| cifar100 | conv3 | 50 | real-train subset geometry (n/cls=50) | 76.98 | 5.148 | 0.2178 | 0.1551 | 0.0246 |  |  |  |
| cifar100 | conv3 | 50 | teacher reference (real val vs SVHN) | 59.67 | 8.1319 | 0.2495 | 0.1717 | 0.1035 | 0.5633 | 0.8801 | 0.4855 |
| cifar100 | resnet18_modified | 1 | distilled-set geometry (teacher feats) | 100 | 0 | 0.1453 | 0.0782 | 0.0148 |  |  |  |
| cifar100 | resnet18_modified | 1 | real-train subset geometry (n/cls=1) | 100 | 0 | 0.1474 | 0.0869 | 0.0155 |  |  |  |
| cifar100 | resnet18_modified | 1 | teacher reference (real val vs SVHN) | 70.93 | 6.6222 | 0.1759 | 0.146 | 0.1216 | 0.6619 | 0.8649 | 0.5985 |
| cifar100 | resnet18_modified | 10 | distilled-set geometry (teacher feats) | 99 | 0.9469 | 0.1561 | 0.0944 | 0.0101 |  |  |  |
| cifar100 | resnet18_modified | 10 | real-train subset geometry (n/cls=10) | 99.5 | 1.9139 | 0.1265 | 0.1152 | 0.0177 |  |  |  |
| cifar100 | resnet18_modified | 10 | teacher reference (real val vs SVHN) | 70.93 | 6.6222 | 0.1759 | 0.146 | 0.1216 | 0.6619 | 0.8649 | 0.5985 |
| cifar100 | resnet18_modified | 50 | distilled-set geometry (teacher feats) | 97.8 | 1.7283 | 0.1728 | 0.1052 | 0.0079 |  |  |  |
| cifar100 | resnet18_modified | 50 | real-train subset geometry (n/cls=50) | 99.72 | 2.9123 | 0.1299 | 0.1243 | 0.0189 |  |  |  |
| cifar100 | resnet18_modified | 50 | teacher reference (real val vs SVHN) | 70.93 | 6.6222 | 0.1759 | 0.146 | 0.1216 | 0.6619 | 0.8649 | 0.5985 |
| tinyimagenet | conv4 | 1 | distilled-set geometry (teacher feats) | 100 | 0 | 0.127 | 0.0512 | 0.0191 |  |  |  |
| tinyimagenet | conv4 | 1 | real-train subset geometry (n/cls=1) | 70.5 | 0 | 0.1228 | 0.0513 | 0.0971 |  |  |  |
| tinyimagenet | conv4 | 1 | teacher reference (real val vs SVHN) | 50.15 | 8.9692 | 0.2161 | 0.1691 | 0.0651 | 0.4917 | 0.9446 | 0.2728 |
| tinyimagenet | conv4 | 10 | distilled-set geometry (teacher feats) | 99.75 | 1.4384 | 0.1581 | 0.101 | 0.0461 |  |  |  |
| tinyimagenet | conv4 | 10 | real-train subset geometry (n/cls=10) | 71 | 2.2049 | 0.1487 | 0.1069 | 0.091 |  |  |  |
| tinyimagenet | conv4 | 10 | teacher reference (real val vs SVHN) | 50.15 | 8.9692 | 0.2161 | 0.1691 | 0.0651 | 0.4917 | 0.9446 | 0.2728 |
| tinyimagenet | conv4 | 50 | distilled-set geometry (teacher feats) | 98.58 | 5.1839 | 0.2006 | 0.1415 | 0.1106 |  |  |  |
| tinyimagenet | conv4 | 50 | real-train subset geometry (n/cls=50) | 69.98 | 8.6733 | 0.2146 | 0.1687 | 0.077 |  |  |  |
| tinyimagenet | conv4 | 50 | teacher reference (real val vs SVHN) | 50.15 | 8.9692 | 0.2161 | 0.1691 | 0.0651 | 0.4917 | 0.9446 | 0.2728 |
| tinyimagenet | resnet18_modified | 1 | distilled-set geometry (teacher feats) | 100 | 0 | 0.1364 | 0.0802 | 0.0007 |  |  |  |
| tinyimagenet | resnet18_modified | 1 | real-train subset geometry (n/cls=1) | 100 | 0 | 0.1421 | 0.0856 | 0.0316 |  |  |  |
| tinyimagenet | resnet18_modified | 1 | teacher reference (real val vs SVHN) | 62.02 | 15.193 | 0.21 | 0.1676 | 0.1264 | 0.5973 | 0.907 | 0.3914 |
| tinyimagenet | resnet18_modified | 10 | distilled-set geometry (teacher feats) | 99.95 | 1.8818 | 0.1483 | 0.1002 | 0.0026 |  |  |  |
| tinyimagenet | resnet18_modified | 10 | real-train subset geometry (n/cls=10) | 99.85 | 4.4223 | 0.1484 | 0.1272 | 0.0287 |  |  |  |
| tinyimagenet | resnet18_modified | 10 | teacher reference (real val vs SVHN) | 62.02 | 15.193 | 0.21 | 0.1676 | 0.1264 | 0.5973 | 0.907 | 0.3914 |
| tinyimagenet | resnet18_modified | 50 | distilled-set geometry (teacher feats) | 99.72 | 3.883 | 0.1747 | 0.1158 | 0.0072 |  |  |  |
| tinyimagenet | resnet18_modified | 50 | real-train subset geometry (n/cls=50) | 99.87 | 7.7104 | 0.1581 | 0.1423 | 0.0272 |  |  |  |
| tinyimagenet | resnet18_modified | 50 | teacher reference (real val vs SVHN) | 62.02 | 15.193 | 0.21 | 0.1676 | 0.1264 | 0.5973 | 0.907 | 0.3914 |

## What jumps out (seeds for a thesis)

- **Trustworthiness scales with IPC, but never catches up.** As images-per-class go 1 → 10 → 50, ECE falls and OSCR/AUROC rise across every dataset×arch cell — yet even at IPC 50 students stay ECE ≈ 0.06–0.13 and AUROC ≈ 0.82–0.89, below the teacher's own real-data AUROC (0.86–0.94).
- **Low-IPC students have *inverted* open-set confidence.** At IPC 1, some cells score AUROC < 0.5 — cifar100/resnet18_modified (AUROC 0.48), tinyimagenet/resnet18_modified (AUROC 0.28) — i.e. the distilled-data student is, on average, *more* confident on SVHN than on real images.
- **All miscalibration is overconfidence.** `overconf_gap` ≈ `ece` in every student row, so confidence is uniformly inflated above accuracy — a single-signed, correctable error mode.
- **Capacity hurts here.** At matched low IPC the higher-capacity ResNet-18 students are *worse* calibrated and worse at open-set than the small Conv students — consistent with over-fitting the tiny distilled set.
- **The distilled set is *over-collapsed* vs real data.** Under teacher features the distilled set's NC1 is consistently *below* that of an equally-sized real subset (e.g. CIFAR-100/conv3/ipc50: 2.70 vs 5.15; Tiny/conv4/ipc50: 5.18 vs 8.67), and far below the teacher's NC1 on full real val (8–15). The teacher reads the distilled images as cleaner and more separable than real ones — the synthesis discards within-class spread that real data carries.

Taken together: **RDED optimizes accuracy but yields students that are systematically overconfident and open-set-fragile, and this traces to a teacher-induced geometry in which the distilled set is *over-collapsed* — it lacks the within-class spread of real data.** Natural thesis directions: trustworthiness-aware or geometry-targeted distillation (synthesize/relabel to match real within-class variance), calibration- or OOD-regularized relabeling, and characterizing the accuracy↔trust trade-off across IPC and capacity.
