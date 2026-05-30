import collections

import torch
import torch.nn.functional as F


# Term-function contract for the registry:
#   def <name>_term(student_logits_mixed, teacher_logits_mixed, temperature,
#                   args=None, mix_info=None)
#       -> scalar tensor (batchmean-reduced)
#
# `args` carries optional per-loss hyperparameters (e.g. args.gce_q,
# args.sce_log_floor). `mix_info` carries the cutmix/mixup metadata
# (labels, rand_index, lam, num_classes) for terms that build hard-label
# targets from the mixing. Terms that don't need either argument ignore it.
# Module-level term wrappers below give every entry the same call shape,
# so the training loop can compose them with no per-loss branching.


MixInfo = collections.namedtuple(
    "MixInfo",
    "labels rand_index lam num_classes teacher_logits_unmixed",
)
# teacher_logits_unmixed is optional: only populated when a term that needs the
# teacher's prediction on the original un-mixed images is active (currently mxce).
MixInfo.__new__.__defaults__ = (None,)


def kl_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """Standard FKD knowledge-distillation term (matches upstream RDED's KL)."""
    log_q = F.log_softmax(student_logits_mixed / temperature, dim=1)
    p = F.softmax(teacher_logits_mixed / temperature, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean")


def ockl_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """One-Cold KL-divergence loss (OCKL).

    Both teacher and student logits are computed on the same cutmix-mixed
    images. Forward KL between the student's negated-softmax distribution
    and an inverted-teacher target:
      Q_student = softmax(-z_student / T)
      P_inv[k]  = (1 - p_teacher[k]) / (N - 1)   # equals (1 - p) / sum(1 - p)
      OCKL      = KL(P_inv || Q_student)
    """
    N = student_logits_mixed.shape[1]
    p_teacher = F.softmax(teacher_logits_mixed / temperature, dim=1)
    p_inv = (1.0 - p_teacher) / (N - 1)
    log_q = F.log_softmax(-student_logits_mixed / temperature, dim=1)
    return F.kl_div(log_q, p_inv, reduction="batchmean")


def ockl_logitneg_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """Symmetric negated-softmax KD: negate teacher LOGITS, not invert teacher
    PROBABILITIES.

      Q_student = softmax(-z_student / T)
      P_t_neg   = softmax(-z_teacher / T)     # NOT (1 - p_t)/(N-1)
      loss      = KL(P_t_neg || Q_student)

    Shares the same global minimum as `kl_term` (z_student = z_teacher + c),
    so unlike `ockl_term` it does not impose anti-class pressure on logit
    *shape*. It is a "distill the rejection knowledge" KD signal that
    emphasizes the teacher's most-rejected classes (where P_t_neg is large)
    instead of the teacher's most-confident classes (where p_t is large, as
    `kl_term` does).
    """
    log_q = F.log_softmax(-student_logits_mixed / temperature, dim=1)
    p_t_neg = F.softmax(-teacher_logits_mixed / temperature, dim=1)
    return F.kl_div(log_q, p_t_neg, reduction="batchmean")


def gce_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """Generalized Cross-Entropy (Zhang & Sabuncu 2018), soft-label form.

      L_GCE = mean_batch  sum_c  y_c * (1 - p_c^q) / q

    Reduces to CE as q -> 0, to MAE at q = 1. Noise-tolerant in the soft
    teacher-label regime since the gradient w.r.t. logits is bounded by
    p_c^q (vs unbounded for plain CE).
    """
    q = float(getattr(args, "gce_q", 0.7))
    p = F.softmax(student_logits_mixed / temperature, dim=1)
    y = F.softmax(teacher_logits_mixed / temperature, dim=1)
    per_sample = (y * (1.0 - p.pow(q)) / q).sum(dim=1)
    return per_sample.mean()


def rce_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """Reverse Cross-Entropy (Wang et al. 2019), soft-label form.

      L_RCE = mean_batch  -sum_c  p_c * log(max(y_c, exp(A)))

    The clamp at exp(A) replaces the log(0) substitution used with one-hot
    labels in the original paper; default A = -4 matches their setting.
    Compose with kl_term to realize SCE: --w-kl ALPHA --w-rce BETA.
    """
    A = float(getattr(args, "sce_log_floor", -4.0))
    p = F.softmax(student_logits_mixed / temperature, dim=1)
    y = F.softmax(teacher_logits_mixed / temperature, dim=1)
    log_y = torch.log(y.clamp(min=torch.exp(torch.tensor(A, dtype=y.dtype, device=y.device))))
    per_sample = -(p * log_y).sum(dim=1)
    return per_sample.mean()


def scce_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """Soft Complementary Cross-Entropy.

      L_sCCE = mean_batch  -sum_c  (1 - y_c) * log(1 - p_c)

    Uses log1p(-p) with p clamped below 1 for numerical stability when the
    student becomes confident. Estimates the same V-information quantity as
    KL from the rejection side of the teacher distribution.
    """
    p = F.softmax(student_logits_mixed / temperature, dim=1)
    y = F.softmax(teacher_logits_mixed / temperature, dim=1)
    log_one_minus_p = torch.log1p(-p.clamp(max=1.0 - 1e-7))
    per_sample = -((1.0 - y) * log_one_minus_p).sum(dim=1)
    return per_sample.mean()


def mxce_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """Mix-aware KL: teacher target is the convex mix of per-original-image
    teacher predictions, instead of the teacher's prediction on the mixed image.

      p_T_host    = softmax(teacher(images)            / T)
      p_T_partner = softmax(teacher(images[rand_idx])  / T)
      p_T_target  = lam * p_T_host + (1 - lam) * p_T_partner
      L_mxce      = KL( log_softmax(student(mix_images)/T)  ||  p_T_target )

    Same student forward as kl_term (on the mixed image) -- only the target
    changes. Requires mix_info.teacher_logits_unmixed to be the teacher's
    logits on the un-mixed `images` (main.py populates this when mxce is active).
    When mix_info.rand_index is None (no cutmix/mixup), the target collapses to
    p_T_host and the loss equals kl_term up to the un-mixed teacher pass.
    """
    if mix_info is None or mix_info.teacher_logits_unmixed is None:
        raise ValueError(
            "mxce_term requires mix_info.teacher_logits_unmixed; "
            "main.py must compute teacher_model(images) when --w-mxce > 0"
        )
    z_T_un = mix_info.teacher_logits_unmixed
    p_T_host = F.softmax(z_T_un / temperature, dim=1)
    if mix_info.rand_index is None or mix_info.lam is None or float(mix_info.lam) >= 1.0:
        p_T_target = p_T_host
    else:
        lam = float(mix_info.lam)
        partner_idx = mix_info.rand_index.to(z_T_un.device)
        p_T_partner = p_T_host[partner_idx]
        p_T_target = lam * p_T_host + (1.0 - lam) * p_T_partner
    log_q = F.log_softmax(student_logits_mixed / temperature, dim=1)
    return F.kl_div(log_q, p_T_target, reduction="batchmean")


def socce_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None):
    """Soft One-Cold Cross-Entropy (anti-class structural regularizer).

    Target is built from the cutmix/mixup HARD ground-truth labels, NOT from
    the teacher's soft prediction:
      y_mix_c = lam * 1[c = y_i] + (1 - lam) * 1[c = y_partner]
      y_bar_c = (1 - y_mix_c) / (N - 1)        # valid distribution, sums to 1
    The OCCE cross-entropy is then taken on the student's RAW (un-tempered)
    logits z_S:
      L_sOCCE = mean_batch  -sum_c y_bar_c * log(softmax(-z_S)_c)

    When mix_info.rand_index is None (no mixing), reduces to one-hot
    one-cold encoding from the OCCE paper.
    """
    if mix_info is None or mix_info.labels is None:
        raise ValueError("socce_term requires mix_info with labels")
    labels = mix_info.labels
    rand_index = mix_info.rand_index
    lam = float(mix_info.lam) if mix_info.lam is not None else 1.0
    num_classes = mix_info.num_classes
    B = student_logits_mixed.size(0)

    y_mix = torch.zeros(B, num_classes, device=student_logits_mixed.device, dtype=student_logits_mixed.dtype)
    y_mix.scatter_(1, labels.view(-1, 1), lam)
    if rand_index is not None and lam < 1.0:
        partner = labels[rand_index.to(labels.device)].view(-1, 1)
        y_mix.scatter_add_(1, partner, torch.full_like(partner, 1.0 - lam, dtype=y_mix.dtype))

    y_bar = (1.0 - y_mix) / (num_classes - 1)
    log_anti = F.log_softmax(-student_logits_mixed, dim=1)
    per_sample = -(y_bar * log_anti).sum(dim=1)
    return per_sample.mean()


# Single source of truth for student-loss terms available to validation/main.py.
# To add a new loss "A":
#   1. Implement `def A_term(student_logits_mixed, teacher_logits_mixed, temperature, args=None, mix_info=None)`
#      returning a batchmean-reduced scalar tensor.
#   2. Add `"A": (A_term, <default_weight>)` below.
# argument.py auto-adds --w-A; validation/main.py auto-composes; results_logger
# auto-logs it under row["weights"]["A"]; sweep YAMLs can use `weights: {A: 0.5}`.
LOSS_REGISTRY = {
    "kl":            (kl_term,            1.0),  # stock RDED behavior when no flags are passed
    # DEPRECATED (kept importable for reproducibility; see docs/LOSSES.md "Diagnosis verdicts"):
    # ockl/socce are fully weight-swept and never beat KL -- their optimum is w->0.
    "ockl":          (ockl_term,          0.0),  # DEPRECATED negated-softmax vs inverted target (forward KL)
    "ockl_logitneg": (ockl_logitneg_term, 0.0),  # negate teacher logits (not probabilities); shares KL's minimum
    "gce":           (gce_term,           0.0),  # noise-tolerant CE<->MAE interpolant (q via --gce-q)
    "rce":           (rce_term,           0.0),  # reverse CE; compose with kl for SCE (log floor via --sce-log-floor)
    "scce":          (scce_term,          0.0),  # soft complementary CE; rejection-side V-information estimator
    "socce":         (socce_term,         0.0),  # DEPRECATED soft one-cold CE on cutmix hard labels, raw logits
    "mxce":          (mxce_term,          0.0),  # mix-aware KL: target = lam*p_T(host) + (1-lam)*p_T(partner)
}

# Terms that require teacher_logits on the un-mixed images (populated by main.py).
TERMS_NEEDING_UNMIXED_TEACHER = frozenset({"mxce"})
