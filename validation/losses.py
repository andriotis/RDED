import torch
import torch.nn as nn
import torch.nn.functional as F


class OCKLLoss(nn.Module):
    """One-Cold KL-divergence loss (OCKL).

    Both teacher and student logits are computed on the same cutmix-mixed images.
    Forward KL between the student's negated-softmax distribution and an
    inverted-teacher target:
      Q_student  = softmax(-z_student / T)
      P_inv[k]   = (1 - p_teacher[k]) / (N - 1)   # equals (1 - p) / sum(1 - p)
      OCKL       = KL(P_inv || Q_student),  reduction='batchmean'
    """

    def forward(self, student_logits_mixed, teacher_logits_mixed, temperature):
        N = student_logits_mixed.shape[1]
        p_teacher = F.softmax(teacher_logits_mixed / temperature, dim=1)
        p_inv = (1.0 - p_teacher) / (N - 1)
        log_q = F.log_softmax(-student_logits_mixed / temperature, dim=1)
        return F.kl_div(log_q, p_inv, reduction="batchmean")


# Term-function contract for the registry:
#   def <name>_term(student_logits_mixed, teacher_logits_mixed, temperature)
#       -> scalar tensor (batchmean-reduced)
#
# Module-level term wrappers below give every entry the same call shape,
# so the training loop can compose them with no per-loss branching.


def kl_term(student_logits_mixed, teacher_logits_mixed, temperature):
    """Standard FKD knowledge-distillation term (matches upstream RDED's KL)."""
    log_q = F.log_softmax(student_logits_mixed / temperature, dim=1)
    p = F.softmax(teacher_logits_mixed / temperature, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean")


_OCKL = OCKLLoss()  # singleton; OCKLLoss has no learnable parameters


def ockl_term(student_logits_mixed, teacher_logits_mixed, temperature):
    return _OCKL(student_logits_mixed, teacher_logits_mixed, temperature)


# Single source of truth for student-loss terms available to validation/main.py.
# To add a new loss "A":
#   1. Implement `def A_term(student_logits_mixed, teacher_logits_mixed, temperature)`
#      returning a batchmean-reduced scalar tensor.
#   2. Add `"A": (A_term, <default_weight>)` below.
# argument.py auto-adds --w-A; validation/main.py auto-composes; results_logger
# auto-logs it under row["weights"]["A"]; sweep YAMLs can use `weights: {A: 0.5}`.
LOSS_REGISTRY = {
    "kl":   (kl_term,   1.0),  # stock RDED behavior when no flags are passed
    "ockl": (ockl_term, 0.0),  # disabled unless explicitly enabled
}
