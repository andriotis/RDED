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
