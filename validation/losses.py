import torch
import torch.nn as nn
import torch.nn.functional as F


class OCCELoss(nn.Module):
    """One-Cold Cross-Entropy loss (Anticlasses paper).

    Pushes the softmax over `-logits` toward uniform on the (N-1) non-target
    classes, inducing neural collapse. Kept verbatim from
    Anticlasses/losses.py:6-18 so both repos agree on the math.
    """

    def forward(self, inputs, targets):
        N = inputs.shape[1]
        ycomp = (N - 1) * F.softmax(-inputs, dim=1)
        y = torch.ones((targets.size(0), N), device=inputs.device)
        y.scatter_(1, targets.unsqueeze(1), 0.0)
        loss = -1 / (N - 1) * torch.sum(y * torch.log(ycomp + 1e-7), dim=1)
        return torch.mean(loss)


def occe_per_sample(inputs, targets):
    """OCCE value per sample (no mean reduction). Used as a synthesis-time score."""
    N = inputs.shape[1]
    ycomp = (N - 1) * F.softmax(-inputs, dim=1)
    y = torch.ones((targets.size(0), N), device=inputs.device)
    y.scatter_(1, targets.unsqueeze(1), 0.0)
    return -1 / (N - 1) * torch.sum(y * torch.log(ycomp + 1e-7), dim=1)
