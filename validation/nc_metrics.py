"""NC1-NC4 metrics. Mirror of Anticlasses/utils.py:compute_nc_metrics.

If you change the math here, mirror it in the Anticlasses repo (or vice versa)
so the two projects agree on what they're reporting.
"""

import torch


def compute_nc_metrics(features, labels, num_classes, classifier_preds=None):
    """Papyan-Han-Donoho neural-collapse metrics on penultimate features.

    Args:
        features: [N, D] tensor of penultimate-layer activations.
        labels: [N] int tensor of true class labels in [0, num_classes).
        num_classes: total class count (only classes that appear contribute).
        classifier_preds: [N] int tensor of model predictions. If None, NC4 is None.

    Returns:
        dict with keys nc1, nc2, nc3, nc4 (nc4 may be None).
    """
    features = features.detach().to(torch.float64)
    labels = labels.detach().long()
    N, D = features.shape

    mu_G = features.mean(dim=0)

    device = features.device
    class_means = torch.zeros(num_classes, D, dtype=torch.float64, device=device)
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    class_means.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones(N, dtype=torch.float64, device=device))
    present = counts > 0
    K_present = int(present.sum().item())
    class_means[present] = class_means[present] / counts[present].unsqueeze(1)

    centered = features - class_means[labels]
    Sigma_W = centered.t() @ centered / N

    centered_means = class_means[present] - mu_G
    Sigma_B = centered_means.t() @ centered_means / max(K_present, 1)

    Sigma_B_pinv = torch.linalg.pinv(Sigma_B)
    nc1 = torch.trace(Sigma_W @ Sigma_B_pinv).item() / max(K_present, 1)

    norms = torch.linalg.norm(centered_means, dim=1)
    nc2 = (norms.std(unbiased=False) / (norms.mean() + 1e-12)).item()

    if K_present > 1:
        normalized = centered_means / (norms.unsqueeze(1) + 1e-12)
        cos_matrix = normalized @ normalized.t()
        mask = ~torch.eye(K_present, dtype=torch.bool, device=device)
        off_diag = cos_matrix[mask]
        ideal = -1.0 / (K_present - 1)
        nc3 = (off_diag - ideal).abs().mean().item()
    else:
        nc3 = float("nan")

    nc4 = None
    if classifier_preds is not None:
        dists = torch.cdist(features, class_means)
        dists[:, ~present] = float("inf")
        ncm_preds = dists.argmin(dim=1)
        nc4 = (ncm_preds == classifier_preds.to(device).long()).float().mean().item()

    return {"nc1": nc1, "nc2": nc2, "nc3": nc3, "nc4": nc4}
