"""Unit tests for the trustworthiness/geometry metrics.

Pure CPU, no datasets — each test builds a tiny synthetic case with a known answer.
Run with `pytest tests/` or directly: `python tests/test_metrics.py`.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.nc_metrics import compute_nc_metrics
from validation.diagnostics import (
    compute_ece,
    msp_scores,
    feat_norm_scores,
    fit_mahalanobis,
    maha_scores,
    fit_temperature,
    ood_metrics,
    oscr,
)


def _two_class_logits_at_confidence(conf):
    """Logit row [m, 0] whose softmax max equals `conf` (for class 0)."""
    m = float(np.log(conf / (1.0 - conf)))
    return [m, 0.0]


def test_ece_uniform_confidence():
    # All predictions at confidence 0.9, exactly half correct => ECE = |0.9 - 0.5| = 0.4.
    row = _two_class_logits_at_confidence(0.9)
    n = 100
    logits = torch.tensor([row] * n)            # always predicts class 0 at conf 0.9
    labels = torch.tensor([0] * (n // 2) + [1] * (n // 2))  # 50% correct
    out = compute_ece(logits, labels)
    assert abs(out["ece"] - 0.4) < 1e-6, out
    assert abs(out["avg_conf"] - 0.9) < 1e-6, out
    assert abs(out["acc"] - 0.5) < 1e-6, out
    assert abs(out["overconf_gap"] - 0.4) < 1e-6, out


def test_ece_perfect_calibration():
    # Confidence 0.5 with 50% accuracy => perfectly calibrated => ECE = 0.
    row = _two_class_logits_at_confidence(0.5)  # [0, 0] -> conf 0.5
    n = 100
    logits = torch.tensor([row] * n)
    labels = torch.tensor([0] * (n // 2) + [1] * (n // 2))
    out = compute_ece(logits, labels)
    assert out["ece"] < 1e-6, out


def _gaussian_clusters(means, n_per, std, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats, labels = [], []
    for c, mu in enumerate(means):
        mu = torch.tensor(mu, dtype=torch.float32)
        x = mu + std * torch.randn(n_per, len(mu), generator=g)
        feats.append(x)
        labels += [c] * n_per
    return torch.cat(feats, 0), torch.tensor(labels)


def test_nc1_separated_vs_overlapping():
    means = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    tight, lab = _gaussian_clusters(means, 200, std=0.1, seed=1)
    loose, _ = _gaussian_clusters(means, 200, std=3.0, seed=1)
    nc_tight = compute_nc_metrics(tight, lab, num_classes=3)["nc1"]
    nc_loose = compute_nc_metrics(loose, lab, num_classes=3)["nc1"]
    assert nc_tight < 0.05, nc_tight            # near-zero within-class variance
    assert nc_loose > nc_tight * 5, (nc_tight, nc_loose)  # more spread => larger NC1


def test_maha_separates_id_from_ood():
    means = [[8.0, 0.0], [0.0, 8.0], [-8.0, 0.0]]
    fit_feats, fit_lab = _gaussian_clusters(means, 200, std=0.5, seed=2)
    cls_means, precision = fit_mahalanobis(fit_feats, fit_lab, num_classes=3)

    id_feats, _ = _gaussian_clusters(means, 100, std=0.5, seed=3)     # same dist
    g = torch.Generator().manual_seed(4)
    ood_feats = torch.tensor([40.0, 40.0]) + 0.5 * torch.randn(300, 2, generator=g)

    id_s = maha_scores(id_feats, cls_means, precision)
    ood_s = maha_scores(ood_feats, cls_means, precision)
    auroc = ood_metrics(id_s, ood_s)["auroc"]
    assert auroc > 0.99, auroc
    assert id_s.mean() > ood_s.mean()           # higher score = more ID


def test_feat_norm_direction_and_ood():
    g = torch.Generator().manual_seed(5)
    id_feats = 10.0 + torch.randn(200, 4, generator=g)   # large-norm ID
    ood_feats = 0.1 * torch.randn(200, 4, generator=g)   # near-origin OOD
    id_s = feat_norm_scores(id_feats)
    ood_s = feat_norm_scores(ood_feats)
    assert ood_metrics(id_s, ood_s)["auroc"] > 0.99
    assert id_s.mean() > ood_s.mean()


def test_temperature_softens_overconfident_model():
    # Very peaked logits, 30% wrong => overconfident; T should rise and lower NLL.
    g = torch.Generator().manual_seed(6)
    n, k = 600, 5
    labels = torch.randint(0, k, (n,), generator=g)
    preds = labels.clone()
    flip = torch.rand(n, generator=g) < 0.3
    preds[flip] = (labels[flip] + 1) % k
    logits = torch.zeros(n, k)
    logits[torch.arange(n), preds] = 8.0         # scale 8 => very confident

    T = fit_temperature(logits, labels)
    nll = torch.nn.functional.cross_entropy
    assert T > 1.0, T
    assert nll(logits / T, labels) <= nll(logits, labels) + 1e-6


def test_ood_metrics_and_oscr_extremes():
    id_s = np.array([0.9, 0.8, 0.7, 0.6])
    ood_s = np.array([0.1, 0.2, 0.3])
    m = ood_metrics(id_s, ood_s)
    assert abs(m["auroc"] - 1.0) < 1e-9
    assert m["fpr95"] == 0.0
    # fully inverted => auroc 0
    assert abs(ood_metrics(ood_s, id_s)["auroc"]) < 1e-9
    # all-ID correct & perfectly separated => OSCR == 1
    val = oscr(id_s, np.ones(4, bool), ood_s)
    assert val > 0.99, val


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} metric tests passed.")


if __name__ == "__main__":
    _run_all()
