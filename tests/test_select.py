"""Unit tests for the relmatch / reldist (class-relation matrix / distribution) selectors + diagnostics.

Pure CPU, no datasets — each test builds a tiny synthetic case with a known answer.
Run with `pytest tests/` or directly: `python tests/test_select.py`.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesize.utils import _select_relmatch, _select_reldist, _select_facloc
from validation.diagnostics import (
    class_relation_matrix,
    relation_divergence,
    wasserstein1d_sq,
    relation_distribution_divergence,
)


def test_class_relation_matrix_known():
    # logits = log(p) reproduces p under softmax (p already sums to 1), so R is the planted rows.
    p0 = torch.tensor([0.8, 0.2])
    p1 = torch.tensor([0.3, 0.7])
    logits = torch.log(torch.stack([p0, p0, p1, p1]))
    labels = torch.tensor([0, 0, 1, 1])
    R = class_relation_matrix(logits, labels, num_classes=2)
    assert torch.allclose(R, torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=R.dtype), atol=1e-6)
    assert torch.allclose(R.sum(dim=1), torch.ones(2, dtype=R.dtype), atol=1e-6)  # row-stochastic


def test_class_relation_matrix_absent_class_zero_row():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]))
    labels = torch.tensor([0, 0])  # classes 1 and 2 have no samples
    R = class_relation_matrix(logits, labels, num_classes=3)
    assert torch.allclose(R[1], torch.zeros(3, dtype=R.dtype))
    assert torch.allclose(R[2], torch.zeros(3, dtype=R.dtype))


def test_relation_divergence_identity():
    R = torch.softmax(torch.randn(5, 5, generator=torch.Generator().manual_seed(0)), dim=1).double()
    d = relation_divergence(R, R)
    assert d["rel_frob"] < 1e-9
    assert d["rel_frob_offdiag"] < 1e-9
    assert d["rel_row_cos_offdiag"] < 1e-9
    assert abs(d["rel_eig_overlap"] - 1.0) < 1e-6


def test_relation_divergence_offdiag_sensitivity():
    # Perturbing a single off-diagonal entry (kept row-stochastic) must move rel_frob_offdiag.
    R_real = torch.softmax(torch.randn(4, 4, generator=torch.Generator().manual_seed(1)), dim=1).double()
    R_syn = R_real.clone()
    R_syn[0, 1] += 0.1
    R_syn[0, 0] -= 0.1
    d = relation_divergence(R_real, R_syn)
    assert d["rel_frob_offdiag"] > 1e-3


def test_select_relmatch_returns_n_distinct_in_range():
    probs = torch.softmax(torch.randn(12, 4, generator=torch.Generator().manual_seed(0)), dim=1)
    sel = _select_relmatch(5, probs, probs.mean(0), gen=None)
    assert sel.shape == (5,)
    assert sel.unique().numel() == 5
    assert int(sel.min()) >= 0 and int(sel.max()) < 12


def test_select_relmatch_recovers_exact_matching_pair():
    # Two symmetric pairs about the mean mu; any pair averages to mu, so a size-2 subset hits the
    # target exactly and the greedy must find it (full-row match).
    mu = torch.tensor([0.5, 0.3, 0.2])
    probs = torch.tensor([
        [0.6, 0.2, 0.2],   # 0  (partner of 1: row0 + row1 = 2*mu)
        [0.4, 0.4, 0.2],   # 1
        [0.5, 0.4, 0.1],   # 2  (partner of 3)
        [0.5, 0.2, 0.3],   # 3
    ])
    assert torch.allclose(probs.mean(0), mu, atol=1e-6)
    sel = _select_relmatch(2, probs, probs.mean(0), gen=None)
    assert sel.unique().numel() == 2
    assert torch.allclose(probs[sel].mean(0).double(), mu.double(), atol=1e-6)  # J = 0


# ---------------------------------------------------------------------------------------------------
# reldist (Formalization 2: per-class relational distribution) selector + distributional diagnostic
# ---------------------------------------------------------------------------------------------------

def _cat_probs(dog, car_const=0.05):
    """[P, 3] soft-labels for the cat class (col 0=cat, 1=dog, 2=car) from a dog-axis vector, with car
    held constant so the only *active* off-diagonal axis is dog — a clean, hand-checkable 1-D case."""
    dog = torch.as_tensor(dog, dtype=torch.float64)
    car = torch.full_like(dog, car_const)
    cat = 1.0 - dog - car
    return torch.stack([cat, dog, car], dim=1)


def test_wasserstein1d_sq_known():
    z = torch.tensor([0.0, 1.0, 2.0])
    assert wasserstein1d_sq(z, z) < 1e-12                                              # identical -> 0
    assert abs(wasserstein1d_sq(torch.tensor([0.0]), torch.tensor([2.0])) - 4.0) < 1e-9  # |0-2|^2 = 4
    # {.1,.3} reproduces the bimodal pool {.1,.1,.3,.3} exactly; {.1,.1} leaves W2^2 = .02.
    pool = torch.tensor([0.1, 0.1, 0.3, 0.3])
    assert wasserstein1d_sq(torch.tensor([0.1, 0.3]), pool) < 1e-12
    assert abs(wasserstein1d_sq(torch.tensor([0.1, 0.1]), pool) - 0.02) < 1e-6  # float32-input noise


def test_select_reldist_returns_n_distinct_in_range():
    probs = torch.softmax(torch.randn(12, 4, generator=torch.Generator().manual_seed(0)), dim=1)
    sel = _select_reldist(5, probs, probs, gen=None)
    assert sel.shape == (5,)
    assert sel.unique().numel() == 5
    assert int(sel.min()) >= 0 and int(sel.max()) < 12


def test_select_reldist_recovers_distribution():
    # Pool = two distinct soft-label points, each repeated; the W2-optimal size-2 subset takes one of
    # each, reproducing the full per-coordinate distribution exactly (W2 = 0 on every coordinate).
    A, B = [0.7, 0.2, 0.1], [0.5, 0.3, 0.2]
    probs = torch.tensor([A, A, B, B])
    sel = _select_reldist(2, probs, probs, gen=None)
    assert sel.unique().numel() == 2
    for d in range(3):
        assert wasserstein1d_sq(probs[sel, d], probs[:, d]) < 1e-9


def test_select_reldist_does_not_collapse_to_one_mode():
    # Two clusters (3 crops near A, 3 near B). The quantizer must straddle both — one crop from each
    # cluster — rather than piling onto a single mode the way a center-anchored greedy would.
    A, B = [0.80, 0.15, 0.05], [0.40, 0.35, 0.25]
    probs = torch.tensor([A, A, A, B, B, B])
    sel = _select_reldist(2, probs, probs, gen=None)
    rows = {tuple(round(float(v), 3) for v in probs[i]) for i in sel}
    assert tuple(round(x, 3) for x in A) in rows and tuple(round(x, 3) for x in B) in rows


# ---------------------------------------------------------------------------------------------------
# facloc (Formalization 3: submodular / set-level facility-location coverage) selector
# ---------------------------------------------------------------------------------------------------

def test_select_facloc_returns_n_distinct_in_range():
    probs = torch.softmax(torch.randn(12, 4, generator=torch.Generator().manual_seed(0)), dim=1)
    sel = _select_facloc(5, probs, probs, gen=None)
    assert sel.shape == (5,)
    assert sel.unique().numel() == 5
    assert int(sel.min()) >= 0 and int(sel.max()) < 12


def test_select_facloc_straddles_two_clusters():
    # Two well-separated clusters (3 near A, 3 near B). A coverage objective cannot leave a whole
    # cluster uncovered, so the size-2 pick must take one crop from each cluster (not pile onto one).
    A, B = [0.80, 0.15, 0.05], [0.05, 0.15, 0.80]
    probs = torch.tensor([A, A, A, B, B, B])
    sel = _select_facloc(2, probs, probs, gen=None)
    rows = {tuple(round(float(v), 3) for v in probs[i]) for i in sel}
    assert tuple(round(x, 3) for x in A) in rows and tuple(round(x, 3) for x in B) in rows


def test_select_facloc_feature_space_straddles_two_clusters():
    # Space-agnostic: the same straddle guarantee holds on raw (feature-like) embeddings, since
    # _select_facloc L2-normalizes whatever it is given. Two clusters far apart in direction.
    A, B = [3.0, 0.1, 0.0], [0.0, 0.1, 3.0]
    feats = torch.tensor([A, A, A, B, B, B])
    sel = _select_facloc(2, feats, feats, gen=None)
    import torch.nn.functional as _Fn
    Z = _Fn.normalize(feats.double(), dim=1)
    picked = Z[sel]
    # the two picks must be near-orthogonal (one per cluster), not from the same cluster
    assert float((picked[0] @ picked[1]).abs()) < 0.5


def test_select_facloc_first_pick_is_medoid_not_outlier():
    # Asymmetric pool: one outlier at index 0 plus a tight 4-crop cluster. The coverage medoid
    # (the crop with the largest total similarity to the pool) is a cluster crop, NOT the outlier.
    # A -inf coverage frontier makes every first-round marginal gain +inf, so argmax returns index 0
    # (the outlier); the correct -1.0 (min-cosine) frontier ranks by F({j}) = sum_i cos and picks the
    # medoid. n=1 isolates the very first greedy pick.
    O = [0.05, 0.05, 0.90]          # outlier (points to class 2), placed at index 0
    C = [0.80, 0.15, 0.05]          # tight cluster (points to class 0)
    probs = torch.tensor([O, C, C, C, C])
    sel = _select_facloc(1, probs, probs, gen=None)
    assert int(sel[0]) != 0                                              # not the outlier
    assert tuple(round(float(v), 3) for v in probs[sel[0]]) == tuple(round(x, 3) for x in C)


def test_relation_distribution_divergence_identity():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 5, generator=g)
    labels = torch.randint(0, 5, (40,), generator=g)
    d = relation_distribution_divergence(logits, labels, logits, labels, num_classes=5)
    assert d["reldist_w_off"] < 1e-9
    assert d["reldist_w_full"] < 1e-9
    assert d["reldist_sw"] < 1e-9
    assert abs(d["reldist_tail_cov"] - 1.0) < 1e-6


def test_relation_distribution_divergence_sensitivity():
    # Pushing the synthetic cloud's mass toward one class shifts that (off-diagonal, for the other
    # classes) coordinate's distribution, so the headline inter-class Wasserstein must rise above zero.
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(60, 5, generator=g)
    labels = torch.randint(0, 5, (60,), generator=g)
    logits_syn = logits.clone()
    logits_syn[:, 1] += 1.0
    d = relation_distribution_divergence(logits, labels, logits_syn, labels, num_classes=5)
    assert d["reldist_w_off"] > 1e-3


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all tests passed")
