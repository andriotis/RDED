"""Unit tests for the relmatch (class-relation matrix) selector and its diagnostic.

Pure CPU, no datasets — each test builds a tiny synthetic case with a known answer.
Run with `pytest tests/` or directly: `python tests/test_select.py`.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthesize.utils import _select_relmatch
from validation.diagnostics import class_relation_matrix, relation_divergence


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
    sel = _select_relmatch(5, probs, probs.mean(0), diag_weight=0.0, diag_idx=0, gen=None)
    assert sel.shape == (5,)
    assert sel.unique().numel() == 5
    assert int(sel.min()) >= 0 and int(sel.max()) < 12


def test_select_relmatch_recovers_exact_matching_pair():
    # Two symmetric pairs about the mean mu; any pair averages to mu, so a size-2 subset hits the
    # target exactly and the greedy must find it (full-row match, diag_weight=1).
    mu = torch.tensor([0.5, 0.3, 0.2])
    probs = torch.tensor([
        [0.6, 0.2, 0.2],   # 0  (partner of 1: row0 + row1 = 2*mu)
        [0.4, 0.4, 0.2],   # 1
        [0.5, 0.4, 0.1],   # 2  (partner of 3)
        [0.5, 0.2, 0.3],   # 3
    ])
    assert torch.allclose(probs.mean(0), mu, atol=1e-6)
    sel = _select_relmatch(2, probs, probs.mean(0), diag_weight=1.0, diag_idx=0, gen=None)
    assert sel.unique().numel() == 2
    assert torch.allclose(probs[sel].mean(0).double(), mu.double(), atol=1e-6)  # J = 0


def test_select_relmatch_diag_weight_enters_objective():
    # For n=1 the greedy first pick is the global argmin of the single-candidate weighted SSE, so
    # we can brute-force check (in float64, matching the selector) that the self-class weight is
    # applied to coordinate diag_idx and nowhere else.
    probs = torch.softmax(torch.randn(20, 5, generator=torch.Generator().manual_seed(3)), dim=1)
    r = probs.mean(0)
    diag_idx = 2
    for dw in (0.0, 1.0):
        w = torch.ones(5, dtype=torch.float64)
        w[diag_idx] = dw
        brute = int((((probs.double() - r.double()) ** 2) * w).sum(dim=1).argmin())
        got = int(_select_relmatch(1, probs, r, diag_weight=dw, diag_idx=diag_idx, gen=None)[0])
        assert got == brute, f"dw={dw}: greedy {got} != brute {brute}"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all tests passed")
