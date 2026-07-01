"""Tests for deterministic run identity and sweep-level completed-cell skips."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.losses import LOSS_REGISTRY
from validation.run_key import (
    canonical_run_key,
    canonical_weights,
    find_completed_result,
    load_result_history,
    run_hash,
    selector_config,
    selector_from_exp_name,
    selector_suffix,
)


REPO = Path(__file__).resolve().parents[1]


def _params(**overrides):
    base = {
        "dataset": "cifar100",
        "arch": "resnet18_modified",
        "stud_arch": "resnet18_modified",
        "seed": 42,
        "ipc": 10,
        "factor": 1,
        "num_crop": 5,
        "mipc": 300,
        "re_epochs": 300,
        "weights": {"kl": 1.0},
        "select_method": "stock",
    }
    base.update(overrides)
    return base


def _args_namespace(weights, *, gce_q=0.7, select_method="stock"):
    """Mimic the argparse ``args`` object: loss weights live as ``w_<name>``
    attributes (no ``weights`` dict) and dataset/arch use subset/arch_name."""
    ns = SimpleNamespace(
        subset="cifar100",
        arch_name="resnet18_modified",
        stud_name="resnet18_modified",
        seed=42, ipc=10, factor=1, num_crop=5, mipc=300, re_epochs=300,
        select_method=select_method, select_k=8, select_beta=0.0,
        select_quality="confidence", momentmatch_mean_weight=1.0,
        gce_q=gce_q,
    )
    for name in LOSS_REGISTRY:
        setattr(ns, f"w_{name}", float(weights.get(name, 0.0)))
    return ns


def _row(params, *, diagnostics=False):
    key = canonical_run_key(params)
    row = {
        "dataset": key["dataset"],
        "arch": key["arch"],
        "stud": key["stud"],
        "seed": key["seed"],
        "ipc": key["ipc"],
        "factor": key["factor"],
        "num_crop": key["num_crop"],
        "mipc": key["mipc"],
        "re_epochs": key["re_epochs"],
        "weights": canonical_weights(params.get("weights")),
        "gce_q": params.get("gce_q", 0.7),
        "best_top1": 1.0,
        "exp_name": "cifar100_resnet18_modified_f1_mipc300_ipc10_cr5",
        "run_key": key,
        "run_hash": run_hash(key),
    }
    if diagnostics:
        row["diag"] = {"ood": {"svhn": {"auroc_msp": 0.5}}}
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_run_key_matches_same_seed_and_hyperparams():
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        params = _params()
        results = tmp_path / "results.jsonl"
        _write_jsonl(results, [_row(params)])
        assert find_completed_result(params, load_result_history(results)) is not None


def test_run_key_distinguishes_core_hyperparams():
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        params = _params()
        results = tmp_path / "results.jsonl"
        _write_jsonl(results, [_row(params)])
        history = load_result_history(results)

        for field, value in [
            ("factor", 2),
            ("num_crop", 3),
            ("mipc", 100),
            ("re_epochs", 100),
        ]:
            changed = _params(**{field: value})
            assert find_completed_result(changed, history) is None, field


def test_run_key_distinguishes_active_weights_and_gce_q():
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        params = _params(weights={"kl": 0.0, "gce": 1.0}, gce_q=0.5)
        results = tmp_path / "results.jsonl"
        _write_jsonl(results, [_row(params)])
        history = load_result_history(results)

        assert find_completed_result(_params(weights={"kl": 0.0, "gce": 1.0}, gce_q=0.5), history)
        assert find_completed_result(_params(weights={"kl": 0.0, "gce": 1.0}, gce_q=0.7), history) is None
        assert find_completed_result(_params(weights={"kl": 1.0, "gce": 1.0}, gce_q=0.5), history) is None


def test_run_key_from_argparse_namespace_matches_dict_request():
    # An argparse `args` (weights as w_<name> attrs, no `weights` dict) must yield
    # the same identity as the dict-params request experiment.sh / run_sweep build.
    for weights, gce_q in [({"kl": 1.0}, 0.7), ({"kl": 0.0, "gce": 1.0}, 0.5)]:
        args_key = canonical_run_key(_args_namespace(weights, gce_q=gce_q))
        req_key = canonical_run_key(_params(weights=weights, gce_q=gce_q))
        assert args_key["weights"] == req_key["weights"], weights
        assert args_key["loss_hparams"] == req_key["loss_hparams"], weights
        assert run_hash(args_key) == run_hash(req_key), weights


def test_run_key_distinguishes_selector_settings():
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        params = _params(select_method="stratified", select_k=8)
        results = tmp_path / "results.jsonl"
        _write_jsonl(results, [_row(params)])
        history = load_result_history(results)

        assert find_completed_result(_params(select_method="stratified", select_k=8), history)
        assert find_completed_result(_params(select_method="stratified", select_k=16), history) is None
        assert find_completed_result(_params(select_method="relmatch"), history) is None


def test_diagnostics_request_requires_requested_ood_sets():
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        params = _params(diagnostics=True, ood_sets="svhn")
        results = tmp_path / "results.jsonl"
        _write_jsonl(results, [_row(_params(), diagnostics=False)])
        assert find_completed_result(params, load_result_history(results)) is None

        _write_jsonl(results, [_row(_params(), diagnostics=True)])
        assert find_completed_result(params, load_result_history(results)) is not None
        assert find_completed_result(_params(diagnostics=True, ood_sets="dtd"), load_result_history(results)) is None


def _write_sweep(path, results_file):
    path.write_text(
        f"""
defaults:
  dataset: cifar100
  arch: resnet18_modified
  seed: 42
  ipc: 10
  factor: 1
  num_crop: 5
  mipc: 300
  re_epochs: 300
  results_file: "{results_file}"
  weights:
    kl: 1.0

experiments:
  - name: stock
    select_method: stock
""".lstrip()
    )


def _run_sweep(config, *extra):
    return subprocess.run(
        [sys.executable, "scripts/run_sweep.py", str(config), "--dry-run", "--gpus", "0", *extra],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )


def test_run_sweep_dry_run_skips_completed_from_archive():
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        results = tmp_path / "results.jsonl"
        archive = tmp_path / "results-20260101.jsonl"
        _write_jsonl(archive, [_row(_params())])
        config = tmp_path / "sweep.yaml"
        _write_sweep(config, results)

        proc = _run_sweep(config)
        assert "[skip] stock:" in proc.stdout
        assert "0 scheduled, 1 skipped" in proc.stdout
        assert "rotation skipped" in proc.stdout


def test_run_sweep_force_dry_run_prints_command():
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        results = tmp_path / "results.jsonl"
        _write_jsonl(results, [_row(_params())])
        config = tmp_path / "sweep.yaml"
        _write_sweep(config, results)

        proc = _run_sweep(config, "--force")
        assert "Force mode: completed-cell skipping disabled." in proc.stdout
        assert "[dry-run]" in proc.stdout
        assert "--results-file" in proc.stdout


def test_selector_suffix_round_trips_through_exp_name():
    # selector_from_exp_name must invert selector_suffix for every keyed selector, so legacy rows
    # (no stored run_key) reconstruct the exact identity the encoder produced. Pins the encoder/decoder
    # pair: a token change on one side without the other would silently shift archived-row identity.
    prefix = "cifar100_conv3_f1_mipc300_ipc10_cr5"
    configs = [
        {"select_method": "stock"},
        {"select_method": "relmatch"},
        {"select_method": "reldist"},
        {"select_method": "stratified", "select_k": 16},
        {"select_method": "stratified", "select_k": 8},                     # default -> no _k token
        {"select_method": "momentmatch", "momentmatch_mean_weight": 0.5},
        {"select_method": "momentmatch", "momentmatch_mean_weight": 1.0},   # default -> no _mw token
        {"select_method": "qddpp", "select_beta": 0.5, "select_quality": "margin"},
        {"select_method": "qddpp", "select_beta": 0.0, "select_quality": "confidence"},
        {"select_method": "facloc", "facloc_space": "feature"},
        {"select_method": "facloc", "facloc_space": "softlabel"},           # default -> no _sp token
    ]
    for cfg in configs:
        suffix = selector_suffix(cfg)
        recovered = selector_from_exp_name(prefix + suffix)
        expected = selector_config(cfg)
        assert recovered == expected, f"{cfg}: {recovered} != {expected} (suffix {suffix!r})"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all tests passed")
