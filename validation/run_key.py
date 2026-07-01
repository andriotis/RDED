"""Canonical identity helpers for deterministic RDED experiment runs."""

import glob
import hashlib
import json
import os
import re

from validation.losses import LOSS_REGISTRY


DEFAULTS = {
    "factor": 1,
    "num_crop": 5,
    "mipc": 300,
    "re_epochs": 300,
    "gce_q": 0.7,
    "sce_log_floor": -4.0,
    "select_method": "stock",
    "select_k": 8,
    "select_beta": 0.0,
    "select_quality": "confidence",
    "facloc_space": "softlabel",
    "momentmatch_mean_weight": 1.0,
    "ood_dataset": "svhn",
    "fit_ipc": 50,
}


def _get(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict):
            if name in obj and obj[name] is not None:
                return obj[name]
        elif hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _as_int(value, default):
    if value is None or value == "":
        value = default
    return int(value)


def _as_float(value, default):
    if value is None or value == "":
        value = default
    return float(value)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_slug(value):
    return f"{float(value):g}"


def canonical_weights(weights=None):
    """Resolve LOSS_REGISTRY defaults overridden by the explicitly set weights."""
    weights = weights or {}
    return {
        name: float(weights.get(name, default))
        for name, (_, default) in LOSS_REGISTRY.items()
    }


def _resolve_weights(obj):
    """Weights from a ``weights`` dict (rows / sweep params) or per-term
    ``w_<name>`` attributes (argparse ``args``). Falls back to LOSS_REGISTRY
    defaults for anything unset."""
    explicit = _get(obj, "weights")
    if isinstance(explicit, dict):
        return canonical_weights(explicit)
    collected = {}
    for name in LOSS_REGISTRY:
        value = _get(obj, f"w_{name}")
        if value is not None:
            collected[name] = value
    return canonical_weights(collected)


def active_loss_hparams(obj, weights=None):
    weights = _resolve_weights(obj) if weights is None else canonical_weights(weights)
    hparams = {}
    if float(weights.get("gce", 0.0)) > 0.0:
        hparams["gce_q"] = _as_float(_get(obj, "gce_q"), DEFAULTS["gce_q"])
    if float(weights.get("rce", 0.0)) > 0.0:
        hparams["sce_log_floor"] = _as_float(
            _get(obj, "sce_log_floor"), DEFAULTS["sce_log_floor"]
        )
    return hparams


def selector_from_exp_name(exp_name):
    """Best-effort selector reconstruction for legacy rows."""
    if not exp_name or "_sel" not in exp_name:
        return {"method": "stock"}
    suffix = exp_name[exp_name.find("_sel") + 4 :]
    match = re.match(r"([A-Za-z0-9_]+?)(?:_|$)", suffix)
    method = match.group(1) if match else suffix
    # Parse the keyed params from the portion AFTER the method name, so a method whose name starts
    # with a key letter (e.g. "qddpp" vs the `_q` quality token) can't collide with the param regexes.
    params = suffix[len(method):]
    selector = {"method": method}
    if method == "qddpp":
        beta = re.search(r"(?:^|_)b([-+0-9.eE]+)", params)
        quality = re.search(r"(?:^|_)q([A-Za-z0-9_]+)", params)
        selector["select_beta"] = float(beta.group(1)) if beta else DEFAULTS["select_beta"]
        selector["select_quality"] = quality.group(1) if quality else DEFAULTS["select_quality"]
    elif method == "stratified":
        k = re.search(r"(?:^|_)k([0-9]+)", params)
        selector["select_k"] = int(k.group(1)) if k else DEFAULTS["select_k"]
    elif method == "momentmatch":
        mw = re.search(r"(?:^|_)mw([-+0-9.eE]+)", params)
        selector["momentmatch_mean_weight"] = (
            float(mw.group(1)) if mw else DEFAULTS["momentmatch_mean_weight"]
        )
    elif method == "facloc":
        sp = re.search(r"(?:^|_)sp([A-Za-z0-9]+)", params)
        selector["facloc_space"] = sp.group(1) if sp else DEFAULTS["facloc_space"]
    return selector


def selector_config(obj):
    explicit = _get(obj, "selector")
    if isinstance(explicit, dict):
        method = explicit.get("method", DEFAULTS["select_method"]) or DEFAULTS["select_method"]
        source = explicit
    else:
        method = _get(obj, "select_method", default=None)
        if method is None:
            legacy = selector_from_exp_name(_get(obj, "exp_name", default=""))
            method = legacy["method"]
            source = legacy
        else:
            source = obj

    method = str(method or DEFAULTS["select_method"])
    selector = {"method": method}
    if method == "stratified":
        selector["select_k"] = _as_int(_get(source, "select_k"), DEFAULTS["select_k"])
    elif method == "momentmatch":
        selector["momentmatch_mean_weight"] = _as_float(
            _get(source, "momentmatch_mean_weight"),
            DEFAULTS["momentmatch_mean_weight"],
        )
    elif method == "qddpp":
        selector["select_beta"] = _as_float(
            _get(source, "select_beta"), DEFAULTS["select_beta"]
        )
        selector["select_quality"] = str(
            _get(source, "select_quality", default=DEFAULTS["select_quality"])
        )
    elif method == "facloc":
        selector["facloc_space"] = str(
            _get(source, "facloc_space", default=DEFAULTS["facloc_space"])
        )
    return selector


def selector_suffix(obj):
    selector = selector_config(obj)
    method = selector["method"]
    if method == "stock":
        return ""

    suffix = f"_sel{method}"
    if method == "stratified" and selector["select_k"] != DEFAULTS["select_k"]:
        suffix += f"_k{selector['select_k']}"
    elif method == "momentmatch":
        mw = selector["momentmatch_mean_weight"]
        if abs(mw - DEFAULTS["momentmatch_mean_weight"]) > 1e-9:
            suffix += f"_mw{_float_slug(mw)}"
    elif method == "qddpp":
        suffix += f"_b{_float_slug(selector['select_beta'])}"
        if selector["select_quality"] != DEFAULTS["select_quality"]:
            suffix += f"_q{selector['select_quality']}"
    elif method == "facloc":
        if selector["facloc_space"] != DEFAULTS["facloc_space"]:
            suffix += f"_sp{selector['facloc_space']}"
    return suffix


def canonical_run_key(obj):
    """Return the stable identity for a completed deterministic training run."""
    dataset = str(_get(obj, "dataset", "subset"))
    arch = str(_get(obj, "arch", "arch_name"))
    stud = _get(obj, "stud_arch", "stud", "stud_name", default=arch)
    weights = _resolve_weights(obj)
    return {
        "version": 1,
        "dataset": dataset,
        "arch": arch,
        "stud": str(stud),
        "seed": _as_int(_get(obj, "seed"), 0),
        "ipc": _as_int(_get(obj, "ipc"), 0),
        "factor": _as_int(_get(obj, "factor"), DEFAULTS["factor"]),
        "num_crop": _as_int(_get(obj, "num_crop"), DEFAULTS["num_crop"]),
        "mipc": _as_int(_get(obj, "mipc"), DEFAULTS["mipc"]),
        "re_epochs": _as_int(_get(obj, "re_epochs"), DEFAULTS["re_epochs"]),
        "weights": weights,
        "loss_hparams": active_loss_hparams(obj, weights),
        "selector": selector_config(obj),
    }


def run_hash(run_key):
    payload = json.dumps(run_key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def diagnostics_config(obj):
    enabled = _as_bool(_get(obj, "diagnostics", default=False))
    ood_sets = []
    if enabled:
        raw = _get(obj, "ood_sets", default="")
        if raw:
            ood_sets = [s.strip() for s in str(raw).split(",") if s.strip()]
        else:
            ood_sets = [str(_get(obj, "ood_dataset", default=DEFAULTS["ood_dataset"]))]
    return {
        "enabled": enabled,
        "ood_sets": ood_sets,
        "fit_ipc": _as_int(_get(obj, "fit_ipc"), DEFAULTS["fit_ipc"]),
    }


def result_history_paths(results_file):
    """Return active JSONL plus same-stem archives, e.g. results-*.jsonl."""
    path = os.path.abspath(results_file)
    dirname = os.path.dirname(path) or "."
    basename = os.path.basename(path)
    if basename.endswith(".jsonl"):
        stem = basename[: -len(".jsonl")]
    else:
        stem = os.path.splitext(basename)[0]
    archive_pattern = os.path.join(dirname, f"{stem}-*.jsonl")
    paths = []
    if os.path.exists(path):
        paths.append(path)
    for candidate in sorted(glob.glob(archive_pattern)):
        if os.path.abspath(candidate) != path:
            paths.append(os.path.abspath(candidate))
    return paths


def effective_results_file(obj):
    return _get(obj, "results_file", default=os.path.join("logs", "results.jsonl"))


def row_is_complete(row):
    return row.get("best_top1") is not None


def row_has_requested_diagnostics(row, request):
    req = diagnostics_config(request)
    if not req["enabled"]:
        return True

    diag = row.get("diag")
    if not isinstance(diag, dict):
        return False
    if not req["ood_sets"]:
        return True

    ood = diag.get("ood")
    if isinstance(ood, dict):
        return all(name in ood for name in req["ood_sets"])
    return req["ood_sets"] == [DEFAULTS["ood_dataset"]]


def row_run_key(row):
    if isinstance(row.get("run_key"), dict):
        return row["run_key"]
    return canonical_run_key(row)


def load_result_history(results_file):
    rows = []
    for path in result_history_paths(results_file):
        try:
            with open(path) as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not row_is_complete(row):
                        continue
                    try:
                        key = row_run_key(row)
                    except Exception:
                        continue
                    rows.append(
                        {
                            "path": path,
                            "line_no": line_no,
                            "row": row,
                            "run_key": key,
                            "run_hash": row.get("run_hash") or run_hash(key),
                        }
                    )
        except OSError:
            continue
    return rows


def find_completed_result(request, history):
    key = canonical_run_key(request)
    key_hash = run_hash(key)
    for item in history:
        if item["run_hash"] != key_hash:
            continue
        if item["run_key"] != key:
            continue
        if row_has_requested_diagnostics(item["row"], request):
            return item
    return None
