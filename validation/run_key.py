"""Canonical run identifier shared by experiment.sh, curve_logger, and analysis tools.

A run is identified by the tuple (dataset, arch, stud, ipc, seed, weights). The
filename slug and the log-row key must agree, so the same function builds both.
"""

from validation.losses import LOSS_REGISTRY


def _slug(w):
    """Filename-safe representation of a float weight."""
    return f"{w:g}".replace(".", "p").replace("-", "neg")


def canonical_weights(user_weights):
    """LOSS_REGISTRY defaults, overridden by anything in user_weights."""
    return {
        name: float(user_weights.get(name, default))
        for name, (_, default) in LOSS_REGISTRY.items()
    }


def weights_tag(canon):
    """Stable filename slug of the nonzero entries of a canonical weights dict."""
    nz = sorted((n, w) for n, w in canon.items() if w > 0)
    return "_".join(f"{n}{_slug(w)}" for n, w in nz) or "noloss"


def canonical_run_key(dataset, arch, stud, ipc, seed, user_weights):
    """Return (key, tag, canon).

    - canon: full LOSS_REGISTRY-keyed dict of weights (defaults overridden by user_weights).
    - tag:   "kl1p0_ockl0p3" style slug of nonzero entries (or "noloss").
    - key:   "{dataset}_{arch}_to_{stud}_ipc{ipc}_seed{seed}_w{tag}".
    """
    canon = canonical_weights(user_weights)
    tag = weights_tag(canon)
    key = f"{dataset}_{arch}_to_{stud}_ipc{int(ipc)}_seed{int(seed)}_w{tag}"
    return key, tag, canon
