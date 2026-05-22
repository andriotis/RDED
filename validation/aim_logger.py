"""Aim companion to results_logger.

results.jsonl stays the source of truth (one row per finished run). Aim adds
just the loss curves needed for debugging training:
  - total train loss   — name='loss', context={subset:'train'}
  - total val loss     — name='loss', context={subset:'val'} (sparse cadence)
  - per-term raw train loss — name='loss', context={subset:'train', term:<name>}.
    Covers both active terms (w_<name> > 0) and monitor terms passed via
    --monitor. Values are raw (unweighted); the weighted contribution is
    `raw * weights[<name>]` reconstructable from run['hparams']['weights'].

Run identity:
  - experiment = args.exp_name  (groups runs from same synth config)
  - run.name   = args.run_tag (or seed+weights slug if empty)
  - run['hparams']  — args + sweep metadata, written once at start
  - run['summary']  — best_top1, final_top1, written at end

top1/top5 and NC metrics stay in results.jsonl only — not surfaced in Aim.
"""

from aim import Run

from validation.losses import LOSS_REGISTRY


def _slug(w):
    return f"{float(w):g}".replace(".", "p").replace("-", "neg")


def _default_run_tag(args):
    weights = sorted(
        (n, float(getattr(args, f"w_{n}"))) for n in LOSS_REGISTRY
    )
    nz = [(n, w) for n, w in weights if w > 0]
    tag = "_".join(f"{n}{_slug(w)}" for n, w in nz) or "noloss"
    return f"seed{args.seed}_w{tag}"


class AimLogger:
    def __init__(self, args):
        self._args = args
        self.enabled = not getattr(args, "disable_aim", False)
        if not self.enabled:
            self.run = None
            return
        self.run = Run(
            repo=getattr(args, "aim_repo", "./logs/aim"),
            experiment=args.exp_name,
        )
        self.run.name = getattr(args, "run_tag", "") or _default_run_tag(args)
        self.run["hparams"] = {
            "dataset": str(args.subset),
            "arch": str(args.arch_name),
            "stud": str(args.stud_name),
            "ipc": int(args.ipc),
            "mipc": int(args.mipc),
            "factor": int(args.factor),
            "num_crop": int(args.num_crop),
            "re_epochs": int(args.re_epochs),
            "seed": int(args.seed),
            "sweep_name": str(getattr(args, "sweep_name", "") or ""),
            "cell_id": str(getattr(args, "cell_id", "") or ""),
            "weights": {n: float(getattr(args, f"w_{n}")) for n in LOSS_REGISTRY},
        }

    def log_train(self, epoch, loss, components=None):
        if not self.enabled:
            return
        self.run.track(float(loss), name="loss", step=epoch, context={"subset": "train"})
        if components:
            for name, raw_avg in components.items():
                self.run.track(
                    float(raw_avg),
                    name="loss",
                    step=epoch,
                    context={"subset": "train", "term": name},
                )

    def log_val(self, epoch, loss):
        if not self.enabled:
            return
        self.run.track(float(loss), name="loss", step=epoch, context={"subset": "val"})

    def log_hparams(self, args, best_top1, final_top1):
        if not self.enabled:
            return
        self.run["summary"] = {
            "best_top1": float(best_top1),
            "final_top1": float(final_top1),
        }

    def close(self):
        if self.run is not None:
            self.run.close()
            self.run = None
