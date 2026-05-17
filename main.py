from argument import args
from synthesize.main import main as synth_main
from validation.main import (
    main as valid_main,
)  # The relabel and validation are combined here for fast experiment
import os
import warnings

warnings.filterwarnings("ignore")


def _syn_data_populated(path):
    return os.path.isdir(path) and any(os.scandir(path))


if __name__ == "__main__":
    if args.skip_synth and _syn_data_populated(args.syn_data_path):
        print(f"--skip-synth: reusing existing {args.syn_data_path}")
    else:
        synth_main(args)
    valid_main(args)
