import os
import sys
import warnings

from argument import args
from synthesize.main import main as synth_main
from validation.main import (
    main as valid_main,
)  # The relabel and validation are combined here for fast experiment
from validation.utils import seed_everything

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    seed_everything(args.seed)
    if args.skip_synth:
        if not os.path.isdir(args.syn_data_path) or not os.listdir(args.syn_data_path):
            sys.exit(
                f"--skip-synth requires existing syn_data at {args.syn_data_path}; "
                "run once without --skip-synth first."
            )
    else:
        synth_main(args)
        if args.synth_only:
            sys.exit(0)
    valid_main(args)
