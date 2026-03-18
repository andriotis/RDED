from argument import args
from synthesize.main import main as synth_main
from validation.main import (
    main as valid_main,
)  # The relabel and validation are combined here for fast experiment
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. RDED experiments require a GPU. "
            "Check: python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
        )
    synth_main(args)
    valid_main(args)
