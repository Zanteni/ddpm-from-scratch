"""
Reproducibility helper. torch, numpy, and Python's random module each keep
their own independent random state, so all three need seeding separately
to get fully reproducible results. The cudnn flags additionally disable
some GPU-side non-determinism that persists even after seeding.
"""

import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False