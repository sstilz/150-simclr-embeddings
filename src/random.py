import random

import numpy as np
import torch
from numpy.random import Generator


def get_rng(seed: int = -1) -> Generator:
    """
    References:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed == -1:
        seed = random.randint(0, int(1e6))

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    return np.random.default_rng(seed)
