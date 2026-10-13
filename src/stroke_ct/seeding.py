import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id, seed):
    """Only relevant if num_workers > 0. Use via functools.partial(worker_init_fn, seed=seed)."""
    ws = (seed + worker_id) % (2**32)
    np.random.seed(ws)
    random.seed(ws)