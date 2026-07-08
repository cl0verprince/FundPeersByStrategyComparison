"""Seed every source of randomness from one config value, so runs are reproducible."""
import random

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
