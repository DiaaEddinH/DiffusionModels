import os
import random
from typing import Iterator

import numpy as np
import pytest
import torch

# Ensure a non-interactive matplotlib backend for all tests before pyplot is imported anywhere
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    # If matplotlib is not available in some environments, ignore
    pass


@pytest.fixture(autouse=True)
def seed_rng() -> Iterator[None]:
    """Seed NumPy, Python, and Torch RNGs for each test for reproducibility.

    Individual tests may still override seeds explicitly; this just provides
    a stable baseline and removes duplicated seeding code across files.
    """
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some operations rely on deterministic algorithms where available
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    yield
