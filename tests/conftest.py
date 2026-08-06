import os
import random
from typing import Iterator

import numpy as np
import pytest
import torch

from diffusion_models.noise.noise_scheduler import Schedule

# Ensure a non-interactive matplotlib backend for all tests before pyplot is imported anywhere
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    # If matplotlib is not available in some environments, ignore
    pass


# --------------------------------------------------
# Dummy class definitions
# --------------------------------------------------

class DummySchedule(Schedule):
    def __init__(self, std_scale=1.0, eps=1e-3):
        super().__init__(arg_min=0, arg_max=0, eps=eps)
        self._std_scale = float(std_scale)
        self.diffusion_ts = None

    def diffusion_coeff(self, t: torch.Tensor):
        self.diffusion_ts = t.detach().clone()
        return self._std_scale *  torch.sqrt(2 * t)

    def drift_term(self, x, t):
        return -torch.ones_like(x)

    def stddev(self, t: torch.Tensor):
        return self._std_scale * t

    def mean_stddev(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # identically return input `x`
        d = (x.dim() - 1) * (None,)
        return x * torch.exp(-t)[:, *d], self.stddev(t)[:, *d]

    def invert_variance_to_time(self, variance_schedule):
        return variance_schedule.clamp(min=0).sqrt() / self._std_scale


class DummyModel(torch.nn.Module):
    def __init__(self, schedule: Schedule, device="cpu", output_scale: float = 0.8):
        super().__init__()
        self.device = torch.device(device)
        self.schedule = schedule
        self.eval_called = False
        self.seen_labels = None
        self.ot_ts = []
        self.output_scale = float(output_scale)

    def eval(self):  # track eval() usage
        self.eval_called = True
        return super().eval()

    def forward(self, x: torch.Tensor, t: torch.Tensor, *labels):
        # record labels and timesteps used by ot_sampler (called each step)
        self.seen_labels = labels
        # Collect scalar times used by ot_sampler for verification
        if isinstance(t, torch.Tensor) and t.ndim == 1 and t.numel() > 0:
            # t is a batch of identical scalars expanded; capture the scalar value
            self.ot_ts.append(float(t[0].detach().cpu()))
        # simple controlled velocity/drift proportional to x (keeps things stable)
        return self.output_scale * x


class DummyNetwork(torch.nn.Module):
    """
    Small trainable network for testing ScoreModel end-to-end. Has a
    single learnable parameter so EMA/train_step are doing something meaningful.
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
 
    def forward(self, x: torch.Tensor, t: torch.Tensor, *labels):
        return self.scale * x

# --------------------------------------------------
# Fixtures
# --------------------------------------------------

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


@pytest.fixture
def dummy_schedule():
    return DummySchedule(std_scale=2.0)

@pytest.fixture
def dummy_model(dummy_schedule):
    return DummyModel(dummy_schedule)

@pytest.fixture
def dummy_network():
    return DummyNetwork(scale=0.3)
