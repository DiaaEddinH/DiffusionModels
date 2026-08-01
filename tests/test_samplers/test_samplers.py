import torch

import pytest

from diffusion_models.noise.noise_scheduler import Schedule
from diffusion_models.sampling.samplers import em_sampler, ot_sampler


class DummySchedule(Schedule):
    def __init__(self, drift_scale=0.0, diff_coeff=1.0, std=1.0):
        super().__init__(arg_min=0, arg_max=0)
        self.drift_scale = float(drift_scale)
        self._diff = float(diff_coeff)
        self._std = float(std)
        self.diffusion_ts = None

    def diffusion_coeff(self, t: torch.Tensor):
        # record timesteps for em sampler (a single tensor)
        self.diffusion_ts = t.detach().clone()
        return torch.full_like(t, self._diff)

    def drift_term(self, x, t):
        return torch.zeros_like(x)

    def stddev(self, t: torch.Tensor):
        # constant stddev, shape follows input t
        return torch.full_like(t, self._std)

    def mean_stddev(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # identically return input `x`
        d = (x.dim() - 1) * (None,)
        return x, self.stddev(t)[:, *d]


class DummyModel(torch.nn.Module):
    def __init__(self, schedule: Schedule, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.schedule = schedule
        self.eval_called = False
        self.seen_labels = None
        self.ot_ts = []

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
        return self.schedule.drift_scale * x


def test_ot_sampler_history_and_timesteps_and_no_grad_and_labels():
    schedule = DummySchedule(drift_scale=0.0)
    model = DummyModel(device="cpu", schedule=schedule)  # v(x,t)=0 => x stays constant

    batch, dims = 2, 5
    num_steps = 6
    labels = ("cond", torch.tensor([9]))

    hist = ot_sampler(model, (batch, dims), num_steps, *labels, history=True)

    # eval() should have been called
    assert model.eval_called is True

    # labels should be captured
    assert model.seen_labels == labels

    # history shape and requires_grad properties
    assert hist.shape == (num_steps, batch, dims)
    assert hist.requires_grad is False  # @torch.no_grad ensures no grad

    # all frames are identical when velocity is zero
    assert torch.allclose(hist, hist[0].expand_as(hist))

    # timesteps passed to the model should be linspace(0, 1, num_steps)
    assert len(model.ot_ts) == num_steps
    assert torch.allclose(
        torch.tensor(model.ot_ts),
        torch.linspace(0.0, 1.0, num_steps),
        atol=1e-7,
    )

    # Also test history=False path returns correct shape and no grad
    final = ot_sampler(model, (batch, dims), num_steps, *labels, history=False)
    assert isinstance(final, torch.Tensor)
    assert final.shape == (batch, dims)
    assert final.requires_grad is False


@pytest.mark.parametrize("device", ["cpu"])  # keep CPU only in CI
@pytest.mark.parametrize("history", [False, True])
def test_em_sampler_shapes_and_device_and_eval_and_labels(device, history):
    schedule = DummySchedule(drift_scale=0.0, diff_coeff=1.0, std=2.0)
    model = DummyModel(device=device, schedule=schedule)

    batch, dims = 4, 3
    num_steps = 7
    eps = 1e-3
    labels = (torch.tensor([1, 2, 3]), "class-A")

    out = em_sampler(model, (batch, dims), num_steps, *labels, history=history, eps=eps)

    # eval() should have been called
    assert model.eval_called is True

    # labels should have been recorded on model
    assert model.seen_labels == labels

    # diffusion_coeff should receive the full linspace(1, eps, num_steps)
    assert model.schedule.diffusion_ts is not None
    assert model.schedule.diffusion_ts.shape == (num_steps,)
    assert torch.allclose(
        model.schedule.diffusion_ts,
        torch.linspace(1, eps, num_steps, device=model.device),
    )

    if history:
        # history returns [num_steps, batch, dims]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (num_steps, batch, dims)
        assert out.device.type == torch.device(device).type

        # with drift=0 and g_t=1, last step has t==eps => test_noise is zero, so last two frames must be equal
        assert torch.allclose(out[-1], out[-2])
    else:
        # no history returns just the final tensor with requested shape
        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch, dims)
        assert out.device.type == torch.device(device).type

    # --- Deterministic drift check (no extra test; keeps suite minimal) ---
    # Disable test_noise by setting eps = 1.0 and use nonzero drift; with g_t=1 the update becomes:
    #   x_{k+1} = x_k + step_size * (drift_scale * x_k) = (1 + drift_scale/num_steps) * x_k
    schedule = DummySchedule(drift_scale=0.5, diff_coeff=1.0, std=2.0)
    drift_only = DummyModel(device=device, schedule=schedule)
    hist = em_sampler(
        drift_only, (batch, dims), num_steps, *labels, history=True, eps=1.0
    )
    c = 1.0 + drift_only.schedule.drift_scale / num_steps
    # Check geometric progression between consecutive frames: x_{k+1} == c * x_k
    # Use a strict equality; math is exact in our DummyModel path.
    assert torch.allclose(hist[1], hist[0] * c)
    assert torch.allclose(hist[-1], hist[0] * (c ** (hist.shape[0] - 1)))
