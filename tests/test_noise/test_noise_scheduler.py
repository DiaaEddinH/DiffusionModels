import math
import types
import torch
import pytest

from noise_scheduler import GeometricSchedule, LinearSchedule


def test_geometric_schedule_meaningful_properties_and_api():
    sigma_min = 0.05
    sigma_max = 5.0
    obj = GeometricSchedule(sigma_min=sigma_min, sigma_max=sigma_max)

    # Use float64 to make dtype propagation obvious
    t = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64)

    # stddev: starts at 0, strictly increasing on this grid, dtype/shape preserved
    std = obj.stddev(t)
    assert std.shape == t.shape
    assert std.dtype == t.dtype
    assert torch.isclose(std[0], torch.tensor(0.0, dtype=t.dtype), atol=1e-12)
    assert torch.all(std[1:] > std[:-1])

    # diffusion_coeff: endpoints at sigma_min and sigma_max
    d = obj.diffusion_coeff(t)
    assert d.shape == t.shape
    assert d.dtype == t.dtype
    assert torch.isclose(
        d[0], torch.tensor(sigma_min, dtype=t.dtype), rtol=1e-10, atol=0.0
    )
    assert torch.isclose(
        d[-1], torch.tensor(sigma_max, dtype=t.dtype), rtol=1e-10, atol=0.0
    )

    # Geometric progression check: log(d) should be affine in t
    # Differences between consecutive log(d) should be ~constant on a fine grid.
    logd = torch.log(d)
    diffs = logd[1:] - logd[:-1]
    assert torch.allclose(
        diffs, torch.full_like(diffs, diffs.mean()), rtol=1e-6, atol=1e-8
    )

    # get_mean_stddev: mean must be the original x object; std matches stddev(t)
    x = torch.randn(3, dtype=t.dtype)
    mean, std_pair = obj.mean_stddev(x, t)
    assert mean is x
    assert torch.allclose(std_pair, std)


def test_linear_schedule_meaningful_properties_and_api():
    sigma_min = 0.1
    sigma_max = 1.1

    obj = LinearSchedule(sigma_min=sigma_min, sigma_max=sigma_max)

    # Use float32 here to also exercise different dtype
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)

    # stddev: exact linear interpolation
    std = obj.stddev(t)
    expected_std = sigma_min + (sigma_max - sigma_min) * t
    assert std.shape == t.shape and std.dtype == t.dtype
    assert torch.allclose(std, expected_std, rtol=0.0, atol=0.0)

    # diffusion_coeff: constant equal to (sigma_max - sigma_min)
    d = obj.diffusion_coeff(t)
    expected_d = torch.full_like(t, sigma_max - sigma_min)
    assert d.shape == t.shape and d.dtype == t.dtype
    assert torch.allclose(d, expected_d, rtol=0.0, atol=0.0)

    # get_mean_stddev: passthrough + pairing with stddev(t)
    x = torch.randn(4, 3, dtype=t.dtype)
    mean, std_pair = obj.mean_stddev(x, t)
    assert mean is x
    assert torch.allclose(std_pair, std, rtol=0.0, atol=0.0)
