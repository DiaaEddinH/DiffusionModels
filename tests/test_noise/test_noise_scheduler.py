import torch

from diffusion_models.noise.noise_scheduler import GeometricSchedule, LinearSchedule


def test_geometric_schedule_meaningful_properties_and_api():
    sigma_min = 0.05
    sigma_max = 5.0
    num_steps = 11

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
    x = torch.randn(t.shape[0], 3, dtype=t.dtype)
    mean, std_pair = obj.mean_stddev(x, t)
    assert mean is x
    assert torch.allclose(std_pair, std[:, None])

    # build_uniform_variance_schedule
    time_schedule = obj.build_variance_schedule(num_steps=num_steps)
    assert time_schedule.shape[0] == num_steps
    assert torch.all(time_schedule[1:] < time_schedule[:-1])

    # drift_term: needs to be identically zero with same shape/device as input x
    drift = obj.drift_term(x, t)
    excepted_drift = torch.zeros_like(x)
    assert torch.allclose(drift, torch.zeros_like(x), rtol=1e-10, atol=1e-8)
    assert drift.shape == x.shape
    assert drift.dtype == x.dtype
    assert drift.device == x.device



def test_linear_schedule_meaningful_properties_and_api():
    beta_min = 0.1
    beta_max = 1.1
    num_steps = 11

    obj = LinearSchedule(beta_min=beta_min, beta_max=beta_max)

    # Use float32 here to also exercise different dtype
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)

    # schedule: exact linear interpolation
    schedule = obj.beta_schedule(t)
    expected_schedule = beta_min + (beta_max - beta_min) * t
    assert schedule.shape == t.shape and schedule.dtype == t.dtype
    assert torch.allclose(schedule, expected_schedule, rtol=0.0, atol=0.0)

    # diffusion_coeff: equal to sqrt(schedule)
    d = obj.diffusion_coeff(t)
    expected_d = torch.sqrt(expected_schedule)
    assert d.shape == t.shape and d.dtype == t.dtype
    assert torch.allclose(d, expected_d, rtol=0.0, atol=0.0)

    # mean_factor: equal to exp(-1/2 int^t_0 schedule(t))
    m = obj.mean_factor(t)
    expected_m = torch.exp(-0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t**2))
    assert m.shape == t.shape and m.dtype == t.dtype
    assert torch.allclose(m, expected_m, rtol=0.0, atol=0.0)
    assert torch.allclose(
        obj.mean_factor(torch.zeros(1)), torch.ones(1), rtol=0.0, atol=0.0
    )

    # stddev: equal to sqrt(1 - mean_factor^2)
    std = obj.stddev(t)
    expected_std = torch.sqrt(1 - m**2)
    assert obj.stddev(torch.zeros(1)) == torch.zeros(1)
    assert torch.allclose(
        obj.stddev(torch.zeros(1)), torch.zeros(1), rtol=0.0, atol=0.0
    )
    assert std.shape == t.shape and std.dtype == t.dtype
    assert torch.allclose(std, expected_std, rtol=0.0, atol=0.0)

    # get_mean_stddev: passthrough + pairing with stddev(t)
    x = torch.randn(t.shape[0], 4, 3, dtype=t.dtype)
    mean, std_pair = obj.mean_stddev(x, t)
    assert torch.allclose(mean, x * expected_m[:, None, None], rtol=0.0, atol=0.0)
    assert torch.allclose(std_pair, std[:, None, None], rtol=0.0, atol=0.0)

    # build_uniform_variance_schedule
    time_schedule = obj.build_variance_schedule(num_steps=num_steps)
    assert time_schedule.shape[0] == num_steps
    assert torch.all(time_schedule[1:] < time_schedule[:-1])

    # drift_term: needs to be identically zero with same shape/device as input x
    drift = obj.drift_term(x, t)
    excepted_drift = -0.5 * schedule[:, None, None] * x
    assert torch.allclose(drift, excepted_drift, rtol=1e-7, atol=1e-8)
    assert drift.dtype == x.dtype
    assert drift.device == x.device
