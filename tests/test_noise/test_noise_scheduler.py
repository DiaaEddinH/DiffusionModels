import math
import torch
import pytest

from diffusion_models.noise.noise_scheduler import Schedule, GeometricSchedule, LinearSchedule

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
 
@pytest.fixture
def geo_schedule():
    return GeometricSchedule(sigma_min=1.0, sigma_max=10.0, eps=1e-3)

@pytest.fixture
def linear_schedule():
    return LinearSchedule(beta_min=0.02, beta_max=10.0, eps=1e-3)

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class TestScheduleAbstract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Schedule(arg_min=0.0, arg_max=1.0)

    def test_init_sets_bounds_and_eps(self, geo_schedule):
        assert geo_schedule.arg_min == 1.0
        assert geo_schedule.arg_max == 10.0
        assert geo_schedule.eps == 1e-3

    def test_default_eps(self):
        s = GeometricSchedule(sigma_min=1.0, sigma_max=10.0)
        assert s.eps == 1e-3

# ---------------------------------------------------------------------------
# build_variance (uses GeometricSchedule but it tests shared logic from base class)
# ---------------------------------------------------------------------------

class TestBuildVarianceSchedule:
    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_output_length(self, geo_schedule, schedule_type):
        num_steps = 12
        ts = geo_schedule.build_variance_schedule(num_steps, schedule_type)
        assert ts.shape == (num_steps,)

    def test_rejects_invalid_schedule_type(self, geo_schedule):
        with pytest.raises(AssertionError):
            geo_schedule.build_variance_schedule(10, schedule_type="not_a_real_schedule")

    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_monotonically_decreasing(self, geo_schedule, schedule_type):
        ts = geo_schedule.build_variance_schedule(20, schedule_type)
        diffs = ts[1:] - ts[:-1]
        assert (diffs <= 1e-6).all()

    def test_karras_rho_changes_spacing(self, geo_schedule):
        ts_rho1 = geo_schedule.build_variance_schedule(10, schedule_type="karras", rho=1.0)
        ts_rho7 = geo_schedule.build_variance_schedule(10, schedule_type="karras", rho=7.0)

        # Different rho should produce different intermediate spacing.
        # Endpoints are (approximately) fixed.
        assert not torch.allclose(ts_rho1[1:-1], ts_rho7[1:-1], atol=1e-4)

# ---------------------------------------------------------------------------
# GeometricSchedule (Variance expanding)
# ---------------------------------------------------------------------------

class TestGeometricSchedule:
    def test_stddev_at_t0_is_zero(self, geo_schedule):
        t = torch.tensor([0.0])
        assert geo_schedule.stddev(t).item() == pytest.approx(0.0, abs=1e-6)

    def test_stddev_at_t1_equals_sigma_min_scaled(self, geo_schedule):
        t = torch.tensor([1.0])
        alpha = geo_schedule.arg_max / geo_schedule.arg_min
        L = math.log(alpha)
        expected = geo_schedule.arg_min * math.sqrt((alpha**2 - 1) / (2 * L))
        assert geo_schedule.stddev(t).item() == pytest.approx(expected, rel=1e-4)

    def test_stddev_monotonically_increasing_in_t(self, geo_schedule):
        t = torch.linspace(0, 1, steps=20)
        s = geo_schedule.stddev(t)
        assert (s[1:] >= s[:-1]).all()

    def test_diffusion_coeff_correct_at_endpoints(self, geo_schedule):
        t0, t1 = torch.tensor([0.0, 1.0])
        assert geo_schedule.diffusion_coeff(t0).item() == pytest.approx(geo_schedule.arg_min)
        assert geo_schedule.diffusion_coeff(t1).item() == pytest.approx(geo_schedule.arg_max)

    def test_drift_term_is_zero(self, geo_schedule):
        x = torch.randn(5, 3)
        t = torch.rand(5)
        drift = geo_schedule.drift_term(x, t)
        assert torch.equal(drift, torch.zeros_like(x))

    def test_mean_stddev_mean_passthrough(self, geo_schedule):
        x = torch.randn(4, 6)
        t = torch.rand(4)
        mean, _ = geo_schedule.mean_stddev(x, t)
        assert torch.equal(mean, x)

    def test_mean_stddev_broadcasts_over_extra_dims(self, geo_schedule):
        x = torch.randn(4, 6, 7)
        t = torch.rand(4)
        _, std = geo_schedule.mean_stddev(x, t)
        assert std.shape == (4, 1, 1)
        assert torch.allclose(std.squeeze(), geo_schedule.stddev(t))

    def test_invert_variance_to_time_roundtrip(self, geo_schedule):
        t_original = torch.linspace(0.01, 1.0, steps=25)
        variance = geo_schedule.stddev(t_original) ** 2
        t_recovered = geo_schedule.invert_variance_to_time(variance)
        assert torch.allclose(t_recovered, t_original, atol=1e-4)

# ---------------------------------------------------------------------------
# LinearSchedule (Variance preserving)
# ---------------------------------------------------------------------------

class TestLinearSchedule:
    def test_beta_schedule_at_endpoints(self, linear_schedule):
        t0, t1 = torch.tensor([0.0, 1.0])
        assert linear_schedule.beta_schedule(t0).item() == pytest.approx(linear_schedule.arg_min)
        assert linear_schedule.beta_schedule(t1).item() == pytest.approx(linear_schedule.arg_max)

    def test_mean_factor_at_t0_is_one(self, linear_schedule):
        t0 = torch.tensor([0.0])
        assert linear_schedule.mean_factor(t0).item() == pytest.approx(1.0, abs=1e-6)

    def test_mean_factor_decreasing_with_t(self, linear_schedule):
        t = torch.linspace(0., 1., steps=20)
        m = linear_schedule.mean_factor(t)
        assert (m[1:] <= m[:-1] + 1e-8).all()

    def test_stddev_at_t0_is_zero(self, linear_schedule):
        t0 = torch.tensor([0.0])
        assert linear_schedule.stddev(t0).item() == pytest.approx(0.0, abs=1e-6)

    def test_stddev_bounded_between_zero_and_one(self, linear_schedule):
        t = torch.linspace(0., 1., steps=20)
        s = linear_schedule.stddev(t)
        assert (s >= -1e-6).all()
        assert (s <= 1 + 1e-6).all()

    def test_diffusion_coeff_is_sqrt_beta(self, linear_schedule):
        t = torch.rand(10)
        expected = torch.sqrt(linear_schedule.beta_schedule(t))
        assert torch.allclose(linear_schedule.diffusion_coeff(t), expected)

    def test_drift_term_matches_formula(self, linear_schedule):
        x = torch.randn(5, 3)
        t = torch.rand(5)
        excepted = -0.5 * linear_schedule.beta_schedule(t)[:, None] * x
        assert torch.allclose(linear_schedule.drift_term(x, t), excepted)

    def test_drift_zero_when_x_zero(self, linear_schedule):
        x = torch.zeros(4, 2)
        t = torch.rand(4)
        assert torch.equal(linear_schedule.drift_term(x, t), torch.zeros_like(x))

    def test_mean_stddev_matches_mean_factor_and_stddev(self, linear_schedule):
        x = torch.zeros(4, 6)
        t = torch.rand(4)
        mean, std = linear_schedule.mean_stddev(x, t)
        expected_mean = linear_schedule.mean_factor(t)[:, None] * x
        expected_std = linear_schedule.stddev(t)[:, None]

        assert torch.allclose(mean, expected_mean)
        assert torch.allclose(std, expected_std)

    def test_invert_variance_to_time_roundtrip(self, linear_schedule):
        t_original = torch.linspace(0.01, 1.0, steps=25)
        variance = linear_schedule.stddev(t_original) ** 2
        t_recovered = linear_schedule.invert_variance_to_time(variance)
        assert torch.allclose(t_recovered, t_original, atol=1e-4)
