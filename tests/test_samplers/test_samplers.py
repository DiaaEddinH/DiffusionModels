import torch

import pytest

from diffusion_models.noise.noise_scheduler import GeometricSchedule, LinearSchedule
from diffusion_models.sampling.samplers import ot_sampler
from diffusion_models.sampling.samplers import BaseSampler, EulerMaruyamaSampler
from tests.conftest import DummyModel, DummySchedule


# --------------- Fixtures ---------------
@pytest.fixture
def sampler(dummy_model):
    return EulerMaruyamaSampler(dummy_model)


@pytest.fixture(params=["geometric", "linear"])
def real_schedule(request):
    if request.param == "geometric":
        return GeometricSchedule(sigma_min=1.0, sigma_max=10.0, eps=1e-3)
    return LinearSchedule(beta_min=0.02, beta_max=10.0, eps=1e-3)


@pytest.fixture
def real_model(real_schedule):
    return DummyModel(real_schedule)


@pytest.fixture
def real_sampler(real_model):
    return EulerMaruyamaSampler(real_model)


# --------------- BaseSampler construction ---------------


class TestBaseSampler:
    def test_cannot_instantiate_abstract_base(self, dummy_model):
        with pytest.raises(TypeError):
            BaseSampler(dummy_model)

    def test_init_sets_attributes_from_model(
        self, sampler, dummy_model, dummy_schedule
    ):
        assert sampler.model is dummy_model
        assert sampler.device == dummy_model.device
        assert sampler.schedule is dummy_schedule
        assert sampler.eps == dummy_schedule.eps

    def test_score_delegates_to_model_forward(self, sampler, dummy_model):
        x = torch.randn(3, 2)
        t = torch.randn(3)
        labels = (torch.tensor([1, 2, 3]), "label_a")

        out = sampler._score(x, t, *labels)
        assert dummy_model.seen_labels == labels
        assert torch.equal(out, dummy_model.output_scale * x)

    def test_init_history_shape_and_device(self, sampler):
        hist = sampler.init_history(num_steps=5, shape=(4, 3), keep_history=True)
        assert hist.shape == (5, 4, 3)
        assert hist.device == sampler.device

    def test_record_writes_when_flag_true(self, sampler):
        hist = sampler.init_history(2, (2, 2), keep_history=True)
        x = torch.ones(2, 2)
        sampler.record(hist, x, idx=1, flag=True)
        assert torch.equal(hist[1], x)

    def test_record_noop_when_flag_false(self, sampler):
        hist = sampler.init_history(2, (2, 2), keep_history=False)
        sampler.record(hist, torch.ones(2, 2), idx=0, flag=False)
        assert hist is None

    def test_collect_returns_history_when_flag_true(self, sampler):
        hist = sampler.init_history(2, (2, 2))
        x = torch.randn(2, 2)
        assert sampler.collect(hist, x, flag=True) is hist

    def test_collect_returns_x_when_flag_false(self, sampler):
        hist = sampler.init_history(2, (2, 2))
        x = torch.randn(2, 2)
        assert sampler.collect(hist, x, flag=False) is x


# --------------- EulerMaruyamaSampler.init_sample ---------------


class TestInitSampler:
    def test_shape(self, sampler):
        x0 = sampler.init_sample((5, 3))
        assert x0.shape == (5, 3)

    def test_scaled_by_stddev_at_t_equals_one(self, sampler, monkeypatch):
        # Force randn to return ones so we can check the stddev scaling
        monkeypatch.setattr(torch, "randn", lambda *a, **kw: torch.ones(*a))
        x0 = sampler.init_sample((4,))
        expected_std = sampler.schedule.stddev(torch.ones(1))
        assert torch.allclose(x0, expected_std * torch.ones(4))

    def test_real_schedule_init_sample_matches_stddev_at_t1(self, real_sampler):
        torch.manual_seed(0)
        shape = (10_000,)
        x0 = real_sampler.init_sample(shape)
        expected_std = real_sampler.schedule.stddev(torch.ones(1)).item()
        # Statistical check
        assert x0.std().item() == pytest.approx(expected_std, rel=1e-2)


# --------------- EulerMaruyamaSampler.update_step ---------------
class TestUpdateStep:
    def test_euler_maruyama_formula(self, sampler, monkeypatch):
        x = torch.zeros(3)
        drift = torch.full((3,), 2.0)
        step_size = 0.01
        step_size_sqrt = 0.1

        fixed_noise = torch.tensor([1.0, -1.0, 0.5])
        monkeypatch.setattr(torch, "randn_like", lambda t: fixed_noise)

        out = sampler.update_step(x, drift, step_size, step_size_sqrt)
        expected = x + drift * step_size + step_size_sqrt * fixed_noise
        assert torch.allclose(out, expected)

    def test_zero_step_size_sqrt_is_deterministic(self, sampler):
        x = torch.zeros(3)
        drift = torch.full((3,), 2.0)
        out = sampler.update_step(x, drift, step_size=0.1, step_size_sqrt=0.0)
        assert torch.allclose(out, x + 0.1 * drift)


# --------------- EulerMaruyamaSampler.build_schedule ---------------
class TestBuildSchedule:
    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_output_shapes(self, sampler, schedule_type):
        num_steps = 10
        timesteps, g2_t, step_size, step_size_sqrt = sampler.build_schedule(
            num_steps, schedule_type=schedule_type, rho=1.0
        )
        for tensor in (timesteps, g2_t, step_size, step_size_sqrt):
            assert tensor.shape == (num_steps,)

    def test_last_dt_repastes_second_to_last(self, sampler):
        _, _, step_size, _ = sampler.build_schedule(8)
        assert torch.isclose(step_size[-1], step_size[-2])

    def test_dt_matches_consecutive_timestep_diffs(self, sampler):
        timesteps, _, step_size, _ = sampler.build_schedule(8)
        expected = timesteps[:-1] - timesteps[1:]
        assert torch.allclose(step_size[:-1], expected)

    def test_step_size_sqrt_is_scaled_by_diffusion_coeff(self, sampler):
        timesteps, g2_t, step_size, step_size_sqrt = sampler.build_schedule(8)
        g_t = g2_t.sqrt()
        assert torch.allclose(step_size_sqrt, g_t * step_size.clamp(min=0) ** 0.5)

    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_timesteps_from_one_to_eps(self, real_sampler, schedule_type):
        timesteps, _, _, _ = real_sampler.build_schedule(
            30, schedule_type=schedule_type
        )
        assert timesteps[0].item() == pytest.approx(1.0, abs=1e-2)
        assert timesteps[-1].item() == pytest.approx(real_sampler.eps, abs=1e-2)

    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_timesteps_monotonically_decreasing(self, real_sampler, schedule_type):
        timesteps, _, _, _ = real_sampler.build_schedule(
            30, schedule_type=schedule_type
        )
        assert (timesteps[1:] <= timesteps[:-1] + 1e-8).all()

    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_timesteps_monotontest_step_sizes_nonnegative(
        self, real_sampler, schedule_type
    ):
        _, _, step_size, step_size_sqrt = real_sampler.build_schedule(
            30, schedule_type=schedule_type
        )
        assert (step_size >= -1e-8).all()
        assert (step_size_sqrt >= -1e-8).all()

    def test_g2_matches_schedule_diffusion_coeff_squared(self, real_sampler):
        timesteps, g2_t, _, _ = real_sampler.build_schedule(15)
        expected = real_sampler.schedule.diffusion_coeff(timesteps) ** 2
        assert torch.allclose(g2_t, expected)


# --------------- sample() ---------------


class TestSample:
    def test_output_shape_without_history(self, sampler):
        out = sampler.sample((4, 3), num_steps=5, keep_history=False)
        assert out.shape == (4, 3)

    def test_output_shape_with_history(self, sampler):
        out = sampler.sample((4, 3), num_steps=5, keep_history=True)
        assert out.shape == (5, 4, 3)

    def test_model_eval_is_called(self, sampler, dummy_model):
        sampler.sample((2, 2), num_steps=3)
        assert dummy_model.eval_called is True

    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_runs_for_all_schedule_types(self, sampler, schedule_type):
        out = sampler.sample((2, 2), num_steps=4, schedule_type=schedule_type, rho=2.0)
        assert out.shape == (2, 2)
        assert torch.isfinite(out).all()

    def test_final_step_uses_zero_diffusion_when_t_below_eps(
        self, sampler, dummy_model, monkeypatch
    ):
        dummy_model.schedule.eps = 0.5
        sampler.eps = 0.5

        seen_sqrt = []
        orig_update = sampler.update_step

        def spy_update(x, drift, step_size, step_size_sqrt):
            seen_sqrt.append(step_size_sqrt)
            return orig_update(x, drift, step_size, step_size_sqrt)

        sampler.update_step = spy_update
        sampler.sample((2, 2), num_steps=5)
        assert any(v == 0 for v in seen_sqrt)

    def test_no_nan_or_inf_in_output(self, sampler):
        out = sampler.sample((8, 4), num_steps=20)
        assert torch.isfinite(out).all()


class TestSampleRealSchedule:
    def test_runs_and_produces_finite_ouptut(self, real_sampler):
        out = real_sampler.sample((4, 3), num_steps=25)
        assert out.shape == (4, 3)
        assert torch.isfinite(out).all()

    def test_history_shape_matches_num_steps(self, real_sampler):
        out = real_sampler.sample((4, 3), num_steps=10, keep_history=True)
        assert out.shape == (10, 4, 3)

    @pytest.mark.parametrize("schedule_type", ["uniform", "log", "karras"])
    def test_all_schedule_types_run(self, real_sampler, schedule_type):
        out = real_sampler.sample((3, 2), num_steps=12, schedule_type=schedule_type)
        assert torch.isfinite(out).all()


def test_ot_sampler_history_and_timesteps_and_no_grad_and_labels():
    schedule = DummySchedule()
    model = DummyModel(
        device="cpu", schedule=schedule, output_scale=0.0
    )  # v(x,t)=0 => x stays constant

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
