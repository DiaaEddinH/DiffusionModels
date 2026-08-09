import torch
import pytest

from torch import Tensor
from torch.nn import Module, Parameter
from diffusion_models.models.models import ExponentialMovingAverage


class OneParamModel(Module):
    """Single learnable scalar, so `ExponentialMovingAverage` can be verified by hand."""

    def __init__(self, value: float):
        super().__init__()
        self.w = Parameter(torch.tensor(float(value)))

    def forward(self, x: Tensor) -> Tensor:
        return self.w * x


@pytest.fixture
def model():
    return OneParamModel(1.0)


@pytest.fixture
def exponential_moving_average(model):
    return ExponentialMovingAverage(model, decay_rate=0.9)


class TestInit:
    def test_shadow_seeded_from_initial_params(
        self, model: Module, exponential_moving_average: ExponentialMovingAverage
    ):
        assert torch.allclose(exponential_moving_average.shadow["w"], model.w.data)

    def test_only_tracks_params_which_require_grad(self):
        model = OneParamModel(1.0)
        model.w.requires_grad_(False)
        exponential_moving_average = ExponentialMovingAverage(model)
        assert exponential_moving_average.shadow == {}

    def test_shadow_is_a_copy_not_a_view(
        self, model: Module, exponential_moving_average: ExponentialMovingAverage
    ):
        exponential_moving_average.shadow["w"].add_(100.0)
        assert not torch.allclose(model.w.data, exponential_moving_average.shadow["w"])


class TestUpdate:
    def test_matches_closed_form(
        self, model: Module, exponential_moving_average: ExponentialMovingAverage
    ):
        old_shadow = exponential_moving_average.shadow["w"].clone()
        with torch.no_grad():
            model.w.data.fill_(5.0)
        exponential_moving_average.update()
        expected = (
            1 - exponential_moving_average.decay_rate
        ) * 5.0 + exponential_moving_average.decay_rate * old_shadow
        assert torch.allclose(exponential_moving_average.shadow["w"], expected)

    def test_repeated_updates_converge_toward_current_param(self, model: Module):
        ema = ExponentialMovingAverage(model, decay_rate=0.5)
        with torch.no_grad():
            model.w.data.fill_(10.0)
        for _ in range(50):
            ema.update()
        assert ema.shadow["w"].item() == pytest.approx(10.0, abs=1e-6)

    def test_skips_frozen_parameters(self):
        model = OneParamModel(1.0)
        exponential_moving_average = ExponentialMovingAverage(
            model
        )  # w is trainable at construction then frozen
        model.w.requires_grad_(False)
        with torch.no_grad():
            model.w.data.fill_(99.0)
        exponential_moving_average.update()
        assert exponential_moving_average.shadow["w"].item() == pytest.approx(1.0)

    def test_shadow_does_not_require_grad_after_update(
        self, model: Module, exponential_moving_average: ExponentialMovingAverage
    ):
        with torch.no_grad():
            model.w.data.fill_(3.0)
        exponential_moving_average.update()
        assert exponential_moving_average.shadow["w"].requires_grad is False


class TestApplyShadowRestore:
    def test_apply_shadow_swaps_in_moving_average_weights(
        self, model, exponential_moving_average
    ):
        with torch.no_grad():
            model.w.data.fill_(7.0)
        exponential_moving_average.update()
        shadow_val = exponential_moving_average.shadow["w"].item()
        exponential_moving_average.apply_shadow()
        assert model.w.item() == pytest.approx(shadow_val)

    def test_restore_returns_original_weights(self, model, exponential_moving_average):
        original = model.w.item()
        exponential_moving_average.apply_shadow()
        exponential_moving_average.restore()
        assert model.w.item() == pytest.approx(original)

    def test_restore_clears_backup(self, model, exponential_moving_average):
        exponential_moving_average.apply_shadow()
        exponential_moving_average.restore()
        assert exponential_moving_average.backup == {}

    def test_apply_shadow_backup_is_a_copy(self, model, exponential_moving_average):
        exponential_moving_average.apply_shadow()
        model.w.data.fill_(123.0)
        # Mutating the live parameter after apply_shadow must not retroactively change the backup
        assert exponential_moving_average.backup["w"].item() != pytest.approx(123.0)


class TestAverageParameterContextManager:
    def test_swaps_in_and_restores_normal_on_exit(
        self, model, exponential_moving_average
    ):
        with torch.no_grad():
            model.w.data.fill_(42.0)
        exponential_moving_average.update()
        shadow_val = exponential_moving_average.shadow["w"].item()

        with exponential_moving_average.average_parameters():
            assert model.w.item() == pytest.approx(shadow_val)

        assert model.w.item() == pytest.approx(42.0)

    def test_restores_if_exception_raised(self, model, exponential_moving_average):
        with torch.no_grad():
            model.w.data.fill_(42.0)

        with pytest.raises(RuntimeError):
            with exponential_moving_average.average_parameters():
                raise RuntimeError()

        assert model.w.item() == pytest.approx(42.0)


class TestStateDict:
    def test_state_dict_returns_shadow(self, exponential_moving_average):
        assert (
            exponential_moving_average.state_dict() is exponential_moving_average.shadow
        )

    def test_load_state_dict_replaces_shadow_values(self, exponential_moving_average):
        exponential_moving_average.load_state_dict({"w": torch.tensor(4811.0)})
        assert exponential_moving_average.shadow["w"].item() == pytest.approx(4811.0)

    def test_load_state_dict_clones_tensors(self, exponential_moving_average):
        source = {"w": torch.tensor(4811.0)}
        exponential_moving_average.load_state_dict(source)
        source["w"].fill_(0.0)
        assert exponential_moving_average.shadow["w"].item() == pytest.approx(4811.0)


class TestToDevice:
    def test_to_dtype_changes_shadow_and_backup_dtype(self, exponential_moving_average):
        exponential_moving_average.apply_shadow()
        exponential_moving_average.to(dtype=torch.float64)
        assert exponential_moving_average.shadow["w"].dtype == torch.float64
        assert exponential_moving_average.backup["w"].dtype == torch.float64

    def test_to_returns_self(self, exponential_moving_average):
        assert (
            exponential_moving_average.to(dtype=torch.float32)
            is exponential_moving_average
        )

    def test_to_does_not_overwrite_backup_with_shadow(
        self, model, exponential_moving_average
    ):
        with torch.no_grad():
            model.w.data.fill_(123.0)
        exponential_moving_average.update()
        exponential_moving_average.apply_shadow()

        assert exponential_moving_average.shadow["w"].item() != pytest.approx(123.0)
        assert exponential_moving_average.backup["w"].item() == pytest.approx(123.0)

        exponential_moving_average.to(dtype=torch.float64)

        assert exponential_moving_average.backup["w"].item() == pytest.approx(123.0)
        assert exponential_moving_average.backup["w"].item() != pytest.approx(
            exponential_moving_average.shadow["w"].item()
        )
