# tests/test_config_loader.py
import sys
import unittest
import argparse
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path


import pytest
import torch
import yaml

from typing import Optional
from diffusion_models.config.config import (
    Registry,
    YAMLConfig,
    ComponentConfig,
    ScoreModelConfig,
    TrainerConfig,
    RunConfig,
    ExperimentConfig,
    build_optimizer,
    build_lr_scheduler,
    OPTIMIZER_REGISTRY,
    LR_SCHEDULER_REGISTRY,
)


class Foo:
    def __init__(self, value=0):
        self.value = value


class Bar:
    pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_then_get_returns_same_class(self):
        reg = Registry("thing")
        reg.register("foo")(Foo)
        assert reg.get("foo") is Foo

    def test_register_as_decorator(self):
        reg = Registry("thing")

        @reg.register("foo")
        class Decorated:
            pass

        assert reg.get("foo") is Decorated

    def test_register_multiple_aliases_single_call(self):
        reg = Registry("thing")
        reg.register("foo", "f", "the_foo")(Foo)
        assert reg.get("foo") is Foo
        assert reg.get("f") is Foo
        assert reg.get("the_foo") is Foo

    def test_register_no_names_raises(self):
        reg = Registry("thing")
        with pytest.raises(ValueError):
            reg.register()

    def test_duplicate_name_raises(self):
        reg = Registry("thing")
        reg.register("foo")(Foo)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("foo")(Bar)

    def test_duplicate_within_same_multi_name_call_does_not_leak(self):
        reg = Registry("thing")
        reg.register("foo")(Foo)
        with pytest.raises(ValueError):
            reg.register("bar", "foo")(Bar)
        with pytest.raises(KeyError):
            reg.get("bar")

    def test_get_unknown_name_raises_with_available_list(self):
        reg = Registry("thing")
        reg.register("foo")(Foo)
        with pytest.raises(KeyError, match="foo"):
            reg.get("not_registered")

    def test_build_instantiates_with_params(self):
        reg = Registry("thing")
        reg.register("foo")(Foo)
        instance = reg.build("foo", {"value": 42})
        assert isinstance(instance, Foo)
        assert instance.value == 42

    def test_build_with_no_params_uses_defaults(self):
        reg = Registry("thing")
        reg.register("foo")(Foo)
        instance = reg.build("foo")
        assert instance.value == 0

    def test_unregister_removes_entry(self):
        reg = Registry("thing")
        reg.register("foo")(Foo)
        reg.unregister("foo")
        with pytest.raises(KeyError):
            reg.get("foo")

    def test_unregister_unknown_name_is_a_noop(self):
        reg = Registry("thing")
        reg.unregister("never_registered")  # should not raise

    def test_stacked_decorators_register_in_two_registries(self):
        reg_a = Registry("a")
        reg_b = Registry("b")

        @reg_a.register("x")
        @reg_b.register("x")
        class Both:
            pass

        assert reg_a.get("x") is Both
        assert reg_b.get("x") is Both


class TestPrebuiltRegistries:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
            ("sgd", torch.optim.SGD),
            ("rmsprop", torch.optim.RMSprop),
        ],
    )
    def test_builtin_optimizers_registered(self, name, expected):
        assert OPTIMIZER_REGISTRY.get(name) is expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("step", torch.optim.lr_scheduler.StepLR),
            ("cosine", torch.optim.lr_scheduler.CosineAnnealingLR),
            ("exponential", torch.optim.lr_scheduler.ExponentialLR),
        ],
    )
    def test_builtin_lr_schedulers_registered(self, name, expected):
        assert LR_SCHEDULER_REGISTRY.get(name) is expected


# ---------------------------------------------------------------------------
# YAMLConfig: _unwrap_optional
# ---------------------------------------------------------------------------


class TestUnwrapOptional:
    def test_pipe_none_syntax(self):
        assert YAMLConfig._unwrap_optional(int | None) is int

    def test_typing_optional(self):
        assert YAMLConfig._unwrap_optional(Optional[str]) is str

    def test_non_optional_type_passthrough(self):
        assert YAMLConfig._unwrap_optional(int) is int

    def test_ambiguous_union_passthrough_unchanged(self):
        # More than one non-None arm: nothing to unambiguously unwrap to,
        # so the original union is returned as-is.
        t = int | str
        assert YAMLConfig._unwrap_optional(t) == t

    def test_triple_union_with_none_passthrough_unchanged(self):
        t = str | torch.device | None
        assert YAMLConfig._unwrap_optional(t) == t


# ---------------------------------------------------------------------------
# YAMLConfig: to_dict / to_yaml
# ---------------------------------------------------------------------------


class TestToDict:
    def test_flat_dataclass(self):
        cfg = ComponentConfig(name="foo", params={"a": 1})
        assert cfg.to_dict() == {"name": "foo", "params": {"a": 1}}

    def test_recurses_into_nested_yamlconfig_fields(self):
        config = ExperimentConfig(
            network=ComponentConfig(name="unet"),
            schedule=ComponentConfig(name="geometric"),
            trainer=TrainerConfig(file_path="run"),
            run=RunConfig(N_epochs=10),
        )
        d = config.to_dict()
        # asdict() recurses automatically - nested dataclasses become plain
        # nested dicts, not dataclass instances.
        assert isinstance(d["network"], dict)
        assert d["network"] == {"name": "unet", "params": {}}
        assert d["model"] == {"decay_rate": 0.999, "device": None}


class TestToYaml:
    def test_writes_valid_yaml(self, tmp_path):
        cfg = ComponentConfig(name="foo", params={"a": 1})
        path = tmp_path / "out.yaml"
        cfg.to_yaml(path)
        with path.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded == {"name": "foo", "params": {"a": 1}}

    def test_roundtrip_via_from_yaml(self, tmp_path):
        cfg = ComponentConfig(name="foo", params={"a": 1, "b": "x"})
        path = tmp_path / "out.yaml"
        cfg.to_yaml(path)
        loaded = ComponentConfig.from_yaml(path)
        assert loaded == cfg

    def test_experiment_config_roundtrip(self, tmp_path):
        original = ExperimentConfig(
            network=ComponentConfig(name="unet", params={"in_channels": 3}),
            schedule=ComponentConfig(name="geometric"),
            trainer=TrainerConfig(file_path="run"),
            run=RunConfig(N_epochs=10),
            model=ScoreModelConfig(decay_rate=0.5, device="cuda"),
            lr_scheduler=ComponentConfig(name="step", params={"step_size": 5}),
        )
        path = tmp_path / "experiment.yaml"
        original.to_yaml(path)
        loaded = ExperimentConfig.from_yaml(path)
        assert loaded == original


# ---------------------------------------------------------------------------
# YAMLConfig: from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_builds_flat_dataclass(self):
        cfg = ComponentConfig.from_dict({"name": "foo", "params": {"a": 1}})
        assert cfg == ComponentConfig(name="foo", params={"a": 1})

    def test_missing_optional_field_uses_default(self):
        cfg = ComponentConfig.from_dict({"name": "foo"})
        assert cfg.params == {}

    def test_missing_required_field_raises_type_error(self):
        with pytest.raises(TypeError):
            TrainerConfig.from_dict({})

    def test_unknown_key_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown field name"):
            ComponentConfig.from_dict({"name": "foo", "not_a_real_field": 1})

    def test_unknown_key_suggests_close_match(self):
        # "nmae" is a typo of "name" - difflib should catch it.
        with pytest.raises(ValueError, match="did you mean"):
            ComponentConfig.from_dict({"nmae": "foo"})

    def test_recursively_decodes_nested_yamlconfig_field(self):
        data = {
            "network": {"name": "unet", "params": {"in_channels": 3}},
            "schedule": {"name": "geometric"},
            "trainer": {"file_path": "run"},
            "run": {"N_epochs": 10},
        }
        config = ExperimentConfig.from_dict(data)
        assert isinstance(config.network, ComponentConfig)
        assert config.network == ComponentConfig(name="unet", params={"in_channels": 3})
        assert isinstance(config.trainer, TrainerConfig)
        assert config.trainer.file_path == "run"

    def test_optional_nested_field_decoded_when_present(self):
        data = {
            "network": {"name": "unet"},
            "schedule": {"name": "geometric"},
            "trainer": {"file_path": "run"},
            "run": {"N_epochs": 10},
            "lr_scheduler": {"name": "step", "params": {"step_size": 5}},
        }
        config = ExperimentConfig.from_dict(data)
        assert isinstance(config.lr_scheduler, ComponentConfig)
        assert config.lr_scheduler.name == "step"

    def test_optional_nested_field_defaults_when_absent(self):
        data = {
            "network": {"name": "unet"},
            "schedule": {"name": "geometric"},
            "trainer": {"file_path": "run"},
            "run": {"N_epochs": 10},
        }
        config = ExperimentConfig.from_dict(data)
        assert config.lr_scheduler is None
        assert config.model == ScoreModelConfig()
        assert config.optimizer == ComponentConfig(name="adam", params={"lr": 2e-4})

    def test_optional_nested_field_explicit_null_does_not_crash(self):
        # Regression test: an Optional[YAMLConfig] field set to `null` in
        # YAML resolves to the non-None dataclass type via _unwrap_optional,
        # but the actual value is None - from_dict must not blindly call
        # `field_type.from_dict(None)`.
        data = {
            "network": {"name": "unet"},
            "schedule": {"name": "geometric"},
            "trainer": {"file_path": "run"},
            "run": {"N_epochs": 10},
            "lr_scheduler": None,
        }
        config = ExperimentConfig.from_dict(data)
        assert config.lr_scheduler is None

    def test_params_dict_field_is_not_recursively_decoded(self):
        # `params` is typed dict[str, Any], not a YAMLConfig - it should
        # stay a plain dict even though its value happens to be dict-shaped.
        cfg = ComponentConfig.from_dict({"name": "foo", "params": {"nested": {"a": 1}}})
        assert cfg.params == {"nested": {"a": 1}}
        assert isinstance(cfg.params["nested"], dict)


# ---------------------------------------------------------------------------
# YAMLConfig: from_yaml
# ---------------------------------------------------------------------------

MINIMAL_YAML = """
network:
  name: unet
  params:
    in_channels: 3
 
schedule:
  name: geometric
  params:
    sigma_min: 1.0
    sigma_max: 10.0
 
trainer:
  file_path: run001
 
run:
  N_epochs: 50
"""

FULL_YAML = (
    MINIMAL_YAML
    + """
model:
  decay_rate: 0.5
  device: cuda
 
optimizer:
  name: sgd
  params:
    lr: 0.01
 
lr_scheduler:
  name: step
  params:
    step_size: 10
"""
)


class TestExperimentConfigFromYaml:
    def test_returns_experiment_config(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_YAML)
        config = ExperimentConfig.from_yaml(path)
        assert isinstance(config, ExperimentConfig)

    def test_parses_required_sections(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_YAML)
        config = ExperimentConfig.from_yaml(path)
        assert config.network == ComponentConfig(name="unet", params={"in_channels": 3})
        assert config.schedule.params == {"sigma_min": 1.0, "sigma_max": 10.0}
        assert config.trainer.file_path == "run001"
        assert config.run.N_epochs == 50

    def test_optional_sections_fall_back_to_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_YAML)
        config = ExperimentConfig.from_yaml(path)
        assert config.model == ScoreModelConfig()
        assert config.optimizer == ComponentConfig(name="adam", params={"lr": 2e-4})
        assert config.lr_scheduler is None

    def test_optional_sections_used_when_present(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(FULL_YAML)
        config = ExperimentConfig.from_yaml(path)
        assert config.model.decay_rate == pytest.approx(0.5)
        assert config.model.device == "cuda"
        assert config.optimizer.name == "sgd"
        assert config.lr_scheduler.name == "step"

    def test_missing_required_section_raises_type_error(self, tmp_path):
        content = """
network:
  name: unet
schedule:
  name: geometric
"""
        path = tmp_path / "config.yaml"
        path.write_text(content)
        with pytest.raises(TypeError):
            ExperimentConfig.from_yaml(path)

    def test_unknown_top_level_key_raises(self, tmp_path):
        content = MINIMAL_YAML + "\nnot_a_real_section:\n  foo: 1\n"
        path = tmp_path / "config.yaml"
        path.write_text(content)
        with pytest.raises(ValueError, match="Unknown field name"):
            ExperimentConfig.from_yaml(path)

    def test_unknown_key_within_section_raises(self, tmp_path):
        content = (
            MINIMAL_YAML + "\ntrainer:\n  file_path: run001\n  not_a_real_field: 1\n"
        )
        path = tmp_path / "config.yaml"
        path.write_text(content)
        with pytest.raises(ValueError, match="Unknown field name"):
            ExperimentConfig.from_yaml(path)

    def test_accepts_path_as_string(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_YAML)
        config = ExperimentConfig.from_yaml(str(path))
        assert isinstance(config, ExperimentConfig)

    def test_batch_size_present_in_run_section(self, tmp_path):
        content = MINIMAL_YAML.replace("N_epochs: 50", "N_epochs: 50\n  batch_size: 16")
        path = tmp_path / "config.yaml"
        path.write_text(content)
        config = ExperimentConfig.from_yaml(path)
        assert config.run.batch_size == 16

    def test_batch_size_defaults_when_omitted(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_YAML)
        config = ExperimentConfig.from_yaml(path)
        assert config.run.batch_size == 32


# ---------------------------------------------------------------------------
# ExperimentConfig.extra
# ---------------------------------------------------------------------------


class TestExperimentConfigExtra:
    def test_defaults_to_empty_dict(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(MINIMAL_YAML)
        config = ExperimentConfig.from_yaml(path)
        assert config.extra == {}

    def test_populated_from_yaml(self, tmp_path):
        content = MINIMAL_YAML + "\nextra:\n  dataset: mnist\n  num_workers: 4\n"
        path = tmp_path / "config.yaml"
        path.write_text(content)
        config = ExperimentConfig.from_yaml(path)
        assert config.extra == {"dataset": "mnist", "num_workers": 4}

    def test_arbitrary_keys_do_not_raise_unlike_other_sections(self, tmp_path):
        # No schema validation applies inside `extra` - any keys are fine,
        # unlike every other section (which would raise on an unknown key).
        content = (
            MINIMAL_YAML
            + "\nextra:\n  totally_made_up_key: 123\n  another.weird-one: xyz\n"
        )
        path = tmp_path / "config.yaml"
        path.write_text(content)
        config = ExperimentConfig.from_yaml(path)
        assert config.extra["totally_made_up_key"] == 123

    def test_nested_dict_inside_extra_stays_a_plain_dict(self, tmp_path):
        content = (
            MINIMAL_YAML + "\nextra:\n  dataset:\n    name: mnist\n    augment: true\n"
        )
        path = tmp_path / "config.yaml"
        path.write_text(content)
        config = ExperimentConfig.from_yaml(path)
        assert config.extra["dataset"] == {"name": "mnist", "augment": True}

    def test_set_directly_via_from_dict(self):
        data = {
            "network": {"name": "unet"},
            "schedule": {"name": "geometric"},
            "trainer": {"file_path": "run"},
            "run": {"N_epochs": 10},
            "extra": {"seed": 42},
        }
        config = ExperimentConfig.from_dict(data)
        assert config.extra == {"seed": 42}

    def test_roundtrips_via_to_yaml_from_yaml(self, tmp_path):
        original = ExperimentConfig(
            network=ComponentConfig(name="unet"),
            schedule=ComponentConfig(name="geometric"),
            trainer=TrainerConfig(file_path="run"),
            run=RunConfig(N_epochs=10),
            extra={"dataset": "mnist", "num_workers": 4},
        )
        path = tmp_path / "roundtrip.yaml"
        original.to_yaml(path)
        loaded = ExperimentConfig.from_yaml(path)
        assert loaded == original
        assert loaded.extra == {"dataset": "mnist", "num_workers": 4}

    def test_separate_instances_do_not_share_extra_dict(self):
        a = ExperimentConfig(
            network=ComponentConfig(name="unet"),
            schedule=ComponentConfig(name="geometric"),
            trainer=TrainerConfig(file_path="run_a"),
            run=RunConfig(N_epochs=10),
        )
        b = ExperimentConfig(
            network=ComponentConfig(name="unet"),
            schedule=ComponentConfig(name="geometric"),
            trainer=TrainerConfig(file_path="run_b"),
            run=RunConfig(N_epochs=10),
        )
        a.extra["x"] = 1
        assert b.extra == {}


# ---------------------------------------------------------------------------
# Config dataclass defaults
# ---------------------------------------------------------------------------


class TestComponentConfig:
    def test_defaults_to_empty_params(self):
        assert ComponentConfig(name="foo").params == {}

    def test_separate_instances_do_not_share_params_dict(self):
        a = ComponentConfig(name="a")
        b = ComponentConfig(name="b")
        a.params["x"] = 1
        assert b.params == {}


class TestScoreModelConfig:
    def test_defaults(self):
        cfg = ScoreModelConfig()
        assert cfg.decay_rate == pytest.approx(0.999)
        assert cfg.device is None


class TestTrainerConfig:
    def test_requires_file_path(self):
        with pytest.raises(TypeError):
            TrainerConfig()

    def test_defaults(self):
        cfg = TrainerConfig(file_path="run001")
        assert cfg.use_ddp is False
        assert cfg.log_dir == "logs/runs"
        assert cfg.save_weight_history is False
        assert cfg.weight_history_frequency == 10
        assert cfg.weight_dir == "./data/weights"
        assert cfg.checkpoint_dir == "./data/checkpoints"
        assert cfg.metadata_csv_path == "./data/run_metadata.csv"


class TestRunConfig:
    def test_requires_n_epochs(self):
        with pytest.raises(TypeError):
            RunConfig()

    def test_defaults(self):
        cfg = RunConfig(N_epochs=100)
        assert cfg.batch_size == 32
        assert cfg.early_stopping == 10
        assert cfg.min_delta == pytest.approx(1e-4)

    def test_batch_size_is_overridable(self):
        cfg = RunConfig(N_epochs=100, batch_size=64)
        assert cfg.batch_size == 64

    def test_batch_size_from_dict(self):
        cfg = RunConfig.from_dict({"N_epochs": 10, "batch_size": 16})
        assert cfg.batch_size == 16


# ---------------------------------------------------------------------------
# build_optimizer / build_lr_scheduler (MAY DELETE THESE!!!!)
# ---------------------------------------------------------------------------


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


class TestBuildOptimizer:
    def test_builds_correct_optimizer_type(self):
        model = TinyModel()
        cfg = ComponentConfig(name="sgd", params={"lr": 0.05})
        optimizer = build_optimizer(model, cfg)
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_optimizer_tracks_model_parameters(self):
        model = TinyModel()
        optimizer = build_optimizer(
            model, ComponentConfig(name="adam", params={"lr": 1e-3})
        )
        assert optimizer.param_groups[0]["params"][0] is model.w

    def test_unknown_optimizer_name_raises(self):
        model = TinyModel()
        with pytest.raises(KeyError):
            build_optimizer(model, ComponentConfig(name="not_a_real_optimizer"))


class TestBuildLrScheduler:
    def test_returns_none_when_config_is_none(self):
        model = TinyModel()
        optimizer = build_optimizer(
            model, ComponentConfig(name="sgd", params={"lr": 0.1})
        )
        assert build_lr_scheduler(optimizer, None) is None

    def test_builds_correct_scheduler_type(self):
        model = TinyModel()
        optimizer = build_optimizer(
            model, ComponentConfig(name="sgd", params={"lr": 0.1})
        )
        cfg = ComponentConfig(name="step", params={"step_size": 5})
        scheduler = build_lr_scheduler(optimizer, cfg)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_unknown_scheduler_name_raises(self):
        model = TinyModel()
        optimizer = build_optimizer(
            model, ComponentConfig(name="sgd", params={"lr": 0.1})
        )
        with pytest.raises(KeyError):
            build_lr_scheduler(optimizer, ComponentConfig(name="not_a_real_scheduler"))
