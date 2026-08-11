import pytest
from diffusion_models.config.config_cli import (
    _set_nested,
    parse_config,
    build_arg_parser,
    _apply_overrides,
    _extras_to_overrides,
    _parse_override_value,
)
from diffusion_models.config.config import ExperimentConfig


class TestSetNested:
    def test_cretes_new_nested_path(self):
        data = {}
        _set_nested(data, "a.b.c", 42)
        assert data == {"a": {"b": {"c": 42}}}

    def test_preserves_sibling_keys(self):
        data = {"network": {"name": "unet", "params": {"in_channels": 3}}}
        _set_nested(data, "network.params.hidden_dim", 128)
        assert data["network"]["name"] == "unet"
        assert data["network"]["params"] == {"in_channels": 3, "hidden_dim": 128}

    def test_overwrites_existing_leaf(self):
        data = {"trainer": {"file_path": "old_run"}}
        _set_nested(data, "trainer.file_path", "new_run")
        assert data["trainer"]["file_path"] == "new_run"

    def test_single_level_key_no_dots(self):
        data = {}
        _set_nested(data, "foo", "bar")
        assert data == {"foo": "bar"}

    def test_raises_when_intermediate_component_is_not_a_dict(self):
        data = {"network": {"name": "unet"}}
        with pytest.raises(ValueError, match="not a section"):
            _set_nested(data, "network.name.extra", "x")

    def test_deeply_nested_path(self):
        data = {}
        _set_nested(data, "a.b.c.d.e", "deep")
        assert data == {"a": {"b": {"c": {"d": {"e": "deep"}}}}}


class TestParseOverrideValue:
    def test_int(self):
        value = _parse_override_value("50")
        assert value == 50
        assert isinstance(value, int)

    def test_float(self):
        value = _parse_override_value("0.001")
        assert value == pytest.approx(0.001)
        assert isinstance(value, float)

    def test_bool_true(self):
        assert _parse_override_value("true") is True

    def test_bool_false(self):
        assert _parse_override_value("false") is False

    def test_null(self):
        assert _parse_override_value("null") is None

    def test_plain_string(self):
        assert _parse_override_value("run002") == "run002"

    def test_string_that_looks_like_a_path_stays_a_string(self):
        assert _parse_override_value("./data/checkpoints") == "./data/checkpoints"

    def test_negative_number(self):
        value = _parse_override_value("-5")
        assert value == -5
        assert isinstance(value, int)


class TestApplyOverrides:
    def test_applies_single_override(self):
        raw = {"trainer": {"file_path": "old"}}
        result = _apply_overrides(raw, ["trainer.file_path=new"])
        assert result["trainer"]["file_path"] == "new"

    def test_applies_multiple_overrides(self):
        raw = {"trainer": {"file_path": "old"}, "run": {"N_epochs": 10}}
        result = _apply_overrides(raw, ["trainer.file_path=new", "run.N_epochs=99"])
        assert result["trainer"]["file_path"] == "new"
        assert result["run"]["N_epochs"] == 99

    def test_creates_new_section_not_in_base_config(self):
        raw = {"trainer": {"file_path": "run"}}
        result = _apply_overrides(raw, ["lr_scheduler.name=cosine"])
        assert result["lr_scheduler"] == {"name": "cosine"}

    def test_no_overrides_leaves_config_unchanged(self):
        raw = {"trainer": {"file_path": "run"}}
        result = _apply_overrides(raw, [])
        assert result == {"trainer": {"file_path": "run"}}

    def test_missing_equals_sign_raises(self):
        with pytest.raises(ValueError, match=r"KEY\.PATH=VALUE"):
            _apply_overrides({}, ["trainer.file_path"])

    def test_returns_the_same_dict_object(self):
        raw = {}
        result = _apply_overrides(raw, ["a.b=1"])
        assert result is raw

    def test_value_containing_equals_sign_is_split_on_first_only(self):
        # e.g. --set trainer.file_path=run=with=equals
        raw = {}
        result = _apply_overrides(raw, ["trainer.file_path=run=with=equals"])
        assert result["trainer"]["file_path"] == "run=with=equals"


# ---------------------------------------------------------------------------
# _extras_to_overrides
# ---------------------------------------------------------------------------
 
class TestExtrasToOverrides:
    def test_key_value_pair_form(self):
        assert _extras_to_overrides(["--dataset", "mnist"]) == ["extra.dataset=mnist"]
 
    def test_key_equals_value_form(self):
        assert _extras_to_overrides(["--num_workers=4"]) == ["extra.num_workers=4"]
 
    def test_bare_flag_with_nothing_after_becomes_true(self):
        assert _extras_to_overrides(["--verbose"]) == ["extra.verbose=true"]
 
    def test_bare_flag_followed_by_another_flag_becomes_true(self):
        result = _extras_to_overrides(["--verbose", "--dataset", "mnist"])
        assert result == ["extra.verbose=true", "extra.dataset=mnist"]
 
    def test_dotted_key_supported(self):
        assert _extras_to_overrides(["--dataset.name", "mnist"]) == ["extra.dataset.name=mnist"]
 
    def test_multiple_key_value_pairs(self):
        result = _extras_to_overrides(["--dataset", "mnist", "--num_workers", "4"])
        assert result == ["extra.dataset=mnist", "extra.num_workers=4"]
 
    def test_negative_number_value_not_mistaken_for_a_flag(self):
        assert _extras_to_overrides(["--offset", "-1"]) == ["extra.offset=-1"]
 
    def test_empty_list_returns_empty_list(self):
        assert _extras_to_overrides([]) == []
 
    def test_token_without_leading_dashes_raises(self):
        with pytest.raises(ValueError, match="Unrecognized argument"):
            _extras_to_overrides(["dataset", "mnist"])
 
    def test_mixed_forms_together(self):
        result = _extras_to_overrides(["--dataset=mnist", "--num_workers", "4", "--verbose"])
        assert result == ["extra.dataset=mnist", "extra.num_workers=4", "extra.verbose=true"]
 


class TestBuildArgParser:
    def test_config_default(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.config == "configs/example_config.yaml"

    def test_config_long_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--config", "my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_config_short_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-c", "my_config.yaml"])
        assert args.config == "my_config.yaml"

    def test_overrides_default_to_empty_list(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.overrides == []

    def test_set_is_repeatable(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--set", "a=1", "--set", "b=2"])
        assert args.overrides == ["a=1", "b=2"]

    def test_set_short_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["-s", "a=1"])
        assert args.overrides == ["a=1"]


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


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_YAML)
    return path


class TestParseConfigEndToEnd:
    def test_loads_config_with_no_overrides(self, config_path):
        config = parse_config(["--config", str(config_path)])
        assert isinstance(config, ExperimentConfig)
        assert config.trainer.file_path == "run001"
        assert config.run.N_epochs == 50
        assert config.network.name == "unet"

    def test_short_flag_for_config(self, config_path):
        config = parse_config(["-c", str(config_path)])
        assert config.trainer.file_path == "run001"

    def test_single_override_applied(self, config_path):
        config = parse_config(
            ["--config", str(config_path), "--set", "trainer.file_path=overridden"]
        )
        assert config.trainer.file_path == "overridden"

    def test_multiple_overrides_across_sections(self, config_path):
        config = parse_config(
            [
                "--config",
                str(config_path),
                "--set",
                "run.N_epochs=99",
                "--set",
                "network.params.in_channels=64",
            ]
        )
        assert config.run.N_epochs == 99
        assert config.network.params["in_channels"] == 64
        # fields not touched by an override still reflect the base file
        assert config.trainer.file_path == "run001"

    def test_short_flag_for_set(self, config_path):
        config = parse_config(["--config", str(config_path), "-s", "run.N_epochs=7"])
        assert config.run.N_epochs == 7

    def test_override_adds_optional_section_not_in_base_yaml(self, config_path):
        config = parse_config(
            [
                "--config",
                str(config_path),
                "--set",
                "lr_scheduler.name=cosine",
                "--set",
                "lr_scheduler.params.T_max=100",
            ]
        )
        assert config.lr_scheduler.name == "cosine"
        assert config.lr_scheduler.params == {"T_max": 100}

    def test_override_type_coercion_bool_and_float(self, config_path):
        config = parse_config(
            [
                "--config",
                str(config_path),
                "--set",
                "trainer.use_ddp=true",
                "--set",
                "run.min_delta=0.0001",
            ]
        )
        assert config.trainer.use_ddp is True
        assert config.run.min_delta == pytest.approx(0.0001)

    def test_missing_config_file_exits_with_error(self, tmp_path, capsys):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(SystemExit) as exc_info:
            parse_config(["--config", str(missing)])
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_default_config_path_missing_exits_with_error(self, tmp_path, monkeypatch):
        # No --config given; the default relative path won't exist under
        # the test's (empty) working directory.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            parse_config([])
        assert exc_info.value.code == 2

    def test_unknown_key_in_override_raises_with_suggestion(self, config_path):
        # "file_pth" is a typo of "file_path" - should surface the same
        # did-you-mean suggestion as a malformed key directly in the YAML.
        with pytest.raises(ValueError, match="did you mean"):
            parse_config(
                ["--config", str(config_path), "--set", "trainer.file_pth=typo"]
            )

    def test_malformed_override_raises(self, config_path):
        with pytest.raises(ValueError, match=r"KEY\.PATH=VALUE"):
            parse_config(["--config", str(config_path), "--set", "no_equals_sign"])

    def test_override_does_not_mutate_file_on_disk(self, config_path):
        original_content = config_path.read_text()
        parse_config(
            ["--config", str(config_path), "--set", "trainer.file_path=changed"]
        )
        assert config_path.read_text() == original_content

    def test_no_overrides_matches_plain_from_yaml(self, config_path):
        via_cli = parse_config(["--config", str(config_path)])
        via_direct = ExperimentConfig.from_yaml(config_path)
        assert via_cli == via_direct


class TestParseConfigBatchSize:
    def test_default_batch_size(self, config_path):
        config = parse_config(["--config", str(config_path)])
        assert config.run.batch_size == 32
 
    def test_override_batch_size(self, config_path):
        config = parse_config(
            ["--config", str(config_path), "--set", "run.batch_size=64"]
        )
        assert config.run.batch_size == 64
 
 
class TestParseConfigSampler:
    def test_default_sampler(self, config_path):
        config = parse_config(["--config", str(config_path)])
        assert config.sampler.name == "euler_maruyama"
        assert config.sampler.params == {}
 
    def test_override_sampler_name_and_nested_params(self, config_path):
        config = parse_config([
            "--config", str(config_path),
            "--set", "sampler.name=stochastic_heun",
            "--set", "sampler.params.num_steps=100",
            "--set", "sampler.params.S_churn=40.0",
        ])
        assert config.sampler.name == "stochastic_heun"
        assert config.sampler.params == {"num_steps": 100, "S_churn": 40.0}
 
 
class TestParseConfigExtras:
    def test_unrecognized_flag_captured_into_extra(self, config_path):
        config = parse_config(["--config", str(config_path), "--dataset", "mnist"])
        assert config.extra == {"dataset": "mnist"}
 
    def test_multiple_unrecognized_flags(self, config_path):
        config = parse_config([
            "--config", str(config_path),
            "--dataset", "mnist",
            "--num_workers", "4",
        ])
        assert config.extra == {"dataset": "mnist", "num_workers": 4}
 
    def test_bare_flag_becomes_boolean_true(self, config_path):
        config = parse_config(["--config", str(config_path), "--augment"])
        assert config.extra == {"augment": True}
 
    def test_equals_form_extra_flag(self, config_path):
        config = parse_config(["--config", str(config_path), "--num_workers=8"])
        assert config.extra == {"num_workers": 8}
 
    def test_dotted_extra_flag_nests(self, config_path):
        config = parse_config([
            "--config", str(config_path),
            "--dataset.name", "mnist",
            "--dataset.augment", "true",
        ])
        assert config.extra == {"dataset": {"name": "mnist", "augment": True}}
 
    def test_set_overrides_and_bare_extras_both_apply(self, config_path):
        config = parse_config([
            "--config", str(config_path),
            "--set", "trainer.file_path=overridden",
            "--dataset", "mnist",
        ])
        assert config.trainer.file_path == "overridden"
        assert config.extra == {"dataset": "mnist"}
 
    def test_yaml_extra_section_and_cli_extras_merge(self, tmp_path):
        content = MINIMAL_YAML + "\nextra:\n  dataset: cifar10\n  seed: 1\n"
        path = tmp_path / "config.yaml"
        path.write_text(content)
 
        config = parse_config(["--config", str(path), "--num_workers", "4"])
        assert config.extra == {"dataset": "cifar10", "seed": 1, "num_workers": 4}
 
    def test_cli_extra_overrides_yaml_extra_for_same_key(self, tmp_path):
        content = MINIMAL_YAML + "\nextra:\n  dataset: cifar10\n"
        path = tmp_path / "config.yaml"
        path.write_text(content)
 
        config = parse_config(["--config", str(path), "--dataset", "mnist"])
        assert config.extra == {"dataset": "mnist"}
 
    def test_no_extras_given_leaves_extra_empty(self, config_path):
        config = parse_config(["--config", str(config_path)])
        assert config.extra == {}
 
 
class TestAbbreviationDisabled:
    """Regression tests for allow_abbrev=False: since unrecognized flags are
    now meaningful (they become extras), an accidental prefix match against
    --config/--set would silently do the wrong thing rather than error."""
 
    def test_partial_flag_name_is_not_treated_as_set(self, config_path):
        # "--se" would be an unambiguous abbreviation of "--set" if
        # allow_abbrev were left at its (default) True - it must NOT be
        # treated that way here.
        config = parse_config(["--config", str(config_path), "--se", "trainer.file_path=hijacked"])
        # "trainer.file_path=hijacked" should land in extra.se, NOT actually
        # override trainer.file_path.
        assert config.trainer.file_path == "run001"
        assert config.extra == {"se": "trainer.file_path=hijacked"}
 
    def test_partial_flag_name_is_not_treated_as_config(self, tmp_path, monkeypatch):
        # "--co" would be an unambiguous abbreviation of "--config" if
        # allow_abbrev were left at its default. Forcing an empty cwd makes
        # this deterministic regardless of where the suite is actually run
        # from (rather than assuming the default config path just happens
        # not to exist).
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            parse_config(["--co", "somewhere.yaml"])
