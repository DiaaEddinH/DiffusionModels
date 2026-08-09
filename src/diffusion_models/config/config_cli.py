import yaml
import argparse


from typing import Any
from pathlib import Path
from diffusion_models.config.config import ExperimentConfig


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any):
    """
    Set `data[a][b]...[z] = value` for a dotted key "a.b...z", creating intermediate dicts as needed.

    :param data: A nested dictionary
    :type data: dict[str, Any]
    :param dotted_key: A dotted key which maps the keys of the nested dict to a value
    :type dotted_key: str
    :param value: A dictionary value/parameter
    :type value: Any

    :raises ValueError: If an intermediate path component exists but isn't a dict (eg when trying to set "network.name.foo" when "name" is a string).
    """
    *parents, leaf = dotted_key.split(".")
    target = data
    for part in parents:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(
                f"Cannot set '{dotted_key}: '{part}' is not a section "
                f"(got {type(target).__name__})"
            )
    target[leaf] = value


def _parse_override_value(raw: str) -> Any:
    """
    Coerces a CLI override's string value using YAML's own scalar rules, so
    `--set run.N_epochs=50` becomes an int, `--set trainer.use_ddp=true` becomes a bool,
    `--set model.device=null` becomes None, etc.

    :param raw: CLI override string value
    :type raw: str
    :return: _description_
    :rtype: Any
    """
    return yaml.load(raw, Loader=ExperimentConfig.yaml_loader)


def _apply_overrides(
    raw_config: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY.PATH=VALUE, got: {item!r}")
        dotted_key, _, value_str = item.partition("=")
        _set_nested(raw_config, dotted_key.strip(), _parse_override_value(value_str))
    return raw_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load an experiment config, with optional dotted-path overrides."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/example_config.yaml",
        help="Path to the YAML experiment config.",
    )
    parser.add_argument(
        "--set",
        "-s",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Override a config value, eg --set trainer.file_path=run001. Repeatable.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> ExperimentConfig:
    """
    Parse `--config`/`--set` CLI arguments and return the resulting `ExperimentConfig` with any overrides applied.

    :param argv: Argument list to parse, defaults to sys.argv[1:] (via argparse) if None
    :type argv: list[str] | None, optional
    :return: The parsed, ExperimentConfig with overrides
    :rtype: ExperimentConfig
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config file '{config_path}' not found.")

    with config_path.open("r") as fp:
        raw_config = yaml.load(fp, Loader=ExperimentConfig.yaml_loader) or {}

    raw_config = _apply_overrides(raw_config, args.overrides)
    return ExperimentConfig.from_dict(raw_config)


if __name__ == "__main__":

    config = parse_config()
    print(config)
    # args = parser.parse_args()
    # config_path = Path(args.config)
    # if not config_path.exists():
    #     parser.error(f"Config file '{config_path}' not found.")

    # with config_path.open("r") as fp:
    #     raw_config = yaml.load(fp, Loader=ExperimentConfig.yaml_loader) or {}

    # raw_config = _apply_overrides(raw_config, args.overrides)

    # print(ExperimentConfig.from_dict(raw_config))

    # yaml_path = "configs/newexample_config.yaml"
    # config = ExperimentConfig.from_yaml(yaml_path)
    # overrides = ["run.N_epochs=50",]
    # print(config.run.N_epochs)

    # test_data = _apply_overrides(config.to_dict(), overrides)

    # print(config.run.N_epochs)
    # print(test_data["run"]["N_epochs"])
