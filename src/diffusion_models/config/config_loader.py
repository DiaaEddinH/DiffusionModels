from pathlib import Path
import argparse
import yaml
import sys

from diffusion_models.utils import get_activation_func


def load_config(config_path):
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found.", file=sys.stderr)
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def add_args_from_config(parser, config):
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", action="store_true", help=f"Override {key}")
        else:
            parser.add_argument(f"--{key}", type=type(value), help=f"Override {key}")


def parse_configs():
    # 1) Create parser to read configuration file
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/example_config.yaml",
        help="Path to config file",
    )
    base_args, extras = base_parser.parse_known_args()

    # 2) Load config file
    config = load_config(base_args.config)

    # 3) Parser dynamically adds args from config
    parser = argparse.ArgumentParser(parents=[base_parser])
    add_args_from_config(parser, config)

    # 4) Set defaults from config
    parser.set_defaults(**config)

    # 5) Parse CLI, CLI overrides config
    args = parser.parse_args(extras)

    # This part is only for this type of config file
    if "activation" in args:
        act = args.activation
    else:
        act = "leakyrelu"
    args.activation = get_activation_func(act)
    return args
