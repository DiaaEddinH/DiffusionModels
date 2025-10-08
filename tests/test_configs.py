import sys
from pathlib import Path

# # Add src to path if running test directly
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config_loader import parse_configs


def test_config_loading():
    # Point to your YAML config
    config_path = "configs/example_config.yaml"

    # Simulate CLI args override
    test_args = ["--config", config_path, "--lr", "0.01", "--batch_size", "128"]

    # Patch sys.argv
    sys.argv = ["test_configs.py"] + test_args

    # Parse configs
    args = parse_configs()

    # Print results for manual inspection
    print("✅ Loaded config + CLI overrides:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Optional: assert some expected overrides
    assert args.lr == 0.01
    assert args.batch_size == 128


if __name__ == "__main__":
    test_config_loading()
