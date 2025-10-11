import sys
import unittest
import argparse
from pathlib import Path

from src.config_loader import parse_configs, load_config, add_args_from_config


class ConfigTestCase(unittest.TestCase):
    def test_config_loading(self):
        # Point to your YAML config
        project_root = Path(__file__).resolve().parents[1]
        config_path = (project_root / "configs" / "example_config.yaml").as_posix()

        # Simulate CLI args override
        test_args = ["--config", config_path, "--lr", "0.01", "--batch_size", "128"]

        # Patch sys.argv
        sys.argv = ["test_configs.py"] + test_args

        args = parse_configs()

        self.assertEqual(args.file, "example")
        self.assertTrue(args.ddp)
        self.assertEqual(args.device, "gpu")
        self.assertEqual(args.lr, 0.01)
        self.assertEqual(args.max_epochs, 2)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.num_workers, 1)
        self.assertEqual(args.patience, 50)
        self.assertEqual(args.in_channels, 2)
        self.assertEqual(args.hidden_channels, [64, 64])
        self.assertEqual(args.time_channels, 128)
        self.assertEqual(args.label_dim, 128)
        self.assertEqual(args.activation.__name__, "LeakyReLU")
        self.assertEqual(args.sigma_min, 0.02)
        self.assertEqual(args.sigma_max, 10)
        self.assertEqual(args.sample_size, 100000)
        self.assertEqual(args.time_steps, 500)

    def test_load_config_missing_returns_empty(self):
        cfg = load_config("/tmp/this_config_does_not_exist.yaml")
        self.assertEqual(cfg, {})

    def test_add_args_from_config_and_parse(self):
        # Prepare a minimal config dict
        cfg = {"flag": False, "lr": 0.1, "epochs": 5, "name": "exp"}
        parser = argparse.ArgumentParser()
        add_args_from_config(parser, cfg)
        # defaults from config
        parser.set_defaults(**cfg)

        # Case 1: no overrides -> should equal defaults
        args = parser.parse_args([])
        self.assertFalse(args.flag)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.name, "exp")

        # Case 2: override bool via flag and numeric via CLI
        args = parser.parse_args(["--flag", "--lr", "0.2", "--epochs", "7"])
        self.assertTrue(args.flag)
        self.assertEqual(args.lr, 0.2)
        self.assertEqual(args.epochs, 7)
        self.assertEqual(args.name, "exp")  # unchanged


if __name__ == "__main__":
    unittest.main()
