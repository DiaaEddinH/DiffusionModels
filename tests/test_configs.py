# tests/test_config_loader.py
import sys
import unittest
import argparse
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path

from diffusion_models.config.config_loader import parse_configs, add_args_from_config


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self._argv_backup = list(sys.argv)

    def tearDown(self):
        sys.argv = self._argv_backup

    def _project_config_path(self) -> str:
        # repo_root/tests/test_config_loader.py -> repo_root/configs/example_config.yaml
        repo_root = Path(__file__).resolve().parents[1]
        cfg = (repo_root / "configs" / "example_config.yaml").as_posix()
        return cfg

    def test_parse_configs_with_yaml_and_cli_overrides(self):
        """
        End-to-end: parse a real YAML, get defaults, override a couple values via CLI,
        and ensure activation mapping uses the CLI-provided string (present in args).
        Covers: load_config(file exists), add_args_from_config(bool+non-bool),
                parse_configs activation-present branch.
        """
        cfg_path = self._project_config_path()
        sys.argv = [
            "pytest",
            "--config",
            cfg_path,
            "--lr",
            "0.01",
            "--batch_size",
            "128",
            "--activation",
            "relu",  # exercise activation-present branch
        ]
        args = parse_configs()

        # Defaults from YAML
        self.assertEqual(args.file, "example")
        self.assertTrue(args.ddp)
        self.assertEqual(args.device, "gpu")
        self.assertEqual(args.max_epochs, 2)
        self.assertEqual(args.num_workers, 1)
        self.assertEqual(args.patience, 50)
        self.assertEqual(args.in_channels, 2)
        self.assertEqual(args.hidden_channels, [64, 64])
        self.assertEqual(args.time_channels, 128)
        self.assertEqual(args.label_dim, 128)
        self.assertEqual(args.sigma_min, 0.02)
        self.assertEqual(args.sigma_max, 10)
        self.assertEqual(args.sample_size, 100000)
        self.assertEqual(args.time_steps, 500)

        # CLI overrides applied
        self.assertEqual(args.lr, 0.01)
        self.assertEqual(args.batch_size, 128)

        # Activation mapping applied from CLI string
        self.assertEqual(args.activation.__name__, "ReLU")

    def test_parse_configs_missing_file_uses_default_activation_and_reports_stderr(
        self,
    ):
        """
        Missing config path: returns {} from load_config, parser has only base args,
        and parse_configs takes the 'activation not in args' path -> defaults to leakyrelu.
        Also ensure the stderr message is emitted.
        Covers: load_config(missing), parse_configs activation-absent branch.
        """
        missing = "/tmp/definitely_not_here_config.yaml"
        sys.argv = ["pytest", "--config", missing]

        errbuf = StringIO()
        with redirect_stderr(errbuf):
            args = parse_configs()
        err = errbuf.getvalue()
        self.assertIn("Config file", err)
        self.assertIn(missing, err)

        # Only base + injected activation should exist; default activation is leakyrelu
        self.assertEqual(args.activation.__name__, "LeakyReLU")
        # Sanity: attributes from YAML should not exist when YAML is missing
        self.assertFalse(hasattr(args, "lr"))
        self.assertFalse(hasattr(args, "batch_size"))

    def test_add_args_from_config_bool_and_types(self):
        """
        Unit test the dynamic argument wiring:
        - bool becomes a --flag (store_true)
        - numeric/string keep their types
        And verify CLI actually flips the boolean when provided.
        """
        cfg = {
            "flag": False,  # bool path
            "lr": 0.1,  # float path
            "epochs": 5,  # int path
            "name": "exp",  # str path
        }
        parser = argparse.ArgumentParser()
        add_args_from_config(parser, cfg)
        parser.set_defaults(**cfg)

        # No overrides -> equals defaults
        ns = parser.parse_args([])
        self.assertFalse(ns.flag)
        self.assertEqual(ns.lr, 0.1)
        self.assertEqual(ns.epochs, 5)
        self.assertEqual(ns.name, "exp")

        # Override: presence of --flag flips to True; others parse as given
        ns = parser.parse_args(
            ["--flag", "--lr", "0.2", "--epochs", "7", "--name", "run42"]
        )
        self.assertTrue(ns.flag)
        self.assertEqual(ns.lr, 0.2)
        self.assertEqual(ns.epochs, 7)
        self.assertEqual(ns.name, "run42")


if __name__ == "__main__":
    unittest.main()
