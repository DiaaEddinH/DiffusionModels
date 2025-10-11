import sys
import unittest
from pathlib import Path

from config_loader import parse_configs


class ConfigTestCase(unittest.TestCase):
    def test_config_loading(self):
        # Point to your YAML config
        project_root = Path(__file__).resolve().parents[1]
        config_path = (project_root / "configs" / "example_config.yaml").as_posix()

        # Simulate CLI args override
        test_args = ["--config", config_path, "--lr", "0.01", "--batch_size", "128"]

        # Patch sys.argv
        sys.argv = ["test_configs.py"] + test_args

        # Parse configs
        args = parse_configs()

        # assert the above values
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


if __name__ == "__main__":
    unittest.main()
