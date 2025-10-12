import os
import numpy as np
from torch.utils.data import Dataset

from pathlib import Path

CWD = os.getcwd()

datasets_dir = Path("./data/raw")
datasets_dir.mkdir(parents=True, exist_ok=True)


class BaseDataset(Dataset):
    "Abstract dataset"

    def __init__(self, use_labels=False):
        super().__init__()
        self.use_labels = use_labels
        self.images, self.labels = None, None

    def __getitem__(self, index):
        if self.use_labels:
            return self.images[index], self.labels[index]
        return self.images[index]

    def __len__(self):
        return len(self.images)

    def normalise(self, data: np.ndarray, axis=None):
        self.mean = np.mean(data, axis=axis)
        self.stddev = np.std(data, axis=axis)
        return (data - self.mean) / self.stddev


class DoublePeak(BaseDataset):
    def __init__(self, mu: np.ndarray | float, sigma: float, size: int = 100_000):
        super().__init__(use_labels=False)
        self.images = self.init_dataset(mu, sigma, size)

    def init_dataset(self, mu, sigma, size: int):
        data = np.concatenate(
            [
                np.random.normal(mu, sigma, (size // 2, 2)),
                np.random.normal(-mu, sigma, (size // 2, 2)),
            ]
        ).astype(np.float32)
        return data


class DoublePeakMuConditioned(BaseDataset):
    def __init__(
        self, mu_min: float, mu_max: float, sigma: float = 0.25, size: int = 100_000
    ):
        super().__init__(use_labels=True)
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.sigma = sigma
        self.images, self.labels = self.init_dataset(size)

    def sample_mus(self, size: int) -> np.ndarray:
        """Sample continuous mu values uniformly between mu_min and mu_max."""
        return np.random.uniform(self.mu_min, self.mu_max, size).astype(np.float32)

    def init_dataset(self, size: int):
        # Sample μ values
        mus = self.sample_mus(size)

        # Split half for +μ peak, half for -μ peak
        half = size // 2
        mu_pos = mus[:half]
        mu_neg = mus[half:]

        # Generate samples for +μ peak
        data_pos = np.random.normal(
            loc=np.stack([mu_pos, -mu_pos], axis=1),
            scale=self.sigma,
        ).astype(np.float32)

        # Generate samples for -μ peak
        data_neg = np.random.normal(
            loc=np.stack([-mu_neg, mu_neg], axis=1),
            scale=self.sigma,
        ).astype(np.float32)

        data = np.concatenate([data_pos, data_neg], axis=0)
        labels = np.concatenate([mu_pos, mu_neg], axis=0)

        # Shuffle
        idx = np.random.permutation(size)
        return data[idx], labels[idx]


class DoublePeakMuDiscrete(BaseDataset):
    def __init__(
        self,
        mu_min: float = 0.0,
        mu_max: float = 1.0,
        n_classes: int = 5,
        delta: float = 0.05,
        sigma: float = 0.25,
        size: int = 100_000,
    ):
        super().__init__(use_labels=True)
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.n_classes = n_classes
        self.sigma = sigma
        self.delta = delta

        self.mu_values = np.linspace(mu_min, mu_max, n_classes, dtype=np.float32)
        self.images, self.labels = self.init_dataset(size)

    def init_dataset(self, size: int):
        # Sample μ values
        class_ids = np.random.randint(0, self.n_classes, size=size)
        mus = self.mu_values[class_ids]

        labels = mus + np.random.uniform(-self.delta, self.delta, size=size).astype(
            np.float32
        )
        labels = np.clip(labels, self.mu_min, None)

        # Split half for +μ peak, half for -μ peak
        half = size // 2
        mu_pos = mus[:half]
        mu_neg = mus[half:]

        # Generate samples for +μ peak
        data_pos = np.random.normal(
            loc=np.stack([mu_pos, -mu_pos], axis=1),
            scale=self.sigma,
        ).astype(np.float32)

        # Generate samples for -μ peak
        data_neg = np.random.normal(
            loc=np.stack([-mu_neg, mu_neg], axis=1),
            scale=self.sigma,
        ).astype(np.float32)

        data = np.concatenate([data_pos, data_neg], axis=0)
        labels = np.concatenate([labels[:half], labels[half:]], axis=0)

        # Shuffle
        idx = np.random.permutation(size)
        return data[idx], labels[idx]


class QuarticCL(BaseDataset):
    def __init__(self):
        super().__init__(use_labels=False)
        self.images = self.init_dataset()

    def init_dataset(self):
        dataset_file = datasets_dir / "cl_K111_ccc.dat"
        data = np.loadtxt(dataset_file, delimiter=",", dtype=np.float32)
        return self.normalise(data, axis=0)


class Phi4Dataset(BaseDataset):
    def __init__(self):
        super().__init__(use_labels=False)
        self.images = self.init_dataset()

    def init_dataset(self):
        dataset_file = datasets_dir / "cfgs_L32_k0.4_l0.022_10k.npy"
        data = np.load(dataset_file).astype(np.float32)
        data = self.normalise(data)
        return data.reshape(-1, 1, 32, 32)

    def normalise(self, data: np.ndarray):
        self.mean = np.mean(data)
        self.stddev = np.std(data)
        return (data - self.mean) / self.stddev

    def denorm(self, data: np.ndarray):
        return self.stddev * data + self.mean

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
