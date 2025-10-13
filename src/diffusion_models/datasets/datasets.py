import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy._typing import NDArray
from torch.utils.data import Dataset
from typing_extensions import Any, Tuple, Union

CWD = os.getcwd()

datasets_dir = Path("./data/raw")
datasets_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class BaseDataset(Dataset, ABC):

    data: NDArray = field(init=False)

    _normalized: bool = field(default=False, init=False)

    _raw_data: NDArray = field(default=None, init=False)

    def __post_init__(self):
        next_post = getattr(super(), "__post_init__", None)
        if next_post is not None:
            next_post()

        self._raw_data = self.data

    def __getitem__(self, item) -> Union[Any, Tuple[Any]]:
        items = []
        for cls in type(self).__mro__:
            getitem_part = cls.__dict__.get("_getitem_part")
            if getitem_part is not None:
                items.append(getitem_part(self, item))

        items = [
            getitem_part(self, item)
            for clazz in type(self).__mro__
            if callable(getitem_part := clazz.__dict__.get("_getitem_part"))
        ]

        return items[0] if len(items) == 1 else tuple(items)

    def _getitem_part(self, item) -> Any:
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def normalise(self, axis=None):
        if not self._normalized:
            self.data = (self.data - np.mean(self._raw_data, axis)) / np.std(
                self._raw_data, axis
            )
            self._normalized = True

    def denormalise(self, axis=None):
        if self._normalized:
            self.data = self.data * np.std(self._raw_data, axis) + np.mean(
                self._raw_data, axis
            )
            self._normalized = False


@dataclass
class HasLabelsMixin(ABC):
    labels: NDArray = field(init=False)

    def __post_init__(self):
        self.init_labels()

        next_post = getattr(super(), "__post_init__", None)
        if next_post is not None:
            next_post()

    @abstractmethod
    def init_labels(self): ...

    def _getitem_part(self, item):
        return self.labels[item]


@dataclass
class GaussianDataset(BaseDataset, ABC):
    sigma: float = field(default=0.25)
    size: int = field(default=100_000)
    shape: Tuple[int, int] = field(init=False)


@dataclass
class FixedMuGaussianDataset(GaussianDataset, ABC):
    mu: Union[NDArray, float] = field(default=0.5)


@dataclass
class VariableMuGaussianDataset(GaussianDataset, ABC):
    mu_min: float = field(default=0.0)
    mu_max: float = field(default=1.0)

    mus: NDArray = field(init=False)

    shuffled_idx: NDArray = field(init=False)

    def __post_init__(self):
        data = self._sample_data()

        # Shuffle
        self.shuffled_idx = np.random.permutation(self.size)
        self.data = data[self.shuffled_idx]
        super().__post_init__()

    def _sample_data(self):
        # Sample μ values
        self.mus = self.sample_mus()

        # Split half for +μ peak, half for -μ peak
        half = self.size // 2
        mu_pos = self.mus[:half]
        mu_neg = self.mus[half:]

        # Generate samples for +μ peak
        data_pos = self._generate_samples(mu_pos, negation_first=False)
        # Generate samples for -μ peak
        data_neg = self._generate_samples(mu_neg, negation_first=True)

        return np.concatenate([data_pos, data_neg], axis=0)

    def _generate_samples(self, mus: NDArray, negation_first: bool = False):
        first_sign, second_sign = (1, -1) if negation_first else (-1, 1)

        samples = np.random.normal(
            loc=np.stack([first_sign * mus, second_sign * mus], axis=1),
            scale=self.sigma,
        ).astype(np.float32)

        return samples

    @abstractmethod
    def sample_mus(self) -> np.ndarray: ...


@dataclass
class DoublePeak(FixedMuGaussianDataset):

    def __post_init__(self):
        shape = (self.size // 2, 2)
        a = np.random.normal(loc=self.mu, scale=self.sigma, size=shape).astype(
            np.float32
        )
        b = np.random.normal(loc=-self.mu, scale=self.sigma, size=shape).astype(
            np.float32
        )
        data: NDArray[np.float32] = np.concatenate([a, b], axis=0)
        self.data = data


@dataclass
class DoublePeakMuConditioned(VariableMuGaussianDataset, HasLabelsMixin):

    def __post_init__(self):
        super().__post_init__()

    def sample_mus(self) -> np.ndarray:
        """Sample continuous mu values uniformly between mu_min and mu_max."""
        return np.random.uniform(self.mu_min, self.mu_max, self.size).astype(np.float32)

    def init_labels(self):
        labels = self.mus.copy()
        self.labels = labels[self.shuffled_idx]


@dataclass
class DoublePeakMuDiscrete(VariableMuGaussianDataset, HasLabelsMixin):
    n_classes: int = field(default=5)
    delta: float = field(default=0.05)

    def sample_mus(self) -> np.ndarray:
        """Sample continuous mu values uniformly between mu_min and mu_max."""
        mus = np.linspace(self.mu_min, self.mu_max, self.n_classes, dtype=np.float32)
        class_ids = np.random.randint(0, self.n_classes, size=self.size)
        return mus[class_ids]

    def init_labels(self):
        labels = self.mus + np.random.uniform(
            -self.delta, self.delta, size=self.size
        ).astype(np.float32)
        labels = np.clip(labels, self.mu_min, None)
        self.labels = labels[self.shuffled_idx]


@dataclass
class QuarticCL(BaseDataset):
    def __post_init__(self):
        dataset_file = datasets_dir / "cl_K111_ccc.dat"
        self.data = np.loadtxt(dataset_file, delimiter=",", dtype=np.float32)
        super().__post_init__()
        # self.normalise(data, axis=0) was previously here. I dont think it belongs here, but it is up to you


@dataclass
class Phi4Dataset(BaseDataset):
    def __post_init__(self):
        dataset_file = datasets_dir / "cfgs_L32_k0.4_l0.022_10k.npy"
        self.data = np.load(dataset_file).astype(np.float32)
        self.data = self.data.reshape(-1, 1, 32, 32)
        super().__post_init__()
        # self.normalise()  # was previously here. I dont think it belongs here, but it is up to you
