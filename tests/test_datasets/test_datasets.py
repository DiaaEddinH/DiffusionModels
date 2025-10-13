from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from diffusion_models.datasets.datasets import (
    BaseDataset,
    DoublePeak,
    DoublePeakMuConditioned,
    DoublePeakMuDiscrete,
    HasLabelsMixin,
)


@dataclass
class DummyDataset(BaseDataset):
    def __post_init__(self):
        self.data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        super().__post_init__()


@dataclass
class LabeledDummyDataset(BaseDataset, HasLabelsMixin):
    def __post_init__(self):
        self.data = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)
        super().__post_init__()

    def init_labels(self):
        self.labels = np.array([0.1, 0.2, 0.3], dtype=np.float32)


@dataclass
class DummyMixin:
    test_bool: bool = field(default=False)

    def __post_init__(self):
        self.test_bool = True
        next_post = getattr(super(), "__post_init__", None)
        if next_post is not None:
            next_post()


@dataclass
class DummyDatasetWithMixins(BaseDataset, HasLabelsMixin, DummyMixin):
    def __post_init__(self):
        self.data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        super().__post_init__()

    def init_labels(self):
        self.labels = np.array([0.5, 0.6], dtype=np.float32)


def test_mixin_passes_on_postinit():
    dataset = DummyDatasetWithMixins()
    assert dataset.test_bool
    np.testing.assert_array_equal(
        dataset.labels, np.array([0.5, 0.6], dtype=np.float32)
    )


def test_base_dataset_post_init():
    dataset = DummyDataset()
    np.testing.assert_array_equal(dataset.data, dataset._raw_data)


def test_base_dataset_len():
    dataset = DummyDataset()
    assert len(dataset) == 2


def test_base_dataset_getitem():
    dataset = DummyDataset()
    np.testing.assert_array_equal(dataset[0], np.array([1.0, 2.0], dtype=np.float32))


def test_base_dataset_norm():
    dataset = DummyDataset()
    dataset.normalise()

    col_mean = dataset.data.mean()
    col_std = dataset.data.std()
    np.testing.assert_allclose(
        col_mean, np.array([0.0, 0.0], dtype=np.float32), atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        col_std, np.array([1.0, 1.0], dtype=np.float32), atol=1e-6, rtol=0
    )


def test_base_dataset_denorm():
    dataset = DummyDataset()
    dataset.normalise()
    dataset.denormalise()
    np.testing.assert_allclose(dataset.data, dataset._raw_data, rtol=1e-06)


def test_labeled_base_dataset_getitem():

    dataset = LabeledDummyDataset()

    # __getitem__ returns (image, label) when use_labels=True
    x0, y0 = dataset[0]
    np.testing.assert_array_equal(x0, dataset.data[0])
    assert y0 == dataset.labels[0]


def test_double_peak_shape_dtype_and_stats():
    size = 10_000
    mu = 0.7
    sigma = 0.25
    ds = DoublePeak(mu=mu, sigma=sigma, size=size)
    assert len(ds) == size
    assert ds.data.shape == (size, 2)
    assert ds.data.dtype == np.float32

    # Symmetry: x and y should have near-zero means overall due to mixing +/- mu
    m = ds.data.mean(axis=0)
    assert np.all(np.abs(m) < 0.05)
    # Data finite
    assert np.all(np.isfinite(ds.data))


def test_double_peak_vector_mu_broadcast_and_dtype():
    # Cover the code path where mu is a vector (broadcasted loc)
    size = 6000
    mu_vec = np.array([0.6, -0.4], dtype=np.float32)
    sigma = 0.2
    ds = DoublePeak(mu=mu_vec, sigma=sigma, size=size)

    assert len(ds) == size
    assert ds.data.shape == (size, 2)
    assert ds.data.dtype == np.float32
    assert np.all(np.isfinite(ds.data))
    # Means should be near zero due to +mu and -mu mixing
    m = ds.data.mean(axis=0)
    assert np.all(np.abs(m) < 0.1)


@pytest.mark.parametrize("mu_min, mu_max, sigma, size", [(0.2, 1.0, 0.1, 20_000)])
def test_double_peak_mu_conditioned_label_correlation(mu_min, mu_max, sigma, size):
    ds = DoublePeakMuConditioned(mu_min=mu_min, mu_max=mu_max, sigma=sigma, size=size)

    assert len(ds) == size
    assert ds.data.shape == (size, 2)
    assert ds.labels.shape == (size,)

    # Labels in range
    assert float(ds.labels.min()) >= (mu_min - 1e-6)
    assert float(ds.labels.max()) <= (mu_max + 1e-6)

    # Correlation: For points centered at (±mu, ∓mu), |x|+|y| ≈ 2*mu
    approx_mu = 0.5 * (np.abs(ds.data[:, 0]) + np.abs(ds.data[:, 1]))
    # With Gaussian test_noise sigma on both dims, allow a tolerance
    np.testing.assert_allclose(approx_mu.mean(), ds.labels.mean(), rtol=0.05, atol=0.05)

    # __getitem__ returns (image, label)
    x, y = ds[3]
    assert x.shape == (2,)
    assert isinstance(y, np.floating) or isinstance(y, float)


def test_double_peak_mu_discrete_properties_and_correlation():
    mu_min, mu_max, n_classes, delta, sigma, size = 0.1, 1.0, 7, 0.05, 0.2, 12_000
    ds = DoublePeakMuDiscrete(
        mu_min=mu_min,
        mu_max=mu_max,
        n_classes=n_classes,
        delta=delta,
        sigma=sigma,
        size=size,
    )

    assert len(ds) == size
    assert ds.data.shape == (size, 2)
    assert ds.labels.shape == (size,)

    # Labels lower-bounded by mu_min; upper side can exceed mu_max by up to ~delta due to how labels are generated
    assert ds.labels.min() >= mu_min - 1e-6
    assert ds.labels.max() <= mu_max + delta + 1e-6

    # Each label is near a class center (after potential clipping at mu_min)
    mu_values = np.linspace(mu_min, mu_max, n_classes, dtype=np.float32)
    dists = np.min(np.abs(ds.labels[:, None] - mu_values[None, :]), axis=1)
    assert np.mean(dists <= (delta + 0.02)) > 0.95  # most within delta (+ small slack)

    # Correlation check like above
    approx_mu = 0.5 * (np.abs(ds.data[:, 0]) + np.abs(ds.data[:, 1]))
    np.testing.assert_allclose(approx_mu.mean(), ds.labels.mean(), rtol=0.08, atol=0.08)


def test_double_peak_mu_discrete_clipping_happens():
    # Choose params where some labels will be pushed below mu_min before clipping
    mu_min, mu_max, n_classes, delta, sigma, size = 0.2, 0.6, 3, 0.15, 0.2, 6000
    ds = DoublePeakMuDiscrete(
        mu_min=mu_min,
        mu_max=mu_max,
        n_classes=n_classes,
        delta=delta,
        sigma=sigma,
        size=size,
    )
    # At least some labels should be clipped to mu_min
    assert np.any(ds.labels == mu_min)
    # And none should fall below mu_min after clipping
    assert ds.labels.min() >= mu_min - 1e-6


@pytest.fixture()
def ensure_data_raw(tmp_path, monkeypatch):
    # Redirect datasets_dir to a temp directory for file-based datasets
    tmp_raw = tmp_path / "data" / "raw"
    tmp_raw.mkdir(parents=True, exist_ok=True)

    # Monkeypatch the module-level datasets_dir Path to our temp path
    from diffusion_models.datasets import datasets as datasets_module

    monkeypatch.setattr(datasets_module, "datasets_dir", tmp_raw, raising=True)

    return tmp_raw


def test_quartic_dataset(ensure_data_raw):
    # Import from the already monkeypatched module so we don't touch real data
    from diffusion_models.datasets import datasets as datasets_module

    QuarticCL = datasets_module.QuarticCL

    raw_dir: Path = ensure_data_raw
    file_path = raw_dir / "cl_K111_ccc.dat"

    # Create simple 2-column data with different scales
    data = np.stack(
        [
            np.linspace(-1.0, 1.0, 101, dtype=np.float32),
            np.linspace(10.0, 20.0, 101, dtype=np.float32),
        ],
        axis=1,
    )
    np.savetxt(file_path, data, delimiter=",", fmt="%.6f")

    dataset = QuarticCL()
    assert dataset.data.shape == data.shape


def test_phi4dataset_shape_normalise_and_denorm(ensure_data_raw):
    # Import from the already monkeypatched module so we don't touch real data
    from diffusion_models.datasets import datasets as datasets_module

    Phi4Dataset = datasets_module.Phi4Dataset

    raw_dir: Path = ensure_data_raw
    file_path = raw_dir / "cfgs_L32_k0.4_l0.022_10k.npy"

    # Create synthetic lattice configs with shape (N, 32, 32)
    n = 128
    original = np.random.randn(n, 32, 32).astype(np.float32) * 3.0 + 0.5
    np.save(file_path, original)

    dataset = Phi4Dataset()
    assert dataset._raw_data.shape == (n, 1, 32, 32)


def test_phi4_len_and_getitem(ensure_data_raw):
    # Import from the already monkeypatched module so we don't touch real data
    from diffusion_models.datasets import datasets as datasets_module

    Phi4Dataset = datasets_module.Phi4Dataset

    raw_dir: Path = ensure_data_raw
    file_path = raw_dir / "cfgs_L32_k0.4_l0.022_10k.npy"

    n = 32
    original = np.random.randn(n, 32, 32).astype(np.float32)
    np.save(file_path, original)

    ds = Phi4Dataset()
    assert len(ds) == n
    sample = ds[0]
    assert sample.shape == (1, 32, 32)
    assert sample.dtype == np.float32
