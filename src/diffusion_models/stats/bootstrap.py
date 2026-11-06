import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable
from numpy.random import Generator

def binning_sampler(data: NDArray, observable: Callable[[NDArray], NDArray], n_bins: int = 100, rng: Generator | int | None = None) -> NDArray:
    """
    Generate bootstrap-binned samples of an observable (ie The output of callable :py:exc:`observable` with :py:data:`data`as input.)

    :param data: Input data (N, ...) where first dim indexes samples.
    :type data: NDArray
    :param observable: Function applied to each resampled dataset.
    :type observable: Callable[[NDArray], NDArray]
    :param n_bins: Number of bootstrap resamples, defaults to 100
    :type n_bins: int, optional
    :param rng: Random generator or seed for reproducibility, defaults to None
    :type rng: Generator | int | None, optional
    :return: Observable evaluated on each resample, shape (n_bins, ...)
    :rtype: NDArray
    """
    data = np.asarray(data)
    rng = np.random.default_rng(rng)
    n_samples = len(data)

    bins = []
    for _ in range(n_bins):
        idx = rng.integers(0, n_samples, size=n_samples)
        bins.append(observable(data[idx]))

    return np.asarray(bins)

def bootstrap_estimator(data: NDArray, observable: Callable[[NDArray], NDArray], n_bins: int = 100, rng: Generator | int | None = None) -> NDArray:
    """
    Generate bootstrap-binned samples of an observable (ie The output of callable :py:exc:`observable` with :py:data:`data`as input.)

    :param data: Input data (N, ...) where first dim indexes samples.
    :type data: NDArray
    :param observable: Function applied to each resampled dataset.
    :type observable: Callable[[NDArray], NDArray]
    :param n_bins: Number of bootstrap resamples, defaults to 100
    :type n_bins: int, optional
    :param rng: Random generator or seed for reproducibility, defaults to None
    :type rng: Generator | int | None, optional
    :return: Observable evaluated on each resample, shape (n_bins, ...)
    :rtype: NDArray
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2 to estimate error, got {n_bins}.")
    bins = binning_sampler(data, observable, n_bins, rng)
    if np.iscomplexobj(bins):
        bins = np.stack([bins.real, bins.imag], axis=-1)
    bin_estimates = np.mean(bins, axis=1)

    mean = np.mean(bin_estimates, axis=0)
    err = np.std(bin_estimates, axis=0, ddof=1)
    return mean, err


def bootstrap(data, n_bins: int = 100, rng: Generator | int | None = None) -> NDArray:
    """
    Bootstrap of the mean observable.

    :param data: Input data (N, ...) where first dim indexes samples.
    :type data: NDArray
    :param n_bins: Number of bootstrap resamples, defaults to 100
    :type n_bins: int, optional
    :param rng: Random generator or seed for reproducibility, defaults to None
    :type rng: Generator | int | None, optional
    :return: Observable evaluated on each resample, shape (n_bins, ...)
    :rtype: NDArray
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2 to estimate error, got {n_bins}.")
    bins = binning_sampler(data, lambda d: np.mean(d, axis=0), n_bins, rng)
    mean = np.mean(bins, axis=0)
    err = np.std(bins, axis=0, ddof=1)
    return mean, err

# def bootstrap(data: NDArray, n_boot: int=100, rng: Generator|int|None=None):
#     """Bootstrap of the mean observable (compat wrapper)."""
#     return bootstrap_estimator(data, lambda d: np.mean(d, axis=0), n_bins=n_boot, rng=rng)