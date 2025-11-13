import numpy as np


def bootstrap_estimator(data, observable, n_bins=100):
    """
    Generic bootstrap estimator.
    Args:
        data (ndarray): Input data (N, ...) where the first dim indexes samples
        observable (callable): Function to apply to bootstrap samples; returns scalar or vector
        n_bins (int): Number of bootstrap resamples
    Returns:
        mean, error (floats or arrays)
    """
    n_samples = len(data)
    bins = []
    for _ in range(n_bins):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bins.append(observable(data[idx]))

    bins = np.asarray(bins)  # predictable shape handling
    if np.iscomplexobj(bins):
        bins = np.stack([bins.real, bins.imag], axis=-1)

    mean = np.mean(bins, axis=0)
    # Unbiased std across bootstrap replicas
    error = np.sqrt(np.sum((bins - mean) ** 2, axis=0) / max(1, (n_bins - 1)))
    return mean, error


def bootstrap(data, n_boot=100):
    """Bootstrap of the mean observable (compat wrapper)."""
    return bootstrap_estimator(data, lambda d: np.mean(d, axis=0), n_bins=n_boot)
