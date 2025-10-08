import numpy as np
from itertools import product


def bootstrap_estimator(data, observable, n_bins=100):
    """
    Generic bootstrap estimator.
    Args:
        data (ndarray): Input data
        observable (callable): Function to apply to bootstrap samples
        n_bins (int): Number of bootstrap resamples
    Returns:
        mean, error (floats or arrays)
    """
    n_samples = len(data)
    bins = []

    for _ in range(n_bins):
        bin_choice = np.random.choice(n_samples, size=n_samples, replace=True)
        bins.append(observable(data[bin_choice]))

    bins = np.squeeze(bins)
    if np.iscomplexobj(bins):
        bins = np.stack([bins.real, bins.imag], axis=-1)
    mean = np.mean(bins, axis=0)
    error = np.sqrt(np.sum((bins - mean) ** 2, axis=0) / (n_bins - 1))

    return mean, error


def moment(data, order, axis=0, center=None):
    """Compute the central moment of given order."""
    if center is None:
        center = np.mean(data, axis=axis)
    return np.mean((data - center) ** order, axis=axis)


# Cumulant definitions up to 8th order
kappa_fn = {
    1: lambda x: moment(x, 1),
    2: lambda x: moment(x, 2),
    3: lambda x: moment(x, 3),
    4: lambda x: moment(x, 4) - 3 * moment(x, 2) ** 2,
    5: lambda x: moment(x, 5) - 10 * moment(x, 2) * moment(x, 3),
    6: lambda x: moment(x, 6)
    - 15 * moment(x, 2) * moment(x, 4)
    + 30 * moment(x, 2) ** 3,
    7: lambda x: moment(x, 7)
    - 21 * moment(x, 2) * moment(x, 5)
    - 35 * moment(x, 4) * moment(x, 3)
    + 210 * moment(x, 3) * moment(x, 2) ** 2,
    8: lambda x: moment(x, 8)
    - 28 * moment(x, 2) * moment(x, 6)
    - 35 * moment(x, 4) ** 2
    + 420 * moment(x, 4) * moment(x, 2) ** 2
    - 630 * moment(x, 2) ** 4,
}


def calc_moments(data, max_order=8, n_bins=100):
    """Compute bootstrap-estimated moments."""
    vals, errs = [], []
    for n in range(1, max_order + 1):
        obs = lambda d, n=n: np.mean(d**n, axis=0)  # closure
        val, err = bootstrap_estimator(data, obs, n_bins=n_bins)
        vals.append(val)
        errs.append(err)
    return np.array(vals), np.array(errs)


def calc_cumulants(data, max_order=8, n_bins=100):
    """Compute bootstrap-estimated cumulants."""
    vals, errs = [], []
    for n in range(1, max_order + 1):
        obs = kappa_fn[n]
        val, err = bootstrap_estimator(data, obs, n_bins=n_bins)
        vals.append(val)
        errs.append(err)
    return np.array(vals), np.array(errs)


def other_moments(data, n, m, axis=0):
    x, y = data.T
    if n == m:
        return np.mean(x**n * y**m, axis=axis)
    return np.mean(x**n * y**m + x**m * y**n, axis=axis)


def calc_other_moments(data, max_order=8, n_bins=100):
    """Compute bootstrap-estimated moments."""
    vals, errs = [], []
    for n in range(1, max_order + 1):
        for m in range(1, n + 1):
            if (n + m) % 2 != 0:
                continue
            if (n + m) > max_order:
                continue

            obs = lambda d, n=n, m=m: other_moments(d, n, m, axis=0)  # closure
            val, err = bootstrap_estimator(data, obs, n_bins=n_bins)
            vals.append(val)
            errs.append(err)
    return np.array(vals), np.array(errs)
