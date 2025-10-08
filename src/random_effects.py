import numpy as np
from glob import glob


def load_val_err(files_pattern):
    """Load val and err arrays from .npz files matching the pattern."""
    Y_i, eps_i = [], []
    for f in files_pattern:
        data = np.load(f)
        Y_i.append(data["val"])
        eps_i.append(data["err"])
    return np.stack(Y_i), np.stack(eps_i)


def calc_random_effects(Y_i, eps_i):
    """
    Random-effects meta-analysis across runs.
    Args:
            Y_i (ndarray): shape (n_runs, ..., ) values
            eps_i (ndarray): shape (n_runs, ..., ) errors
    Returns:
            Y_hat, sigma_stat, sigma_sys_mean, sigma_tot
    """
    weights_i = 1 / eps_i**2
    weightedY_i = np.sum(weights_i * Y_i, axis=0) / np.sum(weights_i, axis=0)
    lof_i = np.sum(weights_i * (Y_i - weightedY_i) ** 2, axis=0)

    # Between-run variance
    density = np.sum(weights_i, axis=0) - np.sum(weights_i**2, axis=0) / np.sum(
        weights_i, axis=0
    )
    tau_sq = ((lof_i - (len(Y_i) - 1)) / density).clip(0.0)

    # Random-effects weights
    w_star = 1 / (eps_i**2 + tau_sq)
    Y_hat = np.sum(w_star * Y_i, axis=0) / np.sum(w_star, axis=0)
    a_i = w_star / np.sum(w_star, axis=0)

    # Error estimates
    sigma_stat = np.sqrt(np.sum((a_i * eps_i) ** 2, axis=0))
    sigma_sys_mean = np.sqrt(tau_sq * np.sum(a_i**2, axis=0))
    sigma_tot = np.sqrt(1 / np.sum(w_star, axis=0))

    return Y_hat, sigma_stat, sigma_sys_mean, sigma_tot


def random_effects_from_files(files_pattern: str):
    Y_i, eps_i = load_val_err(files_pattern)
    return calc_random_effects(Y_i, eps_i)
