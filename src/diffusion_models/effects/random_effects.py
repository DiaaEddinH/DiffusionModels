import numpy as np
from numpy._typing import ArrayLike
from typing_extensions import Iterable, Tuple, Self


class RandomEffectsAnalyser:
    """
    Perform random-effects meta-analysis across multiple runs.
    Each run provides an estimate of an observable and its associated statistical error.
    """

    def __init__(self, Y_i: ArrayLike, eps_i: ArrayLike):
        self.Y_i = np.asarray(Y_i)
        self.eps_i = np.asarray(eps_i)

    @classmethod
    def from_file_paths(cls, file_path: Iterable[str]) -> Self:
        Y_i, eps_i = cls._load_val_err(file_path)
        return cls(Y_i, eps_i)

    @staticmethod
    def _load_val_err(files_pattern: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load val and err arrays from .npz files matching the pattern.

        :param files_pattern: Iterable of file paths to .npz files.
        Each file should contain 'val' and 'err' arrays.
        :return: Tuple of two ndarrays
        """
        Y_i, eps_i = [], []
        for f in files_pattern:
            data = np.load(f)
            Y_i.append(data["val"])
            eps_i.append(data["err"])
        return np.stack(Y_i), np.stack(eps_i)

    def analyze(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Random-effects meta-analysis across runs.

        :return: Tuple of ndarrays (Y_hat, sigma_stat, sigma_sys_mean, sigma_tot)
            - Y_hat: combined observable estimate
            - sigma_stat: combined statistical error
            - sigma_sys_mean: systematic error on the mean due to between-run variance
            - sigma_tot: total error (statistical + systematic)
        """
        weights_i = 1 / self.eps_i**2
        weightedY_i = np.sum(weights_i * self.Y_i, axis=0) / np.sum(weights_i, axis=0)
        lof_i = np.sum(weights_i * (self.Y_i - weightedY_i) ** 2, axis=0)

        # Between-run variance
        density = np.sum(weights_i, axis=0) - np.sum(weights_i**2, axis=0) / np.sum(
            weights_i, axis=0
        )
        tau_sq = ((lof_i - (len(self.Y_i) - 1)) / density).clip(0.0)

        # Random-effects weights
        w_star = 1 / (self.eps_i**2 + tau_sq)
        Y_hat = np.sum(w_star * self.Y_i, axis=0) / np.sum(w_star, axis=0)
        a_i = w_star / np.sum(w_star, axis=0)

        # Error estimates
        sigma_stat = np.sqrt(np.sum((a_i * self.eps_i) ** 2, axis=0))
        sigma_sys_mean = np.sqrt(tau_sq * np.sum(a_i**2, axis=0))
        sigma_tot = np.sqrt(1 / np.sum(w_star, axis=0))

        return Y_hat, sigma_stat, sigma_sys_mean, sigma_tot
