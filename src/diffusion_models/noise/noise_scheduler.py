from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Schedule(ABC):
    """
    Abstract class for defining the noise schedule for diffusion models.
    The noise schedule defines how the noise level changes over time.
    """

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 10.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @abstractmethod
    def stddev(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the standard deviation of the noise at time t.
        """

    @abstractmethod
    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion coefficient at time t.
        """

    def mean_stddev(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and standard deviation of the perturbed data at time t.
        """
        return x, self.stddev(t)


class GeometricSchedule(Schedule):
    """
    Geometric noise schedule as used in the NCSN paper (copilot said its from this paper, is this true? if not fix).
    """

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 10.0):
        super().__init__(sigma_min, sigma_max)

    @property
    def _logsigma(self) -> float:
        return math.log(self.sigma_max / self.sigma_min)

    def stddev(self, t: torch.Tensor) -> torch.Tensor:
        L = self._logsigma
        return self.sigma_min * torch.sqrt((torch.exp(2 * t * L) - 1) / (2 * L))

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * torch.exp(t * self._logsigma)


class LinearSchedule(Schedule):
    """
    Linear noise schedule as used in the DDPM paper (copilot said its from this paper, is this true? if not fix).
    """

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 10.0):
        super().__init__(sigma_min, sigma_max)

    @property
    def _delta(self) -> float:
        return self.sigma_max - self.sigma_min

    def stddev(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min + self._delta * t

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        # Constant w.r.t. t
        return self._delta * torch.ones_like(t)
