from __future__ import annotations

import math
from abc import ABC
from typing import Tuple

import torch


class Schedule(ABC):

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 10.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    """Strategy interface for noise schedules."""

    def stddev(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diffusion_coeff(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mean_stddev(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, self.stddev(t)


class GeometricSchedule(Schedule):

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
