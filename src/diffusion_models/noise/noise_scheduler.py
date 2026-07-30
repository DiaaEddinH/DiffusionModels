import math
import torch

from torch import Tensor
from abc import ABC, abstractmethod


class Schedule(ABC):
    """
    Abstract class for defining the noise schedule of diffusion models.

    The noise schedule defines how the noise level changes over time and
    determines the corresponding coefficients of the PDE.

    Parameters
    ----------
    arg_min : float
        Lower bound of the schedule parameter.
    arg_max : float
        Upper bound of the schedule parameter.


    Notes
    -----
    Subclasses must implement :meth:`stddev`, :meth:`diffusion_coeff` and :meth:`mean_stddev`.

    See also
    --------
    :doc:`/diffusion
    """

    def __init__(self, arg_min: float, arg_max: float):
        self.arg_min = arg_min
        self.arg_max = arg_max

    @abstractmethod
    def stddev(self, t: Tensor) -> Tensor:
        """
        Computes the standard deviation of the noise at time ``t``.

        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Standard deviation corresponding to each value ``t``.
        :rtype: Tensor
        """

    @abstractmethod
    def diffusion_coeff(self, t: Tensor) -> Tensor:
        """
        Computes the diffusion coefficient of the diffusion process at time ``t``.

        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Diffusion coefficient corresponding to each value ``t``.
        :rtype: Tensor
        """

    @abstractmethod
    def mean_stddev(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the mean and standard deviation of the perturbed data at time ``t``.

        :param x: Input data
        :type x: Tensor
        :param t: Time values at which to evaluate the perturbation.
        :type t: Tensor
        :return: A tuple ``(mean, stddev)`` of the perturbed data.
        :rtype: tuple[Tensor, Tensor]
        """


class GeometricSchedule(Schedule):
    """
    Geometric noise schedule as used in the NCSN paper.
    """

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 10.0):
        super().__init__(sigma_min, sigma_max)

    @property
    def _logsigma(self) -> float:
        return math.log(self.arg_max / self.arg_min)

    def stddev(self, t: Tensor) -> Tensor:
        L = self._logsigma
        return self.arg_min * torch.sqrt((torch.exp(2 * t * L) - 1) / (2 * L))

    def diffusion_coeff(self, t: Tensor) -> Tensor:
        return self.arg_min * torch.exp(t * self._logsigma)


class LinearSchedule(Schedule):
    """
    Linear noise schedule as used in the DDPM paper.
    """

    def __init__(self, arg_min: float = 0.02, arg_max: float = 10.0):
        super().__init__(arg_min, arg_max)

    @property
    def _delta(self) -> float:
        return self.arg_max - self.arg_min

    def stddev(self, t: Tensor) -> Tensor:
        return self.arg_min + self._delta * t

    def diffusion_coeff(self, t: Tensor) -> Tensor:
        # Constant w.r.t. t
        return self._delta * torch.ones_like(t)
