import math
import torch

from torch import Tensor
from abc import ABC, abstractmethod


class Schedule(ABC):
    """
    Abstract class for defining the noise schedule of diffusion models.

    The noise schedule defines how the noise level changes over time and
    determines the corresponding coefficients of the PDE.

    The backward diffusion process proceeds from :math:`t=1` to :math:`t=\\varepsilon`.

    Parameters
    ----------
    arg_min : float
        Lower bound of the schedule parameter.
    arg_max : float
        Upper bound of the schedule parameter.
    eps : float
        Lower time bound of the schedule. Defaults to 1e-3.


    Notes
    -----
    Subclasses must implement :meth:`stddev`, :meth:`diffusion_coeff`, :meth:`drift_term`, :meth:`mean_stddev` and :meth:`invert_variance_to_time`.

    See also
    --------
    :doc:`../theory/diffusion`
    """

    def __init__(self, arg_min: float, arg_max: float, eps: float = 1e-3):
        self.arg_min = arg_min
        self.arg_max = arg_max
        self.eps = eps

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
    def drift_term(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Computes the drift term of the diffusion process at time ``t``.

        :param x: Input state/data
        :type x: Tensor
        :param t: Time values at which to evaluate.
        :type t: Tensor
        :return: Drift term corresponding to each pair ``(x, t)``.
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

    @abstractmethod
    def invert_variance_to_time(self, variance_schedule: Tensor) -> Tensor:
        """
        Maps a variance schedule to a time step schedule.

        :param variance_schedule: The variance schedule. Should be monotonically decreasing.
        :type variance_schedule: Tensor
        :return: A monotonically decreasing timestep schedule.
        :rtype: Tensor
        """

    def build_variance_schedule(
        self, num_steps: int, schedule_type: str = "uniform", rho: float = 1
    ) -> Tensor:
        """
        Builds a time schedule with uniform spacing in variance.

        :param num_steps: Number of steps in the schedule
        :type num_steps: int
        :param schedule_type: Type of variance schedule to build from. Must be one of {`uniform`, `log`, `karras`}. Choose `uniform` for uniform variance spacing, `log` for logarithmically uniform spacing and `karras` to implement a variance schedule according to `arXiv:2206.00364 <https://arxiv.org/pdf/2206.00364>`_. Defaults to `uniform`.
        :type schedule_type: str, optional
        :param rho: Parameter which controls step size across noise levels if the `karras` schedule has been specified. Large `rho` corresponds to bigger step sizes at large noise levels. Defaults to 1.0
        :type rho: float, optional
        :return: A time schedule of size `num_steps`
        :rtype: Tensor
        """
        assert schedule_type in ["uniform", "log", "karras"]

        # Variance schedule in decreasing order
        if schedule_type == "log":
            std2_max, std2_min = self.stddev(torch.tensor([1, self.eps])) ** 2
            variance_schedule = torch.logspace(
                std2_max.log10(), std2_min.log10(), steps=num_steps
            )
        elif schedule_type == "uniform":
            std2_max, std2_min = self.stddev(torch.tensor([1, self.eps])) ** 2
            variance_schedule = torch.linspace(std2_max, std2_min, steps=num_steps)
        elif schedule_type == "karras":
            std_max, std_min = self.stddev(torch.tensor([1, self.eps]))
            i_range = torch.arange(num_steps, dtype=std_min.dtype)
            inv_rho = 1.0 / rho
            variance_schedule = (
                std_max.pow(inv_rho)
                + i_range
                / (num_steps - 1)
                * (std_min.pow(inv_rho) - std_max.pow(inv_rho))
            ).pow(2 * rho)

        timesteps = self.invert_variance_to_time(variance_schedule)
        return timesteps


class GeometricSchedule(Schedule):
    """
    Noise scheduler based on the geometric schedule used in NCSN, see `arXiv:1907.05600 <https://arxiv.org/abs/1907.05600>`_.
    Schedule has been adjusted for continuous time, see `arXiv:2011.13456 <https://arxiv.org/abs/2011.13456>`_.
    This schedule is intended for variance expanding (VE) diffusion processes. This process is defined by the following SDE

    .. math::

        dx_t = \\sigma_{\\min} \\left( \\frac{\\sigma_{\\max}}{\\sigma_{\\min}} \\right)^t dW_t


    Parameters
    ----------
    sigma_min : float
        Lower bound of the schedule noise scale.
    sigma_max : float
        Upper bound of the schedule noise scale.
    eps : float
        Lower time bound of the schedule. Defaults to 1e-3.
    """

    def __init__(
        self, sigma_min: float = 1.0, sigma_max: float = 10.0, eps: float = 1e-3
    ):
        super().__init__(sigma_min, sigma_max, eps)

    @property
    def _logsigma(self) -> float:
        """
        :return: Logarithmic ratio of the noise scale bounds
        :rtype: float
        """
        return math.log(self.arg_max / self.arg_min)

    def stddev(self, t: Tensor) -> Tensor:
        """
        Evaluates the standard deviation of the noise schedule

        .. math::

            \\sigma(t) = \\sigma_{\\min} \\left(\\frac{\\alpha^{2t} - 1}{\\log(\\alpha^2)}\\right)^\\frac{1}{2},

        where :math:`\\alpha = \\frac{\\sigma_{\\max}}{\\sigma_{\\min}}`.


        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Standard deviation corresponding to each value ``t``.
        :rtype: Tensor
        """
        L = self._logsigma
        return self.arg_min * torch.sqrt((torch.exp(2 * t * L) - 1) / (2 * L))

    def diffusion_coeff(self, t: Tensor) -> Tensor:
        """
        Computes the diffusion coefficient of the diffusion process at time ``t``.

        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Diffusion coefficient corresponding to each value ``t``.
        :rtype: Tensor
        """
        return self.arg_min * torch.exp(t * self._logsigma)

    def drift_term(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Computes the drift term of the diffusion process at time ``t``.
        In the VE scheme it is identically zero.

        :param x: Input state/data
        :type x: Tensor
        :param t: Time values at which to evaluate.
        :type t: Tensor
        :return: Identically zero vector field with ``x``'s shape.
        :rtype: Tensor
        """
        return torch.zeros_like(x)

    def mean_stddev(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Computes the mean and standard deviation of the perturbed data at time ``t``.
        In the variance expanding scheme, the mean doesn't change so the input data `x` passes as is.

        :param x: Input data
        :type x: Tensor
        :param t: Time values at which to evaluate the perturbation.
        :type t: Tensor
        :return: A tuple ``(mean, stddev)`` of the perturbed data.
        :rtype: tuple[Tensor, Tensor]
        """
        d = (x.dim() - 1) * (None,)
        return x, self.stddev(t)[:, *d]

    def invert_variance_to_time(self, variance_schedule: Tensor) -> Tensor:
        """
        Maps a variance schedule to a time step schedule.

        :param variance_schedule: The variance schedule. Should be monotonically decreasing.
        :type variance_schedule: Tensor
        :return: A monotonically decreasing timestep schedule.
        :rtype: Tensor
        """
        L = self._logsigma
        s2_min = self.arg_min**2
        return torch.log1p(2 * L * variance_schedule / s2_min) / (2 * L)


class LinearSchedule(Schedule):
    """
    Noise scheduler based on the linear schedule used in DDPMs, see `arXiv:2006.11239 <https://arxiv.org/abs/2006.11239>`_.
    Schedule has been adjusted for continuous time, see `arXiv:2011.13456 <https://arxiv.org/abs/2011.13456>`_.
    This schedule is intended for variance preserving (VP) diffusion processes. This process is defined by the following SDE

    .. math::

        \\rm dx_t = -\\frac{1}{2} \\beta(t) x_t + \\sqrt{\\beta(t)} dW_t


    Parameters
    ----------
    sigma_min : float
        Lower bound of the schedule noise scale.
    sigma_max : float
        Upper bound of the schedule noise scale.
    eps : float
        Lower time bound of the schedule. Defaults to 1e-3.
    """

    def __init__(
        self, beta_min: float = 0.02, beta_max: float = 10.0, eps: float = 1e-3
    ):
        super().__init__(beta_min, beta_max, eps)

    @property
    def _delta(self) -> float:
        return self.arg_max - self.arg_min

    def beta_schedule(self, t: Tensor) -> Tensor:
        """
        Evaluates the noise schedule :math:`\\beta(t) = \\beta_{\\min} + (\\beta_{\\max} - \\beta_{\\min})` for each time ``t``.

        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Noise scale corresponding to each value ``t``.
        :rtype: Tensor
        """
        return self.arg_min + self._delta * t

    def mean_factor(self, t: Tensor) -> Tensor:
        """
        Calculates rescaling factor for the mean value during the process

        .. math::

            m(t) = \\exp\\left(-\\frac{1}{2}\\int_0^t \\; \\beta(s) \\rm ds\\right).

        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Mean rescaling corresponding to each value ``t``.
        :rtype: Tensor
        """
        beta_integral = self.arg_min * t + 0.5 * self._delta * t**2
        return torch.exp(-0.5 * beta_integral)

    def stddev(self, t: Tensor) -> Tensor:
        """
        Evaluates the standard deviation of the noise schedule, :math:`\\sigma(t) = \\sqrt{1 - m^2(t)}`.

        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Standard deviation corresponding to each value ``t``.
        :rtype: Tensor
        """
        return torch.sqrt(1 - self.mean_factor(t) ** 2)

    def diffusion_coeff(self, t: Tensor) -> Tensor:
        """
        Computes the diffusion coefficient of the diffusion process at time ``t``.

        :param t: Time values at which to evaluate the schedule.
        :type t: Tensor
        :return: Diffusion coefficient corresponding to each value ``t``.
        :rtype: Tensor
        """
        return torch.sqrt(self.beta_schedule(t))

    def drift_term(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Computes the drift term of the diffusion process at time ``t``.

        :param x: Input state/data
        :type x: Tensor
        :param t: Time values at which to evaluate.
        :type t: Tensor
        :return: Drift term corresponding to each pair ``(x, t)``.
        :rtype: Tensor
        """
        d = (x.dim() - 1) * (None,)
        return -0.5 * self.beta_schedule(t)[:, *d] * x

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
        d = (x.dim() - 1) * (None,)
        return self.mean_factor(t)[:, *d] * x, self.stddev(t)[:, *d]

    def invert_variance_to_time(self, variance_schedule: Tensor) -> Tensor:
        """
        Maps a variance schedule to a time step schedule.

        :param variance_schedule: The variance schedule. Should be monotonically decreasing.
        :type variance_schedule: Tensor
        :return: A monotonically decreasing timestep schedule.
        :rtype: Tensor
        """
        delta = self._delta
        arg_min = self.arg_min
        log1m_variance = torch.log(1 - variance_schedule).abs()

        return (arg_min / delta) * (
            -1 + torch.sqrt(1 + 2 * delta * log1m_variance / arg_min**2)
        )
