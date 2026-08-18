import math
import torch

from tqdm import tqdm
from torch import Tensor
from abc import ABC, abstractmethod

from math import pi
from collections.abc import Callable
from diffusion_models.models.models import ScoreModel


class BaseSampler(ABC):
    """
    Base class for reverse-time SDE/ODE samplers. The process proceeds from :math:`t=1` to :math:`t=\\varepsilon`.
    It includes optional trajectory recording.

    Specifying initial states and sampling algorithm itself is left to subclasses.

    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    device: torch.device
        Device on which sampling is performed.
    schedule : :class:`~diffusion_models.noise.noise_scheduler.Schedule`
        Noise schedule of the diffusion process.
    eps : float
        Lower time bound of the reverse process.


    Notes
    -----
        Subclasses must implement :meth:`init_sample`, :meth:`sample` and :meth:`update_step`.
    """

    def __init__(self, model: ScoreModel):
        super().__init__()
        self.model = model
        self.device = model.device
        self.schedule = model.schedule
        self.eps = model.schedule.eps

    @abstractmethod
    def init_sample(self, shape: tuple) -> Tensor:
        """
        Initiates the initial condition for the backward diffusion process.

        :param shape: The shape of the data
        :type shape: tuple
        :return: An initial state sampled from a simple prior distribution.
        :rtype: Tensor
        """

    @abstractmethod
    def sample(self, shape: tuple, num_steps: int, *labels, **kwargs) -> Tensor:
        """
        Sampling loop of the reverse diffusion process.

        :param shape: The shape of the data
        :type shape: tuple
        :param num_steps: Number of steps in the reverse process
        :type num_steps: int
        :return: Final samples
        :rtype: Tensor
        """

    @abstractmethod
    def update_step(
        self,
        x: Tensor,
        drift: Tensor,
        step_size: Tensor | float,
        step_size_sqrt: Tensor | float,
    ) -> Tensor:
        """
        Integration scheme for the sampler of the reverse diffusion process.
        Subclasses generally differ in implementation.
        Typically, SDEs would use an Euler-Maruyama scheme.

        :param x: Input state
        :type x: Tensor
        :param drift: Drift term of the process.
        :type drift: Tensor
        :param step_size: Step size of the integration
        :type step_size: Tensor | float
        :param step_size_sqrt: Square root of the step size. Used in SDEs.
        :type step_size_sqrt: Tensor | float
        """

    def _score(self, x: Tensor, t: Tensor, *args) -> Tensor:
        """
        Subclasses or mixins that need to rescale, cache,
        or post-process the raw forward pass should override this method instead
        rather than calling ``self.model(...)`` directly.


        :param x: Input state
        :type x: Tensor
        :param t: Time values at which to evaluate.
        :type t: Tensor
        :return: Model output as ``(x, t)``.
        :rtype: Tensor
        """
        return self.model(x, t, *args)

    def init_history(self, num_steps: int, shape: tuple, keep_history: bool = False) -> Tensor | None:
        return torch.empty(num_steps, *shape, device=self.device) if keep_history else None

    def record(self, history: Tensor, x: Tensor, idx: int, flag: bool):
        if flag:
            history[idx] = x

    def collect(self, history: Tensor, x: Tensor, flag: bool) -> Tensor:
        return history if flag else x


class EulerMaruyamaSampler(BaseSampler):
    """
    This class implements a Euler-Maruyama based sampler intended for diffusion model sample generation.

    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    device: torch.device
        Device on which sampling is performed.
    schedule : :class:`~diffusion_models.noise.noise_scheduler.Schedule`
        Noise schedule of the diffusion process.
    """

    def __init__(self, model: ScoreModel):
        super().__init__(model)

    def init_sample(self, shape: tuple) -> Tensor:
        """
        Initiates the initial condition for the backward diffusion process.

        :param shape: The shape of the data
        :type shape: tuple
        :return: An initial state sampled from a simple prior distribution.
        :rtype: Tensor
        """
        T = torch.ones(1, device=self.device)
        stddev = self.schedule.stddev(t=T)
        return stddev * torch.randn(*shape, device=self.device)

    def update_step(
        self,
        x: Tensor,
        drift: Tensor,
        step_size: Tensor | float,
        step_size_sqrt: Tensor | float,
    ) -> Tensor:
        """
        Integration scheme for the sampler of the reverse diffusion process.
        Subclasses generally differ in implementation. Typically, SDEs would use an Euler-Maruyama scheme.

        :param x: Input state
        :type x: Tensor
        :param drift: Drift term of the process.
        :type drift: Tensor
        :param step_size: Step size of the integration
        :type step_size: Tensor | float
        :param step_size_sqrt: Square root of the step size scaled by the diffusion coefficient. Used in SDEs.
        :type step_size_sqrt: Tensor | float
        """
        noise = torch.randn_like(x)
        return x + drift * step_size + step_size_sqrt * noise

    def sample(
        self,
        shape: tuple,
        num_steps: int,
        keep_history: bool = False,
        schedule_type: str = "uniform",
        rho: float = 1.0,
        *labels,
        **kwargs,
    ) -> Tensor:
        """
        Sampling loop of the reverse diffusion process.

        :param shape: The shape of the data
        :type shape: tuple
        :param num_steps: Number of steps in the reverse process
        :type num_steps: int
        :param keep_history: Flag to keep the entire trajectory history. Defaults to False.
        :type keep_history: bool
        :param schedule_type: Type of variance schedule to build from. Must be one of {`uniform`, `log`, `karras`}. Defaults to `uniform`.
        :type schedule_type: str, optional
        :param rho: Parameter which controls step size across noise levels if the `karras` schedule has been specified. Defaults to 1.0
        :type rho: float, optional
        :return: Samples of the final distribution. If `keep_history=True`, returns the trajectory history
        :rtype: Tensor
        """
        timesteps, g2_t, step_size, step_size_sqrt = self.build_schedule(
            num_steps, schedule_type, rho
        )

        x = self.init_sample(shape)
        hist = self.init_history(num_steps, shape, keep_history)
        self.model.eval()

        for i, t_i in enumerate(tqdm(timesteps)):
            batch_t = t_i.expand(shape[0])
            drift = -self.schedule.drift_term(x, batch_t) + g2_t[i] * self._score(
                x, batch_t
            )
            x = self.update_step(
                x, drift, step_size[i], step_size_sqrt[i] if t_i > self.eps else 0
            )

            self.record(hist, x, idx=i, flag=keep_history)

        return self.collect(hist, x, flag=keep_history)

    def build_schedule(
        self, num_steps: int, schedule_type: str = "uniform", rho: float = 1.0
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Builds integration schedule. Uses the :class:`~diffusion_models.noise.noise_scheduler.Schedule`'s :meth:`build_variance_schedule` method.

        :param num_steps: Number of steps in the schedule
        :type num_steps: int
        :param schedule_type: Type of variance schedule to build from. Must be one of {`uniform`, `log`, `karras`}. Defaults to `uniform`.
        :type schedule_type: str, optional
        :param rho: Parameter which controls step size across noise levels if the `karras` schedule has been specified. Defaults to 1.0
        :type rho: float, optional
        :return: Returns the time schedule, the diffusion coefficient squared, the step sizes, and their square root rescaled by the diffusion coefficient.
        :rtype: tuple[Tensor, Tensor, Tensor, Tensor]
        """
        timesteps = self.schedule.build_variance_schedule(
            num_steps, schedule_type, rho
        ).to(self.device)

        dt = torch.zeros_like(timesteps)
        dt[:-1] = timesteps[:-1] - timesteps[1:]
        dt[-1] = dt[-2]
        g_t = self.model.schedule.diffusion_coeff(timesteps)
        step_size_sqrt = g_t * dt**0.5

        return timesteps, g_t**2, dt, step_size_sqrt


class StochasticHeunSampler(EulerMaruyamaSampler):
    """
    This class implements a Heun sampler with stochastic injection as described in Algorithm 2 of `arXiv:2206.00364 <https://arxiv.org/pdf/2206.00364>`_.
    In simple terms, each step predicts a new state from a previous noise scale and corrects it with the current noise scale prediction.

    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    device: torch.device
        Device on which sampling is performed.
    schedule : :class:`~diffusion_models.noise.noise_scheduler.Schedule`
        Noise schedule of the diffusion process.
    """

    def __init__(self, model: ScoreModel):
        super().__init__(model)
        self.gamma_max = math.sqrt(2) - 1

    def prob_flow_drift(self, x: Tensor, t: Tensor) -> Tensor:
        d = (x.dim() - 1) * (None,)
        score = self._score(x, t)
        force = self.schedule.drift_term(x, t)
        g2 = self.schedule.diffusion_coeff(t) ** 2
        return -force + 0.5 * g2[:, *d] * score

    def _mean_scale(self, x: Tensor, t: Tensor) -> Tensor:
        mean, _ = self.schedule.mean_stddev(torch.ones_like(x), t)
        return mean

    def sample(
        self,
        shape: tuple,
        num_steps: int,
        S_churn: float,
        S_noise: float,
        std_range: tuple[float, float],
        schedule_type: str = "uniform",
        rho: float = 1.0,
        keep_history: bool = False,
        *args,
        **kwargs,
    ):
        """
        Sampling loop of the reverse diffusion process.

        :param shape: The shape of the data
        :type shape: tuple
        :param num_steps: Number of steps in the reverse process
        :type num_steps: int
        :param S_churn: Noise injection parameter.
        :type S_churn: float
        :param S_noise: Controls the strength of the stochasticity.
        :type S_noise: float
        :param std_range: Noise range in which to inject noise.
        :type std_range: tuple[float, float]
        :param schedule_type: Type of variance schedule to build from. Must be one of {`uniform`, `log`, `karras`}. Defaults to `uniform`.
        :type schedule_type: str, optional
        :param rho: Parameter which controls step size across noise levels if the `karras` schedule has been specified. Defaults to 1.0
        :type rho: float, optional
        :param keep_history: Flag to keep the entire trajectory history. Defaults to False.
        :type keep_history: bool
        :return: Samples of the final distribution. If `keep_history=True`, returns the trajectory history
        :rtype: Tensor
        """
        timesteps, _, _, _ = self.build_schedule(
            num_steps,
            schedule_type,
            rho,
        )
        t_next_all = torch.cat([timesteps[1:], timesteps.new_tensor([self.eps])])

        x = self.init_sample(shape)
        hist = self.init_history(num_steps, shape, keep_history)
        self.model.eval()

        gamma = min(S_churn / num_steps, self.gamma_max)
        std_min, std_max = std_range
        std_cap = self.schedule.stddev(timesteps.new_tensor(1.0)) / (1 + gamma) ** 2
        std_max = min(std_max, std_cap)

        for i, t_i in enumerate(tqdm(timesteps)):
            t_next = t_next_all[i]
            std_i = self.schedule.stddev(t_i)

            # Churn
            gamma_i = gamma if std_min <= std_i <= std_max else 0.0
            std_hat = std_i * (1 + gamma_i)
            t_hat = self.schedule.invert_variance_to_time(std_hat**2)

            ratio = self._mean_scale(x, t_hat.unsqueeze(0)) / self._mean_scale(
                x, t_i.unsqueeze(0)
            )
            noise_var = (std_hat**2 - (ratio * std_i) ** 2).clamp(min=0)
            x_hat = self.update_step(
                ratio * x,
                torch.zeros_like(x),
                step_size=0,
                step_size_sqrt=noise_var.sqrt() * S_noise,
            )

            # Predictor
            batch_t_hat = t_hat.expand(shape[0])
            drift_hat = self.prob_flow_drift(x_hat, batch_t_hat)
            dt = t_hat - t_next
            x_euler = self.update_step(x_hat, drift_hat, step_size=dt, step_size_sqrt=0)

            # Corrector
            if i < num_steps - 1:
                batch_t_next = t_next.expand(shape[0])
                drift_next = self.prob_flow_drift(x_euler, batch_t_next)
                avg_drift = 0.5 * (drift_hat + drift_next)
                x = self.update_step(x_hat, avg_drift, step_size=dt, step_size_sqrt=0)
            else:
                x = x_euler

            self.record(hist, x, idx=i, flag=keep_history)

        return self.collect(hist, x, flag=keep_history)


class AngularMixin:
    """
    Mixin to introduce handling of angular data eg U(1), XY angles etc.

    Must appear *before* the sampler class in the MRO, eg::

        class AngularEMSampler(AngularMixin, EulerMaruyamaSampler):
            pass
    """

    @staticmethod
    def _wrap(x: float | Tensor) -> float | Tensor:
        """
        Wraps input data ``x`` to the range :math:`\\left[-\\pi, \\pi\\right)`.

        :param x: Input data
        :type x: float | Tensor
        :return: Data wrapped in value range :math:`\\left[-\\pi, \\pi\\right)`.
        :rtype: float | Tensor
        """
        return (x + pi) % (2 * pi) - pi

    def init_sample(self, shape: tuple) -> Tensor:
        """
        Initiates the initial condition for the backward diffusion process.
        For typical angle systems, the initial prior is the Haar uniform :math:`\\left[-\\pi, \\pi\\right)`.

        :param shape: The shape of the data
        :type shape: tuple
        :return: An initial state sampled from a simple prior distribution.
        :rtype: Tensor
        """
        return (torch.rand(*shape, device=self.device) * 2 - 1) * pi

    def update_step(
        self,
        x: Tensor,
        drift: Tensor,
        step_size: Tensor | float,
        step_size_sqrt: Tensor | float,
    ):
        """
        Wraps the result of the next class's integration step to :math:`[-\\pi, \\pi)`.
        Delegates the actual update formula to ``super().update_step``

        :param x: Input state
        :type x: Tensor
        :param drift: Drift term of the process.
        :type drift: Tensor
        :param step_size: Step size of the integration
        :type step_size: Tensor | float
        :param step_size_sqrt: Square root of the step size scaled by the diffusion coefficient. Used in SDEs.
        :type step_size_sqrt: Tensor | float
        """
        return self._wrap(super().update_step(x, drift, step_size, step_size_sqrt))

    def logq(
            self,
            initial: Tensor,
            proposal: Tensor,
            drift_initial: Tensor,
            drift_proposal: Tensor,
            step_size: Tensor | float,
            dims: int | tuple[int, ...] = -1,
        ) -> Tensor:
            """
            Log-ratio ``log q(initial|proposal) - log q(proposal|initial)`` of the Langevin proposal kernel for use in Metropolis-hastings correction.
            It corrects the asymmetry in the forward and reverse proposals in the diffusion process.

            :param initial: State before Langevin step
            :type initial: Tensor
            :param proposal: State after Langevin step
            :type proposal: Tensor
            :param drift_initial: Drift evaluated at ``initial``
            :type drift_initial: Tensor
            :param drift_proposal: Drift evaluated at ``proposal``
            :type drift_proposal: Tensor
            :param step_size: Langevin step size
            :type step_size: Tensor | float
            :param dims: Dimensions to sum the squared residual, defaults to -1
            :type dims: int | tuple[int, ...], optional
            :return: Log-ratio for the MH acceptance probability.
            :rtype: Tensor
            """
            sigma2 = 2 * step_size
            log_q_initial_given_proposal = (
                -0.5
                * torch.sum(
                    self._wrap(initial - proposal - step_size * drift_proposal) ** 2, dim=dims
                )
                / sigma2
            )
            log_q_proposal_given_initial = (
                -0.5
                * torch.sum(
                    self._wrap(proposal - initial - step_size * drift_initial) ** 2, dim=dims
                )
                / sigma2
            )
            return log_q_initial_given_proposal - log_q_proposal_given_initial


class AngularEMSampler(AngularMixin, EulerMaruyamaSampler):
    """
    This class has equipped :class:`~diffusion_models.sampling.samplers.EulerMaruyamaSampler` for diffusion model sample generation of angular data, eg U(1), XY angles etc.

    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    device : torch.device
        Device on which sampling is performed.
    schedule : :class:`Schedule`
        Noise schedule of the diffusion process.
    """

    pass


class AngularStochasticHeunSampler(AngularMixin, StochasticHeunSampler):
    """
    This class has equipped :class:`~diffusion_models.sampling.samplers.StochasticHeunSampler` for diffusion model sample generation of angular data, eg U(1), XY angles etc.

    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    device : torch.device
        Device on which sampling is performed.
    schedule : :class:`Schedule`
        Noise schedule of the diffusion process.
    """

    pass


class ScoreRescalingMixin:
    """
    Mixin that allows score rescaling of a raw model output by an arbitrary multiplicative factor.
    Works by overriding :meth:`BaseSampler._score`.

    Must appear *before* the sampler class in the MRO, eg::

        class ScaledEuler(ScoreRescalingMixin, EulerMaruyamaSampler):
            pass

    Parameters
    ----------
    *args, **kwargs
        Forwarded to the next class in the MRO
    rescaling_factor: float, optional.
        A constant scalar, defaults to 1.0

    Note
    ----
    While an implementation with a ``t``-dependent factor is possible, it is not currently necessary.
    """

    def __init__(self, *args, rescaling_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rescaling_factor = rescaling_factor

    def _score(self, x: Tensor, t: Tensor, *args) -> Tensor:
        return self.rescaling_factor * super()._score(x, t, *args)


class AngularEMSamplerWRescaling(ScoreRescalingMixin, AngularEMSampler):
    pass


class AngularSHeunSamplerWRescaling(ScoreRescalingMixin, AngularStochasticHeunSampler):
    pass

class MetropolisMixin:
    """
    Mixin that equips the sampler with an Metropolis-Hastings accept/reject method :meth:`mh_accept`.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to the next class in the MRO
    action: Callable.
        A callable function that returns the action/potential used in the Metropolis-Hastings step.
        
    .. warning:: Take care when assigning `action` positionally. It is advised to assign it as a keyword argument.
    """
    def __init__(self, *args, action: Callable, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = action

    def mh_accept(self, initial: Tensor, proposal: Tensor, logq_diff: Tensor) -> Tensor:
        """
        Metropolis-Hastings accept/reject between ``initial`` and ``proposal``.

        :param initial: Current state
        :type initial: Tensor
        :param proposal: Proposed state
        :type proposal: Tensor
        :param logq_diff: ``logq(initial|proposal) - logq(proposal|initial)``.
        :type logq_diff: Tensor
        :return: ``proposal`` where accepted, ``initial`` otherwise.
        :rtype: Tensor
        """
        d = (initial.dim() - 1) * (None,)
        log_accept = self.action(initial) - self.action(proposal) + logq_diff
        u = torch.rand(initial.shape[0], device=self.device)
        accept = u < torch.exp(log_accept)
        return torch.where(accept[:, *d], proposal, initial)

    def logq(
        self,
        initial: Tensor,
        proposal: Tensor,
        drift_initial: Tensor,
        drift_proposal: Tensor,
        step_size: Tensor | float,
        dims: int | tuple[int, ...] = -1,
    ) -> Tensor:
        """
        Log-ratio ``log q(initial|proposal) - log q(proposal|initial)`` of the Langevin proposal kernel for use in Metropolis-hastings correction.
        It corrects the asymmetry in the forward and reverse proposals in the diffusion process.

        :param initial: State before Langevin step
        :type initial: Tensor
        :param proposal: State after Langevin step
        :type proposal: Tensor
        :param drift_initial: Drift evaluated at ``initial``
        :type drift_initial: Tensor
        :param drift_proposal: Drift evaluated at ``proposal``
        :type drift_proposal: Tensor
        :param step_size: Langevin step size
        :type step_size: Tensor | float
        :param dims: Dimensions to sum the squared residual, defaults to -1
        :type dims: int | tuple[int, ...], optional
        :return: Log-ratio for the MH acceptance probability.
        :rtype: Tensor
        """
        sigma2 = 2 * step_size
        log_q_initial_given_proposal = (
            -0.5
            * torch.sum(
                (initial - proposal - step_size * drift_proposal) ** 2, dim=dims
            )
            / sigma2
        )
        log_q_proposal_given_initial = (
            -0.5
            * torch.sum((proposal - initial - step_size * drift_initial) ** 2, dim=dims)
            / sigma2
        )
        return log_q_initial_given_proposal - log_q_proposal_given_initial


class MAALASampler(MetropolisMixin, EulerMaruyamaSampler):
    """
    This class implements a Metropolis-Adjusted Annealed Langevin sampler as described in Algorithm 2 of `J. High Energ. Phys. 2026, 111 (2026) <https://doi.org/10.1007/JHEP03(2026)111>`_.

    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.
    action: Callable
        The action or log-likelihood as a function of state.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    action : Callable
        Log-likelihood function used in the Metropolis-Hastings accept/reject step.
    device: torch.device
        Device on which sampling is performed.
    schedule : :class:`~diffusion_models.noise.noise_scheduler.Schedule`
        Noise schedule of the diffusion process.
    """
    def __init__(self, model: ScoreModel, action: Callable, **kwargs):
        super().__init__(model, action=action, **kwargs)

    def sample(
        self,
        shape: tuple,
        num_steps: int,
        l_steps: int,
        l_stepsize: float,
        mh_threshold: float = 0.95,
        schedule_type: str = "uniform",
        rho: float = 1.0,
        keep_history: bool = False,
        *labels,
        **kwargs,
    ) -> Tensor:
        """
        :param l_steps: Number of inner Langevin steps per annealing step.
        :type l_steps: int
        :param l_stepsize: Step size ``alpha`` used for the inner Langevin dynamics
            (constant across the schedule, unlike the outer predictor step).
        :type l_stepsize: float
        :param mh_threshold: Fraction of the schedule (in ``[0,1]``) after which inner steps
            switch from plain (unadjusted) Langevin to MH-corrected.
        :type mh_threshold: float
        :param schedule_type: See :meth:`build_schedule`.
        :type schedule_type: str, optional
        :param rho: See :meth:`build_schedule`.
        :type rho: float, optional
        """
        timesteps, _, _, _ = self.build_schedule(
            num_steps, schedule_type, rho
        )

        y = self.init_sample(shape)
        hist = self.init_history(num_steps, shape, keep_history)
        self.model.eval()

        for i, t_i in enumerate(tqdm(timesteps)):
            batch_t = t_i.expand(shape[0])

            # Predictor --- One step at the schedule's noise level ---
            # drift = self._score(x, batch_t, *labels)
            # y = self.update_step(x, g2_t[i] * drift, step_size[i], step_size_sqrt[i])

            # Corrector --- ``l_steps`` of Langevin dynamics ---
            alpha = l_stepsize
            alpha_sqrt = math.sqrt(2 * alpha)

            for _ in range(l_steps):
                drift_initial = self._score(y, batch_t, *labels)
                y_hat = self.update_step(y, drift_initial, alpha, alpha_sqrt)

                if i >= mh_threshold * num_steps:
                    drift_proposal = self._score(y_hat, batch_t, *labels)
                    logq_diff = self.logq(
                        y,
                        y_hat,
                        drift_initial,
                        drift_proposal,
                        alpha,
                        dims=tuple(range(1, y.ndim)),
                    )
                    y = self.mh_accept(y, y_hat, logq_diff)
                else:
                    y = y_hat
            # x = y
            self.record(hist, y, idx=i, flag=keep_history)

        return self.collect(hist, y, flag=keep_history)


class MAALASamplerWRescaling(ScoreRescalingMixin, MAALASampler):
    pass


class AngularMAALASamplerWRescaling(AngularMixin, MAALASamplerWRescaling):
    pass



class PoorManEulerSampler(MetropolisMixin, EulerMaruyamaSampler):
    """
    This class implements a hybrid sampling system. It uses an Euler-Maruyama sampler for the majority of a trajectory.
    The rest follows a Metropolis-adjusted Langevin update similar to :class:`MAALASampler`.


    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.
    action: Callable
        The action or log-likelihood as a function of state.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    action : Callable
        Log-likelihood function used in the Metropolis-Hastings accept/reject step.
    device: torch.device
        Device on which sampling is performed.
    schedule : :class:`~diffusion_models.noise.noise_scheduler.Schedule`
        Noise schedule of the diffusion process.
    """
    def __init__(self, model: ScoreModel, action: Callable, **kwargs):
        super().__init__(model, action=action, **kwargs)

    def sample(
        self,
        shape: tuple,
        num_steps: int,
        l_steps: int,
        l_stepsize: float,
        mh_threshold: float = 0.95,
        schedule_type: str = "uniform",
        rho: float = 1.0,
        keep_history: bool = False,
        *labels,
        **kwargs,
    ) -> Tensor:
        """
        :param l_steps: Number of inner Langevin steps per annealing step.
        :type l_steps: int
        :param l_stepsize: Step size ``alpha`` used for the inner Langevin dynamics
            (constant across the schedule, unlike the outer predictor step).
        :type l_stepsize: float
        :param mh_threshold: Fraction of the schedule (in ``[0,1]``) after which inner steps
            switch from plain (unadjusted) Langevin to MH-corrected.
        :type mh_threshold: float
        :param schedule_type: See :meth:`build_schedule`.
        :type schedule_type: str, optional
        :param rho: See :meth:`build_schedule`.
        :type rho: float, optional
        """
        timesteps, g2_t, step_size, step_size_sqrt = self.build_schedule(
            num_steps, schedule_type, rho
        )

        x = self.init_sample(shape)
        hist = self.init_history(num_steps, shape, keep_history)
        self.model.eval()

        for i, t_i in enumerate(tqdm(timesteps)):
            batch_t = t_i.expand(shape[0])

            # Predictor --- One step at the schedule's noise level ---
            if t_i >= 10 * self.eps:
                drift = self._score(x, batch_t, *labels)
                x = self.update_step(x, g2_t[i] * drift, step_size[i], step_size_sqrt[i])
            else:
                # Corrector --- ``l_steps`` of Langevin dynamics ---
                alpha = l_stepsize
                alpha_sqrt = math.sqrt(2 * alpha)

                for _ in range(l_steps):
                    drift_initial = self._score(x, batch_t, *labels)
                    x_hat = self.update_step(x, drift_initial, alpha, alpha_sqrt)

                    drift_proposal = self._score(x_hat, batch_t, *labels)
                    logq_diff = self.logq(
                        x,
                        x_hat,
                        drift_initial,
                        drift_proposal,
                        alpha,
                        dims=tuple(range(1, x.ndim)),
                    )
                    x = self.mh_accept(x, x_hat, logq_diff)

            self.record(hist, x, idx=i, flag=keep_history)

        return self.collect(hist, x, flag=keep_history)


class PoorManHeunSampler(MetropolisMixin, StochasticHeunSampler):
    """
    This class implements a hybrid sampling system. It uses an Euler-Maruyama sampler for the majority of a trajectory.
    The rest follows a Metropolis-adjusted Langevin update similar to :class:`MAALASampler`.


    Parameters
    ----------
    model : :class:`ScoreModel`
        Trained score model.
    action: Callable
        The action or log-likelihood as a function of state.

    Attributes
    ----------
    model : :class:`ScoreModel`
        Score model used during reverse-time integration.
    action : Callable
        Log-likelihood function used in the Metropolis-Hastings accept/reject step.
    device: torch.device
        Device on which sampling is performed.
    schedule : :class:`~diffusion_models.noise.noise_scheduler.Schedule`
        Noise schedule of the diffusion process.
    """
    def __init__(self, model: ScoreModel, action: Callable, **kwargs):
        super().__init__(model, action=action, **kwargs)


    def sample(
        self,
        shape: tuple,
        num_steps: int,
        l_steps: int,
        l_stepsize: float,
        S_churn: float,
        S_noise: float,
        std_range: tuple[float, float],
        mh_threshold: float = 0.95,
        schedule_type: str = "uniform",
        rho: float = 1.0,
        keep_history: bool = False,
        *labels,
        **kwargs,
    ) -> Tensor:
        """
        :param l_steps: Number of inner Langevin steps per annealing step.
        :type l_steps: int
        :param l_stepsize: Step size ``alpha`` used for the inner Langevin dynamics
            (constant across the schedule, unlike the outer predictor step).
        :type l_stepsize: float
        :param mh_threshold: Fraction of the schedule (in ``[0,1]``) after which inner steps
            switch from plain (unadjusted) Langevin to MH-corrected.
        :type mh_threshold: float
        :param schedule_type: See :meth:`build_schedule`.
        :type schedule_type: str, optional
        :param rho: See :meth:`build_schedule`.
        :type rho: float, optional
        """
        timesteps, _, _, _ = self.build_schedule(
            num_steps, schedule_type, rho
        )
        t_next_all = torch.cat([timesteps[1:], timesteps.new_tensor([self.eps])])

        x = self.init_sample(shape)
        hist = self.init_history(num_steps, shape, keep_history)
        self.model.eval()

        gamma = min(S_churn / num_steps, self.gamma_max)
        std_min, std_max = std_range
        std_cap = self.schedule.stddev(timesteps.new_tensor(1.0)) / (1 + gamma) ** 2
        std_max = min(std_max, std_cap)

        for i, t_i in enumerate(tqdm(timesteps)):
            batch_t = t_i.expand(shape[0])

            if i <= mh_threshold * num_steps:
                # Predictor --- One step at the schedule's noise level ---
                t_next = t_next_all[i]
                std_i = self.schedule.stddev(t_i)

                # Churn
                gamma_i = gamma if std_min <= std_i <= std_max else 0.0
                std_hat = std_i * (1 + gamma_i)
                t_hat = self.schedule.invert_variance_to_time(std_hat**2)

                ratio = self._mean_scale(x, t_hat.unsqueeze(0)) / self._mean_scale(
                    x, t_i.unsqueeze(0)
                )
                noise_var = (std_hat**2 - (ratio * std_i) ** 2).clamp(min=0)
                x_hat = self.update_step(
                    ratio * x,
                    torch.zeros_like(x),
                    step_size=0,
                    step_size_sqrt=noise_var.sqrt() * S_noise,
                )

                # Predictor
                batch_t_hat = t_hat.expand(shape[0])
                drift_hat = self.prob_flow_drift(x_hat, batch_t_hat)
                dt = t_hat - t_next
                x_euler = self.update_step(x_hat, drift_hat, step_size=dt, step_size_sqrt=0)

                batch_t_next = t_next.expand(shape[0])
                drift_next = self.prob_flow_drift(x_euler, batch_t_next)
                avg_drift = 0.5 * (drift_hat + drift_next)
                x = self.update_step(x_hat, avg_drift, step_size=dt, step_size_sqrt=0)
            else:
                # Corrector --- ``l_steps`` of Langevin dynamics ---
                alpha = l_stepsize
                alpha_sqrt = math.sqrt(2 * alpha)

                for _ in range(l_steps):
                    drift_initial = self._score(x, batch_t, *labels)
                    x_hat = self.update_step(x, drift_initial, alpha, alpha_sqrt)

                    drift_proposal = self._score(x_hat, batch_t, *labels)
                    logq_diff = self.logq(
                        x,
                        x_hat,
                        drift_initial,
                        drift_proposal,
                        alpha,
                        dims=tuple(range(1, x.ndim)),
                    )
                    x = self.mh_accept(x, x_hat, logq_diff)

            self.record(hist, x, idx=i, flag=keep_history)

        return self.collect(hist, x, flag=keep_history)


class AngularPoorManEulerSamplerWRescaling(AngularMixin, ScoreRescalingMixin, PoorManEulerSampler):
    pass

class AngularPoorManHeunSamplerWRescaling(AngularMixin, ScoreRescalingMixin, PoorManHeunSampler):
    pass


@torch.no_grad()
def ot_sampler(
    model: torch.nn.Module,
    shape: tuple,
    num_steps: int = 50,
    *labels,
    history: bool = False,
):
    """
    Euler sampler for flow matching (deterministic).

    Args:
        model: trained FlowMatchingModel (time-dependent velocity field)
        shape: tuple, shape of samples (batch_size, dims...)
        num_steps: number of integration steps
        *labels: optional conditioning inputs
        history: if True, returns all intermediate states
    """
    output = []
    batch_size = shape[0]
    device = model.device

    # Start from source distribution ~ N(0, I)
    x = torch.randn(*shape, device=device)

    # Integration parameters
    dt = 1.0 / num_steps
    timesteps = torch.linspace(0, 1, num_steps, device=device)

    model.eval()

    for i, t_i in enumerate(tqdm(timesteps)):
        # Expand scalar timestep for batch
        batch_t = t_i.expand(batch_size)

        # Predict velocity
        v = model(x, batch_t, *labels)

        # Euler integration step
        x = x + dt * v

        if history:
            output.append(x.clone())

    if history:
        return torch.stack(output)  # [num_steps, batch_size, ...]
    return x
