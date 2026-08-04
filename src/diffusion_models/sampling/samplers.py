import math
import torch

from tqdm import tqdm
from torch import Tensor
from abc import ABC, abstractmethod

from diffusion_models.models.models import ScoreModel
from diffusion_models.noise.noise_scheduler import Schedule


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
    schedule : :class:`Schedule`
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

    def init_history(self, num_steps: int, shape: tuple) -> Tensor:
        return torch.empty(num_steps, *shape, device=self.device)

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
    schedule : :class:`Schedule`
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
        hist = self.init_history(num_steps, shape)
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
        Builds integration schedule. Uses the :class:`Schedule`'s :meth:`build_variance_schedule` method.

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


# def em_sampler(
#     model: torch.nn.Module,
#     shape: tuple,
#     num_steps: int,
#     *labels,
#     history: bool = False,
#     eps=1e-3,
# ):
#     """EULER-MARUYAMA stochastic sampler ~ √dt"""
#     output = []
#     batch_size = shape[0]
#     device = model.device

#     timesteps = torch.linspace(1, eps, num_steps, device=device)

#     g_t = model.schedule.diffusion_coeff(timesteps)
#     step_size = 1 / num_steps
#     step_size_sqrt = step_size**0.5

#     t0 = torch.ones(1, device=device)
#     std = model.schedule.stddev(t0)
#     x = torch.randn(*shape, device=device) * std

#     model.eval()

#     for i, t_i in enumerate(tqdm(timesteps)):
#         batch_t = t_i.expand(batch_size)

#         drift = g_t[i] ** 2 * model(x, batch_t, *labels)
#         noise = g_t[i] * torch.randn_like(x) if t_i > eps else 0.0

#         x = x + drift * step_size + step_size_sqrt * noise

#         if history:
#             output.append(x)
#     if history:
#         return torch.stack(output)
#     return x


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
