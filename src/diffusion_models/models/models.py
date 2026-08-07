from __future__ import annotations

import torch
import warnings
from torch import Tensor
from typing import Optional
from torch.nn import Module

from pathlib import Path
from contextlib import contextmanager
from diffusion_models.noise.noise_scheduler import Schedule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any

from diffusion_models.config.config import NETWORK_REGISTRY, SCHEDULE_REGISTRY, ExperimentConfig


class ExponentialMovingAverage:
    """
    This class handles the exponential moving average of a network's parameters.

    :param model: Network whose parameters are tracked
    :type model: Module
    :param decay_rate: Exponential moving average decay rate. Closer to 1.0 means slower-changing averages.
    :param decay_rate: float
    """    
    def __init__(self, model: Module, decay_rate: float = 0.999):
        self.decay_rate = decay_rate
        self.model = model
        self.shadow: dict[str, Tensor] = {}
        self.backup: dict[str, Tensor] = {}

        # Copy initial parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update the moving average with the model's current parameters."""
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            self.shadow[name] = (
                (1.0 - self.decay_rate) * param.data + self.decay_rate * self.shadow[name]
            ).clone()

    def apply_shadow(self):
        """Backup the current parameters and swap-in the ExponentialMovingAverage weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore the parameters that were active before :meth:`apply_shadow`."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}

    @contextmanager
    def average_parameters(self):
        """
        Temporarily swap in ExponentialMovingAverage weights; restores the original weights even if an exception is raised inside the block.

        Usage::

            with score_model.exponential_moving_average.average_parameters():
                ...
        """
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def to(self, *args, **kwargs) -> ExponentialMovingAverage:
        """Move the tracked shadow/backup tensor (eg to a new device)."""
        self.shadow = {k: v.to(*args, **kwargs) for k, v in self.shadow.items()}
        self.backup = {k: v.to(*args, **kwargs) for k, v in self.backup.items()}
        return self

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class ScoreModel(Module):
    """
    Equips a network with a :class:`~diffusion_models.noise.noise_scheduler.Schedule` to form a trainable score model,
    with exponential moving average weight tracking and checkpointing.

    :param network: Network used to model the score.
    :type network: Module
    :param schedule: Noise schedule defining the diffusion process.
    :type schedule: Schedule
    :param device: Device to move the model to. Defaults to the network's device if not given.
    :type device: str | torch.device | None, optional
    :param decay_rate: Decay rate for the ExponentialMovingAverage of the model's weights
    :type decay_rate: float
    """
    def __init__(
        self,
        network: Module,
        schedule: Schedule,
        device: str | torch.device | None = None,
        decay_rate: float = 0.999,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.network = network
        self.schedule = schedule
        self.history: list[float] = []

        if device is not None:
            self.to(device)

        self.exponential_moving_average = ExponentialMovingAverage(self, decay_rate=decay_rate)

        if kwargs:
            warnings.warn(
                f"ScoreModel received unused keyword arguments: {sorted(kwargs)}."
                "This usually mean a YAML config has outdated or invalid keys.",
                stacklevel=2,
            )
        self.extra_kwargs = kwargs

        # Populated by `from_config`; kept so Trainer can log exactly what
        # config was used to build this model.
        self.config: dict[str, Any] | None = None

    @property
    def device(self) -> torch.device:
        """Inferred from the network's parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to(self, *args, **kwargs) -> ScoreModel:
        result = super().to(*args, **kwargs)
        if hasattr(result, "exponential_moving_average"):
            result.exponential_moving_average.to(*args, **kwargs)
        return result

    def forward(self, x: Tensor, t: Tensor, *labels: Tensor | tuple[Tensor, ...]) -> Tensor:
        """
        :param x: Input state.
        :type x: Tensor
        :param t: Time values at which to evaluate.
        :type t: Tensor
        :return: Score estimate at ``(x, t)``.
        :rtype: Tensor
        """
        d = (x.dim() - 1) * (None,)
        return self.network(x, t, *labels) / self.schedule.stddev(t)[:, *d]

    def loss_fn(self, batch: Tensor, *labels: Tensor | tuple[Tensor, ...]) -> Tensor:
        """
        Evaluate a denoising score-matching loss.

        :param batch: Clean data batch
        :type batch: Tensor
        :return: Scalar loss
        :rtype: Tensor
        """
        eps = self.schedule.eps
        z = torch.randn_like(batch)

        random_t = torch.rand(batch.shape[0], device=self.device).clamp(min=eps)
        mean, std = self.schedule.mean_stddev(batch, random_t)
        perturbed_x = mean + z * std

        score = self.forward(perturbed_x, random_t, *labels)
        return 0.5 * torch.mean((score * std + z) ** 2)

    def train_step(self, batch: Tensor, optimizer: Optimizer, *labels: Tensor | tuple[Tensor, ...], lr_scheduler: LRScheduler | None = None) -> Tensor:
        """
        Runs a single optimization step and updates the ExponentialMovingAverage weights.

        :param batch: Clea data batch.
        :type batch: Tensor
        :param optimizer: Optimizer to use for the step.
        :type optimizer: Optimizer
        :param lr_scheduler: Optional LR scheduler, defaults to None
        :type lr_scheduler: LRScheduler | None, optional
        :return: Scalar loss for this step.
        :rtype: Tensor
        """
        self.train()
        optimizer.zero_grad(set_to_none=True)

        loss = self.loss_fn(batch, *labels)
        loss.backward()
        optimizer.step()
        self.exponential_moving_average.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        self.history.append(loss.item())
        return loss

    def _load_weights(self, file_path: str | Path):
        """
        Load weights from file.

        :param file_path: File path of weights to be loaded from.
        :type file_path: str | Path
        """        
        save_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        self.load_state_dict(save_dict["MODEL_STATE"])
        self.history = save_dict.get("HISTORY", [])
        self.exponential_moving_average.load_state_dict(save_dict.get("EMA", {}))

    def _save_weights(self, file_path: str | Path):
        """
        Save weight to file. If the directory doesn't exit it is created.

        :param file_path: File path of file to save weights in.
        :type file_path: str | Path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "MODEL_STATE": self.state_dict(),
            "EMA": self.exponential_moving_average.state_dict(),
            "HISTORY": self.history,
        }
        torch.save(save_dict, file_path)
        print(f"Weights saved at {file_path}...")

    @classmethod
    def from_config(cls, config: ExperimentConfig, device: str | torch.device | None = None,) -> ScoreModel:
        """
        Build a ScoreModel instance from a parsed config `ExperimentConfig`. `network`/`schedule` names are resolved via registries (NETWORK_REGISTRY/SCHEDULE_REGISTRY).
        Such classes need to be registered before this constructor is called, otherwise they won't be recognized.


        :param config: Parsed experiment config (see `ExperimentConfig.from_yaml`)
        :type config: ExperimentConfig
        :param device: Overrides `model.device` in the YAML if given, defaults to None.
        :type device: str | torch.device | None, optional
        :return: A constructed ScoreModel.
        :rtype: ScoreModel
        """
        network = NETWORK_REGISTRY.build(config.network.name, config.network.params)
        schedule = SCHEDULE_REGISTRY.build(config.schedule.name, config.schedule.params)
        resolved_device = device if device is not None else config.model.device

        model = cls(
            network=network,
            schedule=schedule,
            device=resolved_device,
            decay_rate=config.model.decay_rate,
        )
        model.config = config
        return model

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, device: str | torch.device | None = None,) -> ScoreModel:
        """
        Convenience wrapper: parses a YAML file into an ExperimentConfig and builds the model in one call. Equivalent to::

            config = ExperimentConfig.from_yaml(yaml_path)
            model = ScoreModel.from_config(config, device=device)

        :param yaml_path: Path to the YAML config file.
        :type yaml_path: str | Path
        :param device: Overrides `model.device` in the YAML if given, defaults to None.
        :type device: str | torch.device | None, optional
        :return: A constructed ScoreModel.
        :rtype: ScoreModel
        """
        config = ExperimentConfig.from_yaml(yaml_path)
        return cls.from_config(config, device=device)


class EnergyBasedModel(ScoreModel):
    def __init__(
        self,
        network: Module,
        schedule: Optional[Schedule] = None,
        device: str = None,
        **kwargs,
    ):
        super().__init__(network, schedule, device, **kwargs)

    def energy(self, x: Tensor, t: Tensor, *labels):
        score = self.network(x, t, *labels)
        return (
            -0.5
            * torch.sum(score**2, dim=tuple(range(1, x.dim())))
            / self.schedule.stddev(t)
        )

    def forward(self, x: Tensor, t: Tensor, *labels, create_graph=False):
        x.requires_grad_(True)
        E = self.energy(x, t, *labels)
        return torch.autograd.grad(
            E,
            x,
            grad_outputs=torch.ones_like(E),
            create_graph=create_graph,
            # retain_graph=True,
            only_inputs=True,
        )[0]

    def loss_fn(self, batch, *labels, eps: float = 1e-5):
        if self.dims is None:
            self.dims = tuple(range(1, batch.dim()))
        d = (batch.dim() - 1) * [
            None,
        ]

        z = torch.randn_like(batch)

        random_t = torch.rand(batch.shape[0], device=self.device) * (1.0 - eps) + eps
        mean, std = self.schedule.mean_stddev(batch, random_t)

        perturbed_x = mean + z * std

        score = self.forward(perturbed_x, random_t, *labels, create_graph=True)
        return 0.5 * torch.mean(torch.sum((score * std + z) ** 2, dim=self.dims))


class FlowMatchingModel(Module):
    def __init__(self, network: Module, device: str = None) -> None:
        super().__init__()
        self.network = network
        self.device = device
        self.history = []
        self.dims = None
        self.exponential_moving_average = ExponentialMovingAverage(self)

    def forward(self, x: Tensor, t: Tensor, *labels):
        """
        Forward pass of velocity field network.
        Args:
                x: current state (interpolated sample)
                t: current time (in [0,1])
                *labels: optional conditioning
        Returns:
                velocity prediction (same shape as x)
        """
        return self.network(x, t, *labels)

    def loss_fn(self, batch: Tensor, *labels, eps: float = 1e-5):
        """
        Flow matching loss.
        Args:
                batch: samples from target distribution
        """
        if self.dims is None:
            self.dims = tuple(range(1, batch.dim()))

        z = torch.randn_like(batch)  # source

        # pick random interpolation times
        random_t = torch.rand(batch.shape[0], 1, device=self.device)

        # interpolate between source and target
        z_t = (1.0 - random_t) * z + random_t * batch

        # true velocity is displacement between endpoints
        v_target = batch - z

        # predicted velocity from network
        v_pred = self.forward(z_t, random_t, *labels)

        return 0.5 * torch.mean(torch.sum((v_pred - v_target) ** 2, dim=self.dims))

    def train_step(self, batch, optimizer, *labels, scheduler=None):
        optimizer.zero_grad()

        loss = self.loss_fn(batch, *labels)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return loss

    def _load_weights(self, file_path):
        save_dict = torch.load(file_path, map_location=self.device, weights_only=True)
        self.load_state_dict(save_dict["MODEL_STATE"])
        self.history = save_dict.get("HISTORY", [])
        self.exponential_moving_average.shadow = save_dict.get("EMA", {})

    def _save_weights(self, file_path):
        save_dict = {
            "MODEL_STATE": self.state_dict(),
            "EMA": self.exponential_moving_average.shadow,
            "HISTORY": self.history,
        }
        torch.save(save_dict, file_path)
        print(f"Weights saved at {file_path}...")
