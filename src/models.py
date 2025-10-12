from typing import Optional
import torch
from torch import Tensor
from torch.nn import Module

from noise_scheduler import Schedule, GeometricSchedule


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}

        """Copy initial parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update moving averages with current parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Backup current params and apply EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


class ScoreModel(Module):
    def __init__(
        self,
        network: Module,
        schedule: Optional[Schedule] = None,
        device: str = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.device = device
        self.history = []
        self.dims = None
        self.ema = EMA(self)
        self.schedule = schedule or GeometricSchedule()

    def forward(self, x: Tensor, t: Tensor, *labels):
        d = (x.dim() - 1) * [
            None,
        ]
        return self.network(x, t, *labels) / self.schedule.stddev(t)[:, *d]

    def loss_fn(self, batch, *labels, eps: float = 1e-5):
        if self.dims is None:
            self.dims = tuple(range(1, batch.dim()))
        d = (batch.dim() - 1) * [
            None,
        ]

        z = torch.randn_like(batch)

        random_t = torch.rand(batch.shape[0], device=self.device) * (1.0 - eps) + eps
        mean, std = self.schedule.mean_stddev(batch, random_t)
        std = std[:, *d]

        perturbed_x = mean + z * std

        score = self.forward(perturbed_x, random_t, *labels)
        return 0.5 * torch.mean(torch.sum((score * std + z) ** 2, dim=self.dims))

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
        self.ema.shadow = save_dict.get("EMA", {})

    def _save_weights(self, file_path):
        save_dict = {
            "MODEL_STATE": self.state_dict(),
            "EMA": self.ema.shadow,
            "HISTORY": self.history,
        }
        torch.save(save_dict, file_path)
        print(f"Weights saved at {file_path}...")


class EnergyBasedModel(ScoreModel):
    def __init__(
        self,
        network: Module,
        schedule: Optional[Schedule] = None,
        device: str = None,
    ):
        super().__init__(network, schedule, device)

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
        std = std[:, *d]

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
        self.ema = EMA(self)

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
        self.ema.shadow = save_dict.get("EMA", {})

    def _save_weights(self, file_path):
        save_dict = {
            "MODEL_STATE": self.state_dict(),
            "EMA": self.ema.shadow,
            "HISTORY": self.history,
        }
        torch.save(save_dict, file_path)
        print(f"Weights saved at {file_path}...")
