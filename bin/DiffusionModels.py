from math import log
from typing import Optional
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module


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
				new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
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


class MarginalProb:
	def __init__(self, sigma: float = 2.0) -> None:
		self.logsigma = log(sigma)

	def get_mean_stddev(self, x: Tensor, t: Tensor) -> Tensor:
		return x, self.stddev(t)
	
	def stddev(self, t: Tensor) -> Tensor:
		return torch.sqrt((torch.exp(2 * t * self.logsigma) - 1.0) / 2 / self.logsigma)

	def diffusion_coeff(self, t: Tensor) -> Tensor:
		return torch.exp(t * self.logsigma)

	def drift(self, x: Tensor, t: Tensor) -> Tensor:
		return torch.zeros_like(x)


class ScoreModel(Module):
	def __init__(
		self,
		network: Module,
		marginal_prob_sigma: float=25,
		device: Optional[str] = None,
	) -> None:
		super().__init__()
		self.network = network
		self.marginal_prob = MarginalProb(sigma=marginal_prob_sigma)
		self.device = device
		self.history = []
		self.dims = None
		self.ema = EMA(self)


	def forward(self, x: Tensor, t: Tensor, *labels):
		d = (x.dim() - 1) * [None,]
		return self.network(x, t, *labels) / self.marginal_prob.stddev(t)[:, *d]
	

	def loss_fn(self, batch, *labels, eps: float = 1e-5):
		if self.dims is None:
			self.dims = tuple(range(1, batch.dim()))
		d = (batch.dim() - 1) * [None,]

		z = torch.randn_like(batch)

		random_t = torch.rand(batch.shape[0], device=self.device) * (1.0 - eps) + eps
		mean, std = self.marginal_prob.get_mean_stddev(batch, random_t); std = std[:, *d]
		
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
		save_dict = {"MODEL_STATE": self.state_dict(), "EMA": self.ema.shadow, "HISTORY": self.history}
		torch.save(save_dict, file_path)
		print(f"Weights saved at {file_path}...")