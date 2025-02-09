from math import log
from typing import Optional
from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn import Module


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


	def forward(self, x: Tensor, t: Tensor, *labels):
		d = (x.dim() - 1) * [None,]
		return self.network(x, t) / self.marginal_prob.stddev(t)[:, *d]
	

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


class ConditionalScoreModel(ScoreModel):
	def __init__(self, network, marginal_prob_sigma = 25, device = None):
		super().__init__(network, marginal_prob_sigma, device)
	
	def forward(self, x, t, label):
		return self.network(x, t, label) / self.marginal_prob.stddev(t)

	def train_step(self, x, labels, optimizer, eps, scheduler=None):
		if self.dims is None:
			self.dims = tuple(range(1, len(x.shape)))
		random_t = torch.rand(x.shape[0], device=self.device) * (1.0 - eps) + eps
		z = torch.randn_like(x, )
		mean, std = self.marginal_prob.get_mean_stddev(x, random_t)
		perturbed_x = mean + z * std

		optimizer.zero_grad()

		score = self.forward(perturbed_x, random_t, labels)
		loss = 0.5 * torch.mean(torch.sum((score * std + z) ** 2, dim=self.dims))

		loss.backward()
		optimizer.step()

		if scheduler is not None:
			scheduler.step()

		return loss

	def sampler(
		self, labels, batch_size: int, shape: tuple, num_steps: int, history: bool = False, eps=1e-5
	):
		output = []
		step_size = 1 / num_steps
		step_size_sqrt = step_size**0.5

		self.eval()

		t_0 = torch.ones(batch_size, device=self.device)
		std = self.marginal_prob.stddev(t_0)
		x = torch.randn(batch_size, *shape, device=self.device) * std
		with torch.no_grad():
			for t_i in tqdm(
				torch.linspace(1, eps, num_steps, device=self.device)
			):
				batch_t = t_i * t_0
				g_t = self.marginal_prob.diffusion_coeff(batch_t)

				score = self.forward(x, batch_t, labels)
				noise = torch.randn_like(x) if t_i > eps else torch.zeros_like(x)

				x = (
					x
					+ step_size * g_t**2 * score
					+ step_size_sqrt * g_t * noise
				)
				if history:
					output.append(x)
			if history:
				return torch.stack(output)
		return x