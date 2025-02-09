import torch
from tqdm import tqdm

def sampler(
	model: torch.nn.Module, shape: tuple, num_steps: int, history: bool = False, eps=1e-3
):
	output = []
	batch_size = shape[0]
	d = len(shape[1:]) * [None,]
	step_size = 1 / num_steps
	step_size_sqrt = step_size**0.5

	model.eval()

	t_0 = torch.ones(1, device=model.device)
	std = model.marginal_prob.stddev(t_0)
	x = torch.randn(*shape, device=model.device) * std

	with torch.no_grad():
		for t_i in tqdm(
			torch.linspace(1, eps, num_steps, device=model.device)
		):
			batch_t = t_i.expand(batch_size)
			g_t = model.marginal_prob.diffusion_coeff(batch_t)[:, *d]

			score = model.forward(x, batch_t)
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