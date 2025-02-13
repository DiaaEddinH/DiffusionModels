import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from typing import Optional



class GaussianFourierProjection(Module):
    def __init__(self, embed_dim: int, scale: float = 30., device: Optional[str]=None) -> None:
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(embed_dim // 2, device=device) * scale, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)


class Embedding(Module):
	def __init__(self, embed_dim: int, activation: Module = torch.nn.SiLU(), device: Optional[str | torch.device]=None):
		super().__init__()
		self.act = activation
		self.embed = torch.nn.Sequential(
			GaussianFourierProjection(embed_dim=embed_dim, device=device),
			torch.nn.Linear(embed_dim, embed_dim, device=device),
		)
	
	def forward(self, x: Tensor):
		return self.act(self.embed(x))

class LinearNet(Module):
	def __init__(
			self, 
			in_channels: int = 2, 
			channels: list = [32, 32], 
			time_channels: int = 32, 
			activation: Module = torch.nn.LeakyReLU(),
			dropout_rate: float = 0.2,
			device: Optional[str | torch.device] = None,
			**kwargs
		) -> None:
		super().__init__()
		self.time_embed = Embedding(embed_dim=time_channels, device=device)
        # Network architecture layers
		self.channels = [in_channels] + channels
		self.t_linears = ModuleList([
			torch.nn.Linear(time_channels, c, device=device) for c in channels
		])
		self.layers = ModuleList([
                torch.nn.Linear(self.channels[i], self.channels[i + 1], device=device)
                for i in range(len(self.channels) - 1)
        ])
		self.final = torch.nn.Linear(channels[-1], in_channels, device=device)
		self.dropout = torch.nn.Dropout(dropout_rate)
		self.act = activation
        # Model's parameter

	def forward(self, *inputs: tuple):
		return self._forward_impl(*inputs)

	def forward(self, x: Tensor, t: Tensor) -> torch.Tensor:
		t_emb = self.time_embed(t)

		for i, layer in enumerate(self.layers):
			x = layer(x)
			x += self.t_linears[i](t_emb)
			x = self.dropout(x)
			x = self.act(x)

		return self.final(x)

class CNet(Module):
	def __init__(
			self, 
			in_channels: int = 2, 
			channels: list = [64, 128, 256], 
			time_channels: int = 32, 
			activation: Module = torch.nn.SiLU(),
			dropout_rate: float = 0.2,
			padding_mode: str = "circular",
			device: Optional[str | torch.device] = None,
			**kwargs
		) -> None:
		super().__init__()
		self.time_embed = Embedding(embed_dim=time_channels, device=device)
		self.channels = channels
		self.channels_r = channels[::-1]

		self.t_linears = ModuleList([
			torch.nn.Linear(time_channels, c, device=device) for c in self.channels + self.channels_r[1:]
		])

		self.norm_layers = ModuleList([
			torch.nn.GroupNorm(c//4, c, device=device) for c in self.channels
		] + [
			torch.nn.GroupNorm(c//4, c, device=device) for c in self.channels_r[1:]
		])
		self.down_layers = ModuleList([
			torch.nn.Conv2d(in_channels, self.channels[0], kernel_size=3, stride=1, bias=False, padding=1, padding_mode=padding_mode, device=device)
		] + [
			torch.nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, bias=False, padding=1, padding_mode=padding_mode, device=device)
			for c_in, c_out in zip(self.channels, self.channels[1:])
		])

		self.up_layers = ModuleList([
			torch.nn.Conv2d(self.channels[-1], self.channels[-2], kernel_size=3, stride=1, bias=False, padding=1, padding_mode=padding_mode, device=device)
		] + [
			torch.nn.Conv2d(2 * c_in, c_out, kernel_size=3, stride=1, bias=False, padding=1, padding_mode=padding_mode, device=device)
			for c_in, c_out in zip(self.channels_r[1:], self.channels_r[2:])
		])
		
		self.act = activation
		self.dropout = torch.nn.Dropout(dropout_rate)
		self.out = torch.nn.Conv2d(channels[0], 1, kernel_size=3, padding=1, padding_mode=padding_mode, device=device)


	def forward(self, *inputs: tuple):
		return self._forward_impl(*inputs)

	def _forward_impl(self, x: Tensor, t: Tensor) -> Tensor:
		skip = []
		t_emb = self.time_embed(t)

		for i, layer in enumerate(self.down_layers):
			x = layer(x)
			x += self.t_linears[i](t_emb)[..., None, None]; 
			x = self.dropout(x)
			x = self.act(self.norm_layers[i](x))
			if i != len(self.down_layers) - 1:
				skip.append(x)
		
		for n, layer in enumerate(self.up_layers):
			x = layer(x)
			x += self.t_linears[i + n + 1](t_emb)[..., None, None]
			x = self.dropout(x)
			x = self.act(self.norm_layers[i + n + 1](x))
			if n != len(self.up_layers) - 1:
				x = torch.cat([x, skip.pop()], dim=1)

		return self.out(x)


class CCNet(CNet):
	def __init__(
			self, 
			in_channels: int = 2, 
			channels: list = [64, 128, 256], 
			time_channels: int = 32,
			beta_channels: int = 32, 
			activation: Module = torch.nn.SiLU(), 
			dropout_rate: float = 0.2, 
			padding_mode: str = "circular", 
			device: Optional[str | torch.device] = None, 
			**kwargs
		):
		super().__init__(in_channels, channels, time_channels, activation, dropout_rate, padding_mode, device, **kwargs)
		self.beta_embed = Embedding(embed_dim=beta_channels, device=device)
		self.b_linears = ModuleList([
			torch.nn.Linear(beta_channels, c, device=device) for c in self.channels + self.channels_r[1:]
		])
	
	def _forward_impl(self, x: Tensor, t: Tensor, beta: Tensor) -> Tensor:
		skip = []
		t_emb = self.time_embed(t)
		b_emb = self.beta_embed(beta)

		for i, layer in enumerate(self.down_layers):
			x = layer(x)
			x += self.t_linears[i](t_emb)[..., None, None] 
			x += self.b_linears[i](b_emb)[..., None, None]
			x = self.dropout(x)
			x = self.act(self.norm_layers[i](x))
			if i != len(self.down_layers) - 1:
				skip.append(x)
		
		for n, layer in enumerate(self.up_layers):
			x = layer(x)
			x += self.t_linears[i + n + 1](t_emb)[..., None, None]
			x += self.b_linears[i + n + 1](b_emb)[..., None, None]
			x = self.dropout(x)
			x = self.act(self.norm_layers[i + n + 1](x))
			if n != len(self.up_layers) - 1:
				x = torch.cat([x, skip.pop()], dim=1)

		return self.out(x)