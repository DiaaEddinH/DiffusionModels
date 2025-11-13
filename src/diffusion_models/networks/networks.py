import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from typing import Optional

from torch.nn.utils.parametrizations import weight_norm


class GaussianFourierProjection(Module):
    """Random Fourier feature mapping"""

    def __init__(
        self, embed_dim: int, scale: float = 30.0, device: Optional[str] = None
    ) -> None:
        super().__init__()
        self.register_buffer("W", torch.randn(embed_dim // 2, device=device) * scale)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[..., None] * self.W[None, ...] * 2 * torch.pi
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)


class Embedding(Module):
    def __init__(
        self,
        embed_dim: int,
        activation: Module = torch.nn.SiLU(),
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.proj = GaussianFourierProjection(embed_dim, device=device)
        self.linear = torch.nn.Linear(embed_dim, embed_dim, device=device)
        self.act = activation

    def forward(self, x: Tensor):
        x = self.proj(x)
        x = self.linear(x)
        x = self.act(x)
        return x


class LinearNet(Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] = [32, 32],
        time_channels: int = 32,
        activation: Module = torch.nn.LeakyReLU(),
        dropout_rate: float = 0.2,
        device: str | torch.device = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.time_embed = Embedding(embed_dim=time_channels, device=device)
        # Network architecture layers
        self.channels = [in_channels] + channels

        self.layers = ModuleList(
            [
                torch.nn.Linear(self.channels[i], self.channels[i + 1], device=device)
                for i in range(len(self.channels) - 1)
            ]
        )
        self.t_linears = ModuleList(
            [torch.nn.Linear(time_channels, c, device=device) for c in channels]
        )
        self.final = torch.nn.Linear(channels[-1], in_channels, device=device)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.act = activation
        # Model's parameter

    def forward(self, *inputs: tuple):
        return self._forward_impl(*inputs)

    def _forward_impl(self, x: Tensor, t: Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        for i, layer in enumerate(self.layers):
            x = layer(x) + self.t_linears[i](t_emb)
            x = self.act(x)
            # x = self.dropout(x)

        return self.final(x)


class EvenLinear(LinearNet):
    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] = [32, 32],
        time_channels: int = 32,
        activation: Module = torch.nn.LeakyReLU(),
        dropout_rate: float = 0.0,
        device: str | torch.device = None,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels,
            channels,
            time_channels,
            activation,
            dropout_rate,
            device,
            **kwargs
        )

    def _forward_impl(self, x, t):
        t_emb = self.time_embed(t)
        x_neg = -x

        for i, layer in enumerate(self.layers):
            time_layer = self.t_linears[i](t_emb)
            x = layer(x) + time_layer
            x_neg = layer(x_neg) + time_layer
            x = self.act(x)
            x_neg = self.act(x_neg)
        return self.final(0.5 * (x + x_neg))


class OddLinear(LinearNet):
    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] = [32, 32],
        time_channels: int = 32,
        activation: Module = torch.nn.LeakyReLU(),
        dropout_rate: float = 0.0,
        device: str | torch.device = None,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels,
            channels,
            time_channels,
            activation,
            dropout_rate,
            device,
            **kwargs
        )
        self.final = torch.nn.Linear(
            channels[-1], in_channels, bias=False, device=device
        )

    def _forward_impl(self, x, t):
        t_emb = self.time_embed(t)
        x_neg = -x

        for i, layer in enumerate(self.layers):
            time_layer = self.t_linears[i](t_emb)
            x = layer(x) + time_layer
            x_neg = layer(x_neg) + time_layer
            x = self.act(x)
            x_neg = self.act(x_neg)
        return self.final(0.5 * (x - x_neg))


class ConditionalLinearWrapper(Module):
    def __init__(
        self, network: Module, label_dim: int = 32, device=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.network = network
        self.label_embed = Embedding(embed_dim=label_dim, device=device)

        self.proj_layers = ModuleList(
            [
                torch.nn.Linear(label_dim, c, device=device)
                for c in self.network.channels[1:]
            ]
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> torch.Tensor:
        t_emb = self.network.time_embed(t)
        y_emb = self.label_embed(y)

        for i, layer in enumerate(self.network.layers):
            x = layer(x) + self.network.t_linears[i](t_emb) + self.proj_layers[i](y_emb)
            x = self.network.act(x)

        return self.network.final(x)


class ShiftWrapper(Module):
    def __init__(self, network: Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network

    def forward(self, x: Tensor, t: Tensor, y: Tensor) -> torch.Tensor:
        shift_vec = torch.stack([y, -y], dim=-1)
        x_shifted = x - shift_vec

        return self.network(x_shifted, t, y)


# Instead we just apply WS to conv layers for stability
def ws_conv(*args, **kwargs):
    return weight_norm(torch.nn.Conv2d(*args, **kwargs))


def ws_convT(*args, **kwargs):
    return weight_norm(torch.nn.ConvTranspose2d(*args, **kwargs))


class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] = [64, 128, 256],
        time_channels: int = 32,
        activation: Module = torch.nn.SiLU(),
        padding_mode: str | torch.device = "circular",
        device=None,
        **kwargs
    ) -> None:
        super().__init__()
        self.time_embed = Embedding(embed_dim=time_channels, device=device)

        self.channels = channels
        self.channels_r = channels[::-1]

        self.t_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(time_channels, c, device=device)
                for c in self.channels + self.channels_r[1:]
            ]
        )

        self.down_layers = torch.nn.ModuleList(
            [
                ws_conv(
                    in_channels,
                    self.channels[0],
                    kernel_size=3,
                    dilation=2,
                    bias=False,
                    padding=1,
                    padding_mode=padding_mode,
                    device=device,
                )
            ]
            + [
                ws_conv(
                    c_in,
                    c_out,
                    kernel_size=3,
                    dilation=2,
                    bias=False,
                    padding=1,
                    padding_mode=padding_mode,
                    device=device,
                )
                for c_in, c_out in zip(self.channels, self.channels[1:])
            ]
        )

        self.up_layers = torch.nn.ModuleList(
            [
                ws_convT(
                    self.channels[-1],
                    self.channels[-2],
                    kernel_size=3,
                    dilation=1,
                    bias=False,
                    output_padding=0,
                    device=device,
                )
            ]
            + [
                ws_convT(
                    2 * c_in,
                    c_out,
                    kernel_size=3,
                    dilation=1,
                    bias=False,
                    output_padding=0,
                    device=device,
                )
                for c_in, c_out in zip(self.channels_r[1:], self.channels_r[2:])
            ]
        )

        self.act = activation
        self.final = ws_convT(
            2 * channels[0], in_channels, kernel_size=3, device=device
        )

    def forward(self, *inputs: tuple):
        return self._forward_impl(*inputs)

    def _forward_impl(self, x, t):
        skip = []
        t_emb = self.time_embed(t)

        for i, layer in enumerate(self.down_layers):
            x = layer(x) + self.t_linears[i](t_emb)[..., None, None]
            x = self.act(x)
            if i != len(self.down_layers) - 1:
                skip.append(x)

        for n, layer in enumerate(self.up_layers):
            x = layer(x) + self.t_linears[i + n + 1](t_emb)[..., None, None]
            x = self.act(x)
            x = torch.cat([x, skip.pop()], dim=1)

        return self.final(x)


class LinearAttention(Module):
    def __init__(self, channels, heads=4, dim_head=32, device: str = None):
        super().__init__()
        hidden_dim = heads * dim_head
        self.heads = heads
        self.to_q = torch.nn.Conv2d(channels, hidden_dim, 1, bias=False, device=device)
        self.to_k = torch.nn.Conv2d(channels, hidden_dim, 1, bias=False, device=device)
        self.to_v = torch.nn.Conv2d(channels, hidden_dim, 1, bias=False, device=device)
        self.to_out = torch.nn.Conv2d(hidden_dim, channels, 1, device=device)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.to_q(x)  # (b, hidden, h, w)
        k = self.to_k(x)
        v = self.to_v(x)

        # reshape for heads
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), (q, k, v))
        q = q.softmax(dim=-2)  # normalize across channel dim
        k = k.softmax(dim=-1)  # normalize across spatial dim

        # linear attention trick
        context = torch.einsum("bhdn,bhen->bhde", k, v)  # sum over spatial dim
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = out.reshape(b, -1, h, w)

        return self.to_out(out)


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels, num_heads=4, reduction=4, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.reduction = reduction
        self.norm = torch.nn.GroupNorm(1, channels, device=device)

        self.to_q = torch.nn.Conv1d(channels, channels, 1, device=device)
        self.to_k = torch.nn.Conv1d(channels, channels, 1, device=device)
        self.to_v = torch.nn.Conv1d(channels, channels, 1, device=device)
        self.proj = torch.nn.Conv1d(channels, channels, 1, device=device)

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        # flatten queries
        q = self.to_q(x.view(b, c, h * w))

        # downsample keys/values for efficiency
        k = torch.nn.functional.avg_pool2d(x, self.reduction)
        v = k
        hk, wk = k.shape[2:]
        k = self.to_k(k.view(b, c, hk * wk))
        v = self.to_v(v.view(b, c, hk * wk))

        # multi-head split
        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, hk * wk)
        v = v.view(b, self.num_heads, head_dim, hk * wk)

        # attention
        attn = torch.einsum("bnch,bnck->bnhk", q, k) * (head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bnhk,bnck->bnch", attn, v)

        out = out.reshape(b, c, h * w)
        out = self.proj(out).view(b, c, h, w)

        return x_in + out


class UNetWAttention(UNet):
    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] = [64, 128, 256],
        time_channels: int = 32,
        activation: Module = torch.nn.SiLU(),
        padding_mode: str | torch.device = "circular",
        device=None,
        **kwargs
    ) -> None:
        super().__init__(
            in_channels,
            channels,
            time_channels,
            activation,
            padding_mode,
            device,
            **kwargs
        )
        self.attention = AttentionBlock(self.channels[-1], num_heads=4, device=device)

    def _forward_impl(self, x, t):
        skip = []
        t_emb = self.time_embed(t)

        for i, layer in enumerate(self.down_layers):
            x = layer(x) + self.t_linears[i](t_emb)[..., None, None]
            x = self.act(x)
            if i != len(self.down_layers) - 1:
                skip.append(x)

        x = self.attention(x)

        for n, layer in enumerate(self.up_layers):
            x = layer(x) + self.t_linears[i + n + 1](t_emb)[..., None, None]
            x = self.act(x)
            x = torch.cat([x, skip.pop()], dim=1)

        return self.final(x)


# class UNetWAttention(torch.nn.Module):
# 	def __init__(
# 			self,
# 			in_channels: int = 2,
# 			channels: list[int] = [64, 128, 256],
# 			time_channels: int = 32,
# 			activation: Module = torch.nn.SiLU(),
# 			dropout_rate: float = 0.2,
# 			padding_mode: str | torch.device = "circular",
# 			device= None,
# 			**kwargs
# 		) -> None:
# 		super().__init__()
# 		self.time_embed = Embedding(embed_dim=time_channels, device=device)

# 		self.channels = channels
# 		self.channels_r = channels[::-1]

# 		self.t_linears = torch.nn.ModuleList([
# 			torch.nn.Linear(time_channels, c, device=device) for c in self.channels + self.channels_r[1:]
# 		])

# 		self.down_layers = torch.nn.ModuleList([
# 			ws_conv(in_channels, self.channels[0], kernel_size=3, dilation=2, bias=False, padding=1, padding_mode=padding_mode, device=device)
# 		] + [
# 			ws_conv(c_in, c_out, kernel_size=3, dilation=2, bias=False, padding=1, padding_mode=padding_mode, device=device)
# 			for c_in, c_out in zip(self.channels, self.channels[1:])
# 		])

# 		self.up_layers = torch.nn.ModuleList([
# 			ws_convT(self.channels[-1], self.channels[-2], kernel_size=3, dilation=1, bias=False, output_padding=0, device=device)
# 		] + [
# 			ws_convT(2 * c_in, c_out, kernel_size=3, dilation=1, bias=False, output_padding=0, device=device)
# 			for c_in, c_out in zip(self.channels_r[1:], self.channels_r[2:])
# 		])

# 		self.act = activation
# 		self.dropout = torch.nn.Dropout(dropout_rate)
# 		self.attention = AttentionBlock(self.channels[-1], num_heads=4, device=device)
# 		self.final = ws_convT(2*channels[0], 1, kernel_size=3, device=device)


# 	def forward(self, *inputs: tuple):
# 		return self._forward_impl(*inputs)

# 	def _forward_impl(self, x, t):
# 		skip = []
# 		t_emb = self.time_embed(t)

# 		for i, layer in enumerate(self.down_layers):
# 			x = layer(x) + self.t_linears[i](t_emb)[..., None, None];
# 			x = self.act(x)
# 			if i != len(self.down_layers) - 1:
# 				skip.append(x)

# 		x = self.attention(x)

# 		for n, layer in enumerate(self.up_layers):
# 			x = layer(x) + self.t_linears[i + n + 1](t_emb)[..., None, None]
# 			x = self.act(x)
# 			x = torch.cat([x, skip.pop()], dim=1)

# 		return self.final(x)
