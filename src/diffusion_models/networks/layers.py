import math
import torch

from torch import Tensor

from collections.abc import Sequence
from torch.nn import Module, Conv2d, Conv3d, ModuleList, Parameter

from .utils import normalize_kernel_size, circular_padding, circular_pad_nd


class TimeModulation(Module):
    """Multiply features by a real, time-dependent scale per channel."""

    def __init__(
        self, embed_dim: int, channels: int, bias: bool = False, device: str | torch.device = None
    ) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(embed_dim, channels * (2 if bias else 1), device=device)
        self._spatial_dims = None
        self.bias = bias

    def forward(self, input: Tensor, t: Tensor) -> Tensor:
        if self._spatial_dims is None:
            self._spatial_dims = (input.dim() - 2) * (None,)
        
        modulation = self.proj(t)

        if self.bias:
            gamma, beta = modulation.chunk(2, dim=-1)
            gamma = gamma[..., *self._spatial_dims]
            beta = beta[..., *self._spatial_dims]
            return input * (1 + gamma) + beta

        return input * modulation[..., *self._spatial_dims]



class CConv2d(Conv2d):
    """Wrapper for `Conv2d` that handles circular padding properly."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        bias: bool = False,
        dilation: int = 1,
        device: str | torch.device = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
            device=device,
        )

        self._pad = circular_padding(
            kernel_size=self.kernel_size,
            ndim=len(self.kernel_size),
            dilation=self.dilation,
        )
        self._do_pad = any(self._pad)

    def forward(self, input: Tensor) -> Tensor:
        if self._do_pad:
            input = torch.nn.functional.pad(
                input=input, pad=self._pad, mode="circular"
            )
        return super().forward(input)


class CConv3d(Conv3d):
    """Wrapper for `Conv3d` that handles circular padding properly."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        bias: bool = False,
        dilation: int = 1,
        device: str | torch.device = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
            device=device,
        )

        self._pad = circular_padding(
            kernel_size=self.kernel_size,
            ndim=len(self.kernel_size),
            dilation=self.dilation,
        )
        self._do_pad = any(self._pad)

    def forward(self, input: Tensor) -> Tensor:
        if self._do_pad:
            input = torch.nn.functional.pad(
                input=input, pad=self._pad, mode="circular"
            )
        return super().forward(input)

class CConv4d(Module):
    """
    4D convolution: input (B, C, D1, D2, D3, D4)
    Implemented as a sum of Conv3d along the 4th axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        ndim = 4
        k = normalize_kernel_size(kernel_size, ndim)
        s = normalize_kernel_size(stride, ndim)
        d = normalize_kernel_size(dilation, ndim)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.stride = s
        self.dilation = d

        self._circular_pad = circular_padding(k, ndim, d)
        p = (0, 0, 0, 0)
        self.padding = p
        self.k4, self.s4, self.d4 = k[3], s[3], d[3]

        self.conv3d_layers = ModuleList(
            [
                Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=k[:3],
                    stride=s[:3],
                    padding=p[:3],
                    dilation=d[:3],
                    bias=False,
                    device=device,
                )
                for _ in range(self.k4)
            ]
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels, device=device))
            fan_in = in_channels * math.prod(k)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        B, C, D1, D2, D3, D4 = x.shape
        assert C == self.in_channels, f"got {C} channels, expected {self.in_channels}."

        k4, s4, d4 = self.k4, self.s4, self.d4

        x = circular_pad_nd(x, self._circular_pad)
        D4 += self._circular_pad[0] + self._circular_pad[1]
        D3 += self._circular_pad[2] + self._circular_pad[3]
        D2 += self._circular_pad[4] + self._circular_pad[5]
        D1 += self._circular_pad[6] + self._circular_pad[7]
        p4 = 0

        D4_out = (D4 - d4 * (k4 - 1) - 1) // s4 + 1
        if D4_out <= 0:
            raise ValueError(
                f"Invalid output size along the 4th dimension, got {D4_out}."
            )

        out = None
        for i, conv3d in enumerate(self.conv3d_layers):
            start = i * d4
            end = start + (D4_out - 1) * s4 + 1
            x_slice = x[..., start:end:s4]
            x_slice = x_slice.permute(0, 5, 1, 2, 3, 4).reshape(
                B * D4_out, C, D1, D2, D3
            )

            y = conv3d(x_slice)
            Co, D1o, D2o, D3o = y.shape[1:]
            y = y.view(B, D4_out, Co, D1o, D2o, D3o)
            y = y.permute(0, 2, 3, 4, 5, 1)

            out = y if out is None else out + y

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out
