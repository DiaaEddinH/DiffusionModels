import torch

from torch import Tensor

from collections.abc import Sequence


def normalize_kernel_size(kernel_size: int | Sequence[int], ndim: int) -> tuple[int]:
    if isinstance(kernel_size, int):
        return (kernel_size,) * ndim
    if not isinstance(kernel_size, Sequence):
        raise TypeError("`kernel_size` must be int or sequence (tuple, list)")
    kernel_size = tuple(kernel_size)
    n = len(kernel_size)
    if n == 1:
        return kernel_size * ndim
    if n != ndim:
        raise ValueError(f"`kernel_size` must have length 1 or {ndim}, " f"got {n}")
    return kernel_size


def circular_padding(
    kernel_size: int | Sequence[int], ndim: int, dilation: int | Sequence[int] = 1
) -> tuple[int]:
    if ndim is None:
        if isinstance(kernel_size, int):
            raise ValueError("`ndim` is required when `kernel_size` is scalar")
        ndim = len(kernel_size)

    kernel_size = normalize_kernel_size(kernel_size, ndim)

    if isinstance(dilation, int):
        dilation = (dilation,) * ndim
    else:
        dilation = tuple(dilation)
        if len(dilation) != ndim:
            raise ValueError(f"`dilation` must have length {ndim}")

    return tuple(
        p
        for k, d in zip(reversed(kernel_size), reversed(dilation))
        for p in ((k - 1) // 2 * d, k // 2 * d)
    )


def circular_pad_nd(x: Tensor, pad: Sequence[int]) -> Tensor:
    assert len(pad) % 2 == 0
    ndim_padded = len(pad) // 2
    for i in range(ndim_padded):
        lo, hi = pad[2 * i], pad[2 * i + 1]
        if lo == 0 and hi == 0:
            continue
        dim = x.dim() - 1 - i
        d_size = x.size(dim)
        idx = torch.arange(-lo, d_size + hi, device=x.device) % d_size
        x = x.index_select(dim, idx)
    return x