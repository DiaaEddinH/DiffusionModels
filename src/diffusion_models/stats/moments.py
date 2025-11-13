import functools
import itertools

import numpy as np
from diffusion_models.stats.bootstrap import bootstrap_estimator
from numpy._typing import ArrayLike


def moment(data, order, axis=0, center=None):
    """Central moment of given order along an axis."""
    if center is None:
        center = np.mean(data, axis=axis, keepdims=True)
    return np.mean((data - center) ** order, axis=axis)


def _central_moments_vector(x: ArrayLike, max_order: int, axis: int = 0):
    """
    Return mu and central moments m_k for k=1..max_order in a numerically stable way.

    Key tricks:
    - Accumulate in float64/complex128 (or longdouble) regardless of input dtype.
    - Standardize: compute moments of z = (x - mu)/s (with s ~ std), then scale back by s**k.
      This keeps powers near O(1) so k=6..8 don't explode.
    """
    x = np.asarray(x)
    is_complex = np.iscomplexobj(x)

    # Choose accumulation dtype
    dtype = np.complex128 if is_complex else np.float64

    xp = x.astype(dtype, copy=False)

    mu = np.mean(xp, axis=axis)
    mu_kd = np.expand_dims(mu, axis=axis)

    # Centered data in high precision
    c = xp - mu_kd

    s = np.sqrt(
        np.mean((c * np.conjugate(c)).real if is_complex else (c * c), axis=axis)
    )
    s = np.where(s == 0, dtype(1.0), s)
    s_kd = np.expand_dims(s, axis=axis)

    # Standardized centered variable
    z = c / s_kd

    out_shape = (max_order + 1,) + np.shape(mu)
    out = np.zeros(out_shape, dtype=dtype)

    # m1 = 0 by definition of central moments; we keep index alignment (m0 unused)
    for k in range(1, max_order + 1):
        # Compute E[z**k] in stabilized space then rescale by s**k
        # For complex data, z**k is the usual complex power.
        mk_std = np.mean(z**k, axis=axis)
        out[k] = mk_std * (s**k)

    # For k=2 small negative noise can appear from rounding; clip to >= 0 for real data
    if not is_complex:
        out[2] = np.maximum(out[2], dtype(0.0))

    return mu.astype(x.dtype, copy=False), out.astype(x.dtype, copy=False)


def calc_moments(data, max_order=8, n_bins=100):
    """Bootstrap-estimated raw moments E[X^n] for n=1..max_order."""
    vals, errs = [], []
    for n in range(1, max_order + 1):
        obs = lambda d, n=n: np.mean(d**n, axis=0)
        val, err = bootstrap_estimator(data, obs, n_bins=n_bins)
        vals.append(val)
        errs.append(err)
    return np.array(vals), np.array(errs)


def other_moments(data, n, m):
    """
    Mixed moments for a symmetric 2D distribution.
    Accepts either:
      - real array of shape (N, 2) interpreted as [x, y]
      - complex array of shape (N,) interpreted as x + i y
    Returns E[x^n y^m] if n==m, else E[x^n y^m + x^m y^n].
    """
    d = np.asarray(data)
    if d.ndim == 1 and np.iscomplexobj(d):
        x = d.real
        y = d.imag
    else:
        x = d[:, 0]
        y = d[:, 1]
    if n == m:
        return np.mean((x**n) * (y**m))
    return np.mean(x**n * y**m + x**m * y**n)


def calc_other_moments(data, max_order=8, n_bins=100):
    """Bootstrap-estimated mixed moments with symmetry constraints."""
    pairs = [
        (n, m)
        for (n, m) in itertools.product(range(1, max_order + 1), repeat=2)
        if (n + m) % 2 == 0 and (n + m) <= max_order and m <= n
    ]

    vals, errs = [], []
    for n, m in pairs:
        func = functools.partial(other_moments, n=n, m=m)
        val, err = bootstrap_estimator(data, func, n_bins=n_bins)
        vals.append(val)
        errs.append(err)
    return np.array(vals), np.array(errs)
