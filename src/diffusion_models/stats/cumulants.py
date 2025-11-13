import numpy as np
from numpy._typing import ArrayLike

from diffusion_models.stats.bootstrap import bootstrap_estimator
from diffusion_models.stats.moments import _central_moments_vector


def _cumulant_from_central_moments(mu: ArrayLike, C: ArrayLike, n: int):
    """
    Cumulant of order n from central moments stored in C.
    """
    assert np.all(
        np.isfinite(np.asarray(mu))
    ), f"mu components must be finite, got {mu!r}"

    C = np.asarray(C)

    assert np.all(np.isfinite(C)), f"C must be finite, got {C!r}"
    assert 1 <= n <= 8, f"Order n must be in 1..8, got {n}"

    if n == 1:
        return mu
    if n == 2:
        return C[2]
    if n == 3:
        return C[3]
    if n == 4:
        m2, m4 = C[2], C[4]
        return m4 - 3.0 * m2 * m2
    if n == 5:
        m2, m3 = C[2], C[3]
        m5 = C[5] if len(C) > 5 else 0.0
        return m5 - 10.0 * m3 * m2
    if n == 6:
        m2, m4 = C[2], C[4]
        m6 = C[6] if len(C) > 6 else 0.0
        return m6 - 15.0 * m4 * m2 + 30.0 * (m2**3)
    if n == 7:
        m2, m3, m4 = C[2], C[3], C[4]
        m5 = C[5] if len(C) > 5 else 0.0
        m7 = C[7] if len(C) > 7 else 0.0
        return m7 - 21.0 * m5 * m2 - 35.0 * m4 * m3 + 210.0 * m3 * (m2**2)
    if n == 8:
        m2, m4 = C[2], C[4]
        m6 = C[6] if len(C) > 6 else 0.0
        m8 = C[8] if len(C) > 8 else 0.0
        return (
            m8
            - 28.0 * m6 * m2
            - 35.0 * (m4**2)
            + 420.0 * m4 * (m2**2)
            - 630.0 * (m2**4)
        )


def calc_cumulants(data: ArrayLike, max_order: int = 8, n_bins: int = 100):
    """Bootstrap-estimated cumulants up to max_order using a single identity source.

    If `data` is complex, we treat the observable as complex to keep bootstrap
    shapes consistent across orders. Otherwise, we keep real-valued outputs.
    :param data: Input data (N, ...) where the first dim indexes samples
    :param max_order: Maximum cumulant order to compute (1..8)
    :param n_bins: Number of bootstrap bins
    """
    data = np.asarray(data)
    force_complex = np.iscomplexobj(data)
    dtype = np.complex128 if force_complex else np.float64

    def obs(d, n):
        mu, C = _central_moments_vector(d.astype(dtype, copy=False), max_order, axis=0)
        k = _cumulant_from_central_moments(mu, C, n)
        if force_complex:
            return np.asarray(k, dtype=dtype)
        return k

    vals, errs = [], []
    for n in range(1, max_order + 1):
        val, err = bootstrap_estimator(data, lambda d, n=n: obs(d, n), n_bins=n_bins)
        vals.append(val)
        errs.append(err)
    return np.array(vals), np.array(errs)
