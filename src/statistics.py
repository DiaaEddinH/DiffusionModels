from math import log, comb
import numpy as np
import torch

# -----------------------------
# Mixture of Gaussians constants
# -----------------------------
m_value = 1.0
SIGMA = 0.25
VAR = SIGMA * SIGMA
TWOPI = 2.0 * np.pi
LOG_NORM_2D = -0.5 * 2.0 * log(TWOPI * VAR)  # 2D isotropic Gaussian

MEAN1_X = m_value
MEAN1_Y = -m_value
MEAN2_X = -m_value
MEAN2_Y = m_value


# -----------------------------
# Bootstrap utilities
# -----------------------------
def bootstrap_estimator(data, observable, n_bins=100):
    """
    Generic bootstrap estimator.
    Args:
        data (ndarray): Input data (N, ...) where the first dim indexes samples
        observable (callable): Function to apply to bootstrap samples; returns scalar or vector
        n_bins (int): Number of bootstrap resamples
    Returns:
        mean, error (floats or arrays)
    """
    n_samples = len(data)
    bins = []
    for _ in range(n_bins):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        bins.append(observable(data[idx]))

    bins = np.asarray(bins)  # predictable shape handling
    if np.iscomplexobj(bins):
        bins = np.stack([bins.real, bins.imag], axis=-1)

    mean = np.mean(bins, axis=0)
    # Unbiased std across bootstrap replicas
    error = np.sqrt(np.sum((bins - mean) ** 2, axis=0) / max(1, (n_bins - 1)))
    return mean, error


def bootstrap(data, n_boot=100):
    """Bootstrap of the mean observable (compat wrapper)."""
    return bootstrap_estimator(data, lambda d: np.mean(d, axis=0), n_bins=n_boot)


# -----------------------------
# Moment and cumulant utilities
# -----------------------------
def moment(data, order, axis=0, center=None):
    """Central moment of given order along an axis."""
    if center is None:
        center = np.mean(data, axis=axis, keepdims=True)
    return np.mean((data - center) ** order, axis=axis)


def _central_moments_vector(x, max_order, axis=0):
    """Return central moments m_k for k=1..max_order as an array-like where idx=k gives m_k."""
    mu = np.mean(x, axis=axis)
    c = x - mu
    out = np.zeros(max_order + 1, dtype=float)
    for k in range(1, max_order + 1):
        out[k] = np.mean(c**k, axis=axis)
    return mu, out  # m1 (mean) returned separately for clarity


def _cumulant_from_central_moments(mu, C, n):
    """Single source of truth for cumulant identities (up to 8th) using central moments C."""
    # C[k] is the k-th central moment; mu is mean (C[1] == 0 by definition)
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
    raise ValueError("Cumulants supported only for orders 1..8")


def calc_moments(data, max_order=8, n_bins=100):
    """Bootstrap-estimated raw moments E[X^n] for n=1..max_order."""
    vals, errs = [], []
    for n in range(1, max_order + 1):
        obs = lambda d, n=n: np.mean(d**n, axis=0)
        val, err = bootstrap_estimator(data, obs, n_bins=n_bins)
        vals.append(val)
        errs.append(err)
    return np.array(vals), np.array(errs)


def calc_cumulants(data, max_order=8, n_bins=100):
    """Bootstrap-estimated cumulants up to max_order using a single identity source."""

    def obs(d, n):
        mu, C = _central_moments_vector(d, max_order, axis=0)
        return _cumulant_from_central_moments(mu, C, n)

    vals, errs = [], []
    for n in range(1, max_order + 1):
        val, err = bootstrap_estimator(data, lambda d, n=n: obs(d, n), n_bins=n_bins)
        vals.append(val)
        errs.append(err)
    return np.array(vals), np.array(errs)


# -----------------------------
# Mixed moments utilities
# -----------------------------
def other_moments(data, n, m):
    """
    Mixed moments for a symmetric 2D distribution.
    Assumes data shape (N, 2). Returns E[x^n y^m] if n==m,
    else E[x^n y^m + x^m y^n].
    """
    x = data[:, 0]
    y = data[:, 1]
    if n == m:
        return np.mean((x**n) * (y**m))
    return np.mean(x**n * y**m + x**m * y**n)


def calc_other_moments(data, max_order=8, n_bins=100):
    """Bootstrap-estimated mixed moments with symmetry constraints."""
    vals, errs = [], []
    for n in range(1, max_order + 1):
        for m in range(1, n + 1):
            if (n + m) % 2 != 0:
                continue
            if (n + m) > max_order:
                continue
            val, err = bootstrap_estimator(
                data, lambda d, n=n, m=m: other_moments(d, n, m), n_bins=n_bins
            )
            vals.append(val)
            errs.append(err)
    return np.array(vals), np.array(errs)


# -----------------------------
# Log-density and MCMC samplers
# -----------------------------
def logpdf_component_batch(x0, x1, mean_x, mean_y):
    dx0 = x0 - mean_x
    dx1 = x1 - mean_y
    quad = (dx0 * dx0 + dx1 * dx1) / VAR
    return LOG_NORM_2D - 0.5 * quad


def logpdf_mixture_batch(x0, x1):
    l1 = logpdf_component_batch(x0, x1, MEAN1_X, MEAN1_Y)
    l2 = logpdf_component_batch(x0, x1, MEAN2_X, MEAN2_Y)
    m = np.maximum(l1, l2)
    return np.log(0.5 * np.exp(l1 - m) + 0.5 * np.exp(l2 - m)) + m


@torch.no_grad()
def logpdf_batch(x0, x1, model):
    torchX = torch.stack([x0, x1], dim=-1)
    t0 = torch.tensor([1e-2], device=model.device)
    return model.energy(torchX, t0)


def mh_parallel(
    n_chains, n_keep, burn_in, thin, seed, init_prop_std, adapt, adapt_window
):
    rng = np.random.default_rng(seed)

    # Initialize chains at different modes to help mixing
    x0 = np.empty(n_chains)
    x1 = np.empty(n_chains)
    half = n_chains // 2
    x0[:half], x1[:half] = MEAN1_X, MEAN1_Y
    x0[half:], x1[half:] = MEAN2_X, MEAN2_Y

    prop_std = np.full(n_chains, float(init_prop_std))
    cur_lp = logpdf_mixture_batch(x0, x1)

    # Burn-in (with simple adaptation every adapt_window steps)
    acc_window = np.zeros(n_chains)
    steps_in_window = 0

    t = 0
    while t < burn_in:
        prop0 = x0 + rng.normal(0.0, prop_std, size=n_chains)
        prop1 = x1 + rng.normal(0.0, prop_std, size=n_chains)
        prop_lp = logpdf_mixture_batch(prop0, prop1)

        u = rng.uniform(size=n_chains)
        accept = np.log(u) < (prop_lp - cur_lp)

        x0 = np.where(accept, prop0, x0)
        x1 = np.where(accept, prop1, x1)
        cur_lp = np.where(accept, prop_lp, cur_lp)

        acc_window += accept.astype(float)
        steps_in_window += 1
        t += 1

        if adapt and steps_in_window >= adapt_window:
            rate = acc_window / float(steps_in_window)
            prop_std = np.where(rate < 0.2, prop_std * 0.9, prop_std)
            prop_std = np.where(rate > 0.5, prop_std * 1.1, prop_std)
            prop_std = np.clip(prop_std, 0.15, 2.5)
            acc_window[:] = 0.0
            steps_in_window = 0

    # Collect kept samples with thinning
    xs_all = np.empty(n_keep * n_chains)
    ys_all = np.empty(n_keep * n_chains)
    base = 0
    accepts = 0.0
    proposals = 0.0

    kept = 0
    while kept < n_keep:
        for _ in range(thin):
            prop0 = x0 + rng.normal(0.0, prop_std, size=n_chains)
            prop1 = x1 + rng.normal(0.0, prop_std, size=n_chains)
            prop_lp = logpdf_mixture_batch(prop0, prop1)

            u = rng.uniform(size=n_chains)
            accept = np.log(u) < (prop_lp - cur_lp)

            x0 = np.where(accept, prop0, x0)
            x1 = np.where(accept, prop1, x1)
            cur_lp = np.where(accept, prop_lp, cur_lp)

            accepts += float(np.sum(accept))
            proposals += float(n_chains)

        xs_all[base : base + n_chains] = x0
        ys_all[base : base + n_chains] = x1
        base += n_chains
        kept += 1

    acc_rate = accepts / proposals if proposals > 0.0 else 0.0
    return np.stack([xs_all, ys_all], axis=-1), acc_rate


def torch_mh_parallel(
    n_chains, n_keep, burn_in, thin, seed, init_prop_std, adapt, adapt_window, model
):
    torch.manual_seed(seed)
    device = model.device

    # --- Init state ---
    x0 = torch.empty(n_chains, device=device)
    x1 = torch.empty(n_chains, device=device)
    half = n_chains // 2
    x0[:half], x1[:half] = MEAN1_X, MEAN1_Y
    x0[half:], x1[half:] = MEAN2_X, MEAN2_Y

    prop_std = torch.full((n_chains,), float(init_prop_std), device=device)
    cur_lp = logpdf_batch(x0, x1, model=model)

    acc_window = torch.zeros(n_chains, device=device)
    steps_in_window = 0

    # --- Burn-in ---
    t = 0
    while t < burn_in:
        prop0 = x0 + torch.randn(n_chains, device=device) * prop_std
        prop1 = x1 + torch.randn(n_chains, device=device) * prop_std
        prop_lp = logpdf_batch(prop0, prop1, model=model)

        u = torch.rand(n_chains, device=device)
        accept = torch.log(u) < (prop_lp - cur_lp)

        x0 = torch.where(accept, prop0, x0)
        x1 = torch.where(accept, prop1, x1)
        cur_lp = torch.where(accept, prop_lp, cur_lp)

        acc_window += accept.float()
        steps_in_window += 1
        t += 1

        if adapt and steps_in_window >= adapt_window:
            rate = acc_window / float(steps_in_window)
            prop_std = torch.where(rate < 0.2, prop_std * 0.9, prop_std)
            prop_std = torch.where(rate > 0.5, prop_std * 1.1, prop_std)
            prop_std = torch.clamp(prop_std, 0.15, 2.5)
            acc_window.zero_()
            steps_in_window = 0

    # --- Sampling ---
    xs_all = torch.empty(n_keep * n_chains, dtype=torch.float32, device="cpu")
    ys_all = torch.empty(n_keep * n_chains, dtype=torch.float32, device="cpu")

    base = 0
    accepts = 0.0
    proposals = 0.0
    kept = 0

    while kept < n_keep:
        for _ in range(thin):
            prop0 = x0 + torch.randn(n_chains, device=device) * prop_std
            prop1 = x1 + torch.randn(n_chains, device=device) * prop_std
            prop_lp = logpdf_batch(prop0, prop1, model=model)

            u = torch.rand(n_chains, device=device)
            accept = torch.log(u) < (prop_lp - cur_lp)

            x0 = torch.where(accept, prop0, x0)
            x1 = torch.where(accept, prop1, x1)
            cur_lp = torch.where(accept, prop_lp, cur_lp)

            accepts += float(accept.sum().item())
            proposals += float(n_chains)

        xs_all[base : base + n_chains] = x0.detach().cpu()
        ys_all[base : base + n_chains] = x1.detach().cpu()
        base += n_chains
        kept += 1

    acc_rate = accepts / proposals if proposals > 0.0 else 0.0
    return torch.stack([xs_all, ys_all], dim=-1), acc_rate


# -----------------------------
# Analytic and helper math
# -----------------------------
def raw_moments(v, max_k):
    out = np.zeros(max_k + 1)
    for k in range(1, max_k + 1):
        out[k] = float(np.mean(v**k))
    return out


def central_moments(v, max_k):
    mu = float(np.mean(v))
    c = v - mu
    out = np.zeros(max_k + 1)
    for k in range(1, max_k + 1):
        out[k] = float(np.mean(c**k))
    return [mu, out]


def cumulants_from_central(mu, C, max_k):
    """
    Delegate to the single identity source to avoid duplication.
    Returns K[0..max_k], where entries 1..max_k are filled.
    """
    K = np.zeros(max_k + 1)
    for n in range(1, max_k + 1):
        K[n] = _cumulant_from_central_moments(mu, C, n)
    return K


def double_factorial_odd(n):
    if n < 1:
        return 1.0
    r = 1.0
    k = 1
    while k <= n:
        r *= float(k)
        k += 2
    return r


def gaussian_even_moment(order_even, sigma):
    if (order_even % 2) == 1:
        return 0.0
    if order_even == 0:
        return 1.0
    return double_factorial_odd(order_even - 1) * (sigma**order_even)


def analytic_raw_marginal(max_k, m_abs, sigma):
    R = np.zeros(max_k + 1)
    for k in range(1, max_k + 1):
        if (k % 2) == 1:
            R[k] = 0.0
        else:
            acc = 0.0
            for i in range(0, k + 1, 2):  # only even i contribute
                acc += comb(k, i) * (m_abs**i) * gaussian_even_moment(k - i, sigma)
            R[k] = acc
    return R


def analytic_cumulants(max_k, m_abs, sigma):
    K = np.zeros(max_k + 1)
    if max_k >= 1:
        K[1] = 0.0
    if max_k >= 2:
        K[2] = sigma * sigma + m_abs * m_abs
    if max_k >= 3:
        K[3] = 0.0
    if max_k >= 4:
        K[4] = -2.0 * (m_abs**4)
    if max_k >= 5:
        K[5] = 0.0
    if max_k >= 6:
        K[6] = 16.0 * (m_abs**6)
    if max_k >= 7:
        K[7] = 0.0
    if max_k >= 8:
        K[8] = -272.0 * (m_abs**8)
    return K


# Convenience functions retained for completeness (not used in tests)
def print_table(label, raw_s, raw_t, cum_s, cum_t, max_k):
    print(label + " (moments & cumulants, orders 1..%d):" % max_k)
    header = "order | raw_sample         | raw_analytic        | cumulant_sample     | cumulant_analytic"
    print(header)
    print("-" * len(header))
    for k in range(1, max_k + 1):
        rs = raw_s[k] if k < len(raw_s) else float("nan")
        rt = raw_t[k] if k < len(raw_t) else float("nan")
        cs = cum_s[k] if k < len(cum_s) else float("nan")
        ct = cum_t[k] if k < len(cum_t) else float("nan")
        line = "{:>5d} | {:>18.10f} | {:>18.10f} | {:>18.10f} | {:>18.10f}"
        print(line.format(k, rs, rt, cs, ct))
    print("")


def run_mcmc_ebm(model):
    n_chains = 10000
    n_keep_per_chain = 100  # kept samples per chain
    burn_in = 5000
    thin = 100  # proposals between kept samples
    seed = 123
    init_prop_std = 0.15
    adapt = True
    adapt_window = 500  # adaptation interval during burn-in

    cfgs_all, acc_rate = torch_mh_parallel(
        n_chains=n_chains,
        n_keep=n_keep_per_chain,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
        init_prop_std=init_prop_std,
        adapt=adapt,
        adapt_window=adapt_window,
        model=model,
    )
    print(f"Acc. rate: {acc_rate:.4f}")
    return cfgs_all
