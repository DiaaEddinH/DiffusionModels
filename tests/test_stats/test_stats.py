import numpy as np
import pytest

from src.stats import (
    bootstrap_estimator,
    moment,
    calc_moments,
    calc_cumulants,
    other_moments,
    calc_other_moments,
)


def test_bootstrap_estimator_real_and_complex():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=1.5, scale=0.7, size=200)

    # Real-valued observable: mean
    np.random.seed(0)
    mean_est, err_est = bootstrap_estimator(data, lambda d: np.mean(d), n_bins=200)
    sample_mean = np.mean(data)
    assert np.isfinite(mean_est)
    assert np.isfinite(err_est)
    # Bootstrap mean should be close to sample mean
    assert np.allclose(mean_est, sample_mean, rtol=0, atol=0.05)
    # Error should be positive (non-zero for non-constant data)
    assert err_est > 0

    # Complex-valued observable path
    def complex_obs(d):
        m = np.mean(d)
        return m + 1j * (m - 1.0)

    np.random.seed(0)
    mean_c, err_c = bootstrap_estimator(data, complex_obs, n_bins=150)
    # Should return 1D vectors [real, imag]
    assert mean_c.shape == (2,)
    assert err_c.shape == (2,)
    # Check consistency of real/imag parts
    sample_c = complex_obs(data)
    assert np.allclose(mean_c[0], np.real(sample_c), atol=0.05)
    assert np.allclose(mean_c[1], np.imag(sample_c), atol=0.05)
    assert np.all(err_c >= 0)


def test_moment_central_and_axis_center():
    # 2D data to exercise axis and center handling
    x = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    # axis=1: per-row moments
    m1_row0 = moment(x[0], 1, axis=0)
    m1_row1 = moment(x[1], 1, axis=0)
    assert np.allclose(m1_row0, 0.0)
    assert np.allclose(m1_row1, 0.0)

    # Provide a custom center and axis over columns (axis=0)
    center = np.array([1.5, 3.0, 4.5])
    m2_cols = moment(x, 2, axis=0, center=center)
    # Manual computation
    manual = np.mean((x - center) ** 2, axis=0)
    assert np.allclose(m2_cols, manual)


def test_calc_moments_shapes_and_values():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(500)
    max_order = 5

    np.random.seed(123)
    vals, errs = calc_moments(data, max_order=max_order, n_bins=300)
    assert vals.shape == (max_order,)
    assert errs.shape == (max_order,)

    # Compare to sample moments of powers
    for n in range(1, max_order + 1):
        target = np.mean(data**n)
        assert np.allclose(vals[n - 1], target, atol=0.07)
        assert errs[n - 1] >= 0


def test_calc_cumulants_gaussian_properties():
    # For a zero-mean Gaussian, true cumulants beyond the 2nd are zero.
    # Use a statistical test: the bootstrap error bars should include zero.
    rng = np.random.default_rng(123)
    data = rng.normal(loc=0.0, scale=2.0, size=2000)

    np.random.seed(321)
    vals, errs = calc_cumulants(data, max_order=8, n_bins=400)
    assert vals.shape == (8,)
    assert errs.shape == (8,)

    # kappa_1 ~= 0 (central first moment)
    assert abs(vals[0]) < 0.05
    # kappa_2 ~= variance
    sample_var = np.mean((data - np.mean(data)) ** 2)
    assert np.isclose(vals[1], sample_var, atol=0.05)
    # For n >= 3, zero should lie within 3 sigma of the bootstrap estimate
    assert np.all(np.abs(vals[2:]) <= 3.0 * errs[2:])
    assert np.all(errs >= 0)


def test_other_moments_symmetry_and_equal_indices():
    # Simple 2D dataset
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # n == m branch
    val_equal = other_moments(data, 2, 2)
    manual_equal = np.mean((data[:, 0] ** 2) * (data[:, 1] ** 2))
    assert np.isclose(val_equal, manual_equal)

    # n != m symmetric branch (n,m) and (m,n) should match
    val_nm = other_moments(data, 1, 3)
    val_mn = other_moments(data, 3, 1)
    assert np.isclose(val_nm, val_mn)

    # Explicit manual check
    x, y = data.T
    manual_sym = np.mean(x**1 * y**3 + x**3 * y**1)
    assert np.isclose(val_nm, manual_sym)


def test_calc_other_moments_selection_and_values():
    rng = np.random.default_rng(7)
    # 2D data as required by other_moments
    data = rng.normal(size=(400, 2))

    max_order = 4
    np.random.seed(999)
    vals, errs = calc_other_moments(data, max_order=max_order, n_bins=250)

    # Determine expected (n,m) pairs: 1<=m<=n, n+m even, n+m<=max_order
    expected_pairs = []
    for n in range(1, max_order + 1):
        for m in range(1, n + 1):
            if (n + m) % 2 != 0:
                continue
            if (n + m) > max_order:
                continue
            expected_pairs.append((n, m))

    assert vals.shape == (len(expected_pairs),)
    assert errs.shape == (len(expected_pairs),)

    # Check values against direct computation of other_moments
    # We iterate the same order as in the function to align indices
    idx = 0
    for n in range(1, max_order + 1):
        for m in range(1, n + 1):
            if (n + m) % 2 != 0 or (n + m) > max_order:
                continue
            direct = other_moments(data, n, m)
            assert np.allclose(vals[idx], direct, atol=0.08)
            assert errs[idx] >= 0
            idx += 1


def test_bootstrap_estimator_vector_observable():
    rng = np.random.default_rng(123)
    data2 = rng.normal(size=(300, 2))

    # Observable returns a real 2-vector [mean_x, mean_y]
    def vec_obs(d):
        mu = np.mean(d, axis=0)  # shape (2,)
        return mu

    np.random.seed(0)
    mean_v, err_v = bootstrap_estimator(data2, vec_obs, n_bins=200)

    assert mean_v.shape == (2,)
    assert err_v.shape == (2,)
    # Close to sample means (loose, deterministic via seed)
    sample_mu = data2.mean(axis=0)
    assert np.allclose(mean_v, sample_mu, atol=0.05)
    assert np.all(err_v >= 0)
    assert np.isfinite(mean_v).all() and np.isfinite(err_v).all()
