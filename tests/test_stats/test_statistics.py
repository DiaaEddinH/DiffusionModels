import unittest
import numpy as np
import torch
import pytest
from math import comb

from diffusion_models.stats.statistics import (
    bootstrap_estimator,
    moment,
    calc_moments,
    calc_cumulants,
    other_moments,
    calc_other_moments,
)
from diffusion_models.stats import statistics as ms

BOOT_N = 80


class DummyEnergyModel:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def energy(self, x, t):
        dx1 = x[:, 0] - ms.MEAN1_X
        dy1 = x[:, 1] - ms.MEAN1_Y
        dx2 = x[:, 0] - ms.MEAN2_X
        dy2 = x[:, 1] - ms.MEAN2_Y
        quad1 = (dx1 * dx1 + dy1 * dy1) / ms.VAR
        quad2 = (dx2 * dx2 + dy2 * dy2) / ms.VAR
        l1 = ms.LOG_NORM_2D - 0.5 * quad1
        l2 = ms.LOG_NORM_2D - 0.5 * quad2
        m = torch.maximum(l1, l2)
        return torch.log(0.5 * torch.exp(l1 - m) + 0.5 * torch.exp(l2 - m)) + m


@pytest.mark.usefixtures("seed_rng")
class TestStatistics(unittest.TestCase):
    def test_bootstrap_estimator_real_and_complex(self):
        data = np.random.normal(loc=1.5, scale=0.7, size=200)

        # Real-valued observable: mean
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

        mean_c, err_c = bootstrap_estimator(data, complex_obs, n_bins=150)
        # Should return 1D vectors [real, imag]
        assert mean_c.shape == (2,)
        assert err_c.shape == (2,)
        # Check consistency of real/imag parts
        sample_c = complex_obs(data)
        assert np.allclose(mean_c[0], np.real(sample_c), atol=0.05)
        assert np.allclose(mean_c[1], np.imag(sample_c), atol=0.05)
        assert np.all(err_c >= 0)

    def test_moment_broadcasting_and_axes(self):
        # axis=0 (per-column)
        x = np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
        )
        expected_axis0 = np.mean((x - x.mean(axis=0)) ** 2, axis=0)
        out_axis0 = moment(x, order=2, axis=0)
        np.testing.assert_allclose(out_axis0, expected_axis0)

        # axis=1 (per-row) with per-row center broadcasting
        x2 = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0]], dtype=float)
        center = x2.mean(axis=1, keepdims=True)  # shape (2,1)
        expected_axis1 = np.mean((x2 - center) ** 2, axis=1)
        out_axis1 = moment(x2, order=2, axis=1, center=center)
        np.testing.assert_allclose(out_axis1, expected_axis1)

    def test_moment_custom_center_scalar_and_vector(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])

        # Default center == mean
        exp2_default = np.mean((v - v.mean()) ** 2)
        exp3_default = np.mean((v - v.mean()) ** 3)
        np.testing.assert_allclose(moment(v, order=2), exp2_default)
        np.testing.assert_allclose(moment(v, order=3), exp3_default)

        # Explicit scalar center
        c = 2.0
        exp2_c = np.mean((v - c) ** 2)
        exp3_c = np.mean((v - c) ** 3)
        np.testing.assert_allclose(moment(v, order=2, center=c), exp2_c)
        np.testing.assert_allclose(moment(v, order=3, center=c), exp3_c)

    def test_moment_edge_cases(self):
        # order=0 returns ones along reduced axis
        x = np.random.RandomState(0).randn(5, 7)
        np.testing.assert_allclose(moment(x, order=0, axis=0), np.ones(x.shape[1]))
        np.testing.assert_allclose(moment(x, order=0, axis=1), np.ones(x.shape[0]))

        # all equal -> zero for order >= 1
        y = np.full((4, 3), 5.0)
        np.testing.assert_allclose(moment(y, order=1, axis=0), 0.0)
        np.testing.assert_allclose(moment(y, order=2, axis=1), 0.0)

    def test_calc_moments_shapes_and_values(self):
        data = np.random.standard_normal(500)
        max_order = 5

        vals, errs = calc_moments(data, max_order=max_order, n_bins=300)
        assert vals.shape == (max_order,)
        assert errs.shape == (max_order,)

        # Compare to sample moments of powers
        for n in range(1, max_order + 1):
            target = np.mean(data**n)
            assert np.allclose(vals[n - 1], target, atol=0.07)
            assert errs[n - 1] >= 0

    def test_calc_cumulants_gaussian_properties(self):
        # For a zero-mean Gaussian, true cumulants beyond the 2nd are zero.
        # Use a statistical test: the bootstrap error bars should include zero.
        data = np.random.normal(loc=0.0, scale=2.0, size=2000)

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

    def test_other_moments_symmetry_and_equal_indices(self):
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

    def test_calc_other_moments_selection_and_values(self):
        # 2D data as required by other_moments
        data = np.random.normal(size=(400, 2))

        max_order = 4
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

    def test_bootstrap_estimator_vector_observable(self):
        data2 = np.random.normal(size=(300, 2))

        # Observable returns a real 2-vector [mean_x, mean_y]
        def vec_obs(d):
            mu = np.mean(d, axis=0)  # shape (2,)
            return mu

        mean_v, err_v = bootstrap_estimator(data2, vec_obs, n_bins=200)

        assert mean_v.shape == (2,)
        assert err_v.shape == (2,)
        # Close to sample means (loose, deterministic via seed)
        sample_mu = data2.mean(axis=0)
        assert np.allclose(mean_v, sample_mu, atol=0.05)
        assert np.all(err_v >= 0)
        assert np.isfinite(mean_v).all() and np.isfinite(err_v).all()

    def _ref_logpdf_mixture_np(self, x0, x1):
        dx1 = x0 - ms.MEAN1_X
        dy1 = x1 - ms.MEAN1_Y
        dx2 = x0 - ms.MEAN2_X
        dy2 = x1 - ms.MEAN2_Y
        quad1 = (dx1 * dx1 + dy1 * dy1) / ms.VAR
        quad2 = (dx2 * dx2 + dy2 * dy2) / ms.VAR
        l1 = ms.LOG_NORM_2D - 0.5 * quad1
        l2 = ms.LOG_NORM_2D - 0.5 * quad2
        m = np.maximum(l1, l2)
        return np.log(0.5 * np.exp(l1 - m) + 0.5 * np.exp(l2 - m)) + m

    def test_logpdf_component_at_mean(self):
        lp = ms.logpdf_component_batch(
            np.array([ms.MEAN1_X]), np.array([ms.MEAN1_Y]), ms.MEAN1_X, ms.MEAN1_Y
        )
        self.assertTrue(np.allclose(lp, ms.LOG_NORM_2D))

    def test_logpdf_mixture_symmetry(self):
        a = ms.logpdf_mixture_batch(np.array([ms.MEAN1_X]), np.array([ms.MEAN1_Y]))
        b = ms.logpdf_mixture_batch(np.array([ms.MEAN2_X]), np.array([ms.MEAN2_Y]))
        self.assertTrue(np.allclose(a, b))

    def test_logpdf_mixture_batch_numeric(self):
        xs = np.array([ms.MEAN1_X, ms.MEAN2_X, 0.0, 5.0, -5.0])
        ys = np.array([ms.MEAN1_Y, ms.MEAN2_Y, 0.0, -5.0, 5.0])
        lp_ms = ms.logpdf_mixture_batch(xs, ys)
        lp_ref = self._ref_logpdf_mixture_np(xs, ys)
        self.assertTrue(np.all(np.isfinite(lp_ms)))
        self.assertTrue(np.allclose(lp_ms, lp_ref, rtol=1e-8, atol=1e-10))

    def test_logpdf_batch_calls_model_and_matches_reference(self):
        model = DummyEnergyModel("cpu")
        xs = np.array([ms.MEAN1_X, ms.MEAN2_X, 0.0, 3.0], dtype=np.float32)
        ys = np.array([ms.MEAN1_Y, ms.MEAN2_Y, 0.0, -3.0], dtype=np.float32)
        out = ms.logpdf_batch(torch.from_numpy(xs), torch.from_numpy(ys), model)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.shape, (len(xs),))
        lp_ref = self._ref_logpdf_mixture_np(xs, ys)
        self.assertTrue(np.allclose(out.numpy(), lp_ref, rtol=1e-6, atol=1e-6))

    def test_double_factorial_odd(self):
        self.assertEqual(ms.double_factorial_odd(1), 1.0)
        self.assertEqual(ms.double_factorial_odd(3), 3.0)
        self.assertEqual(ms.double_factorial_odd(5), 15.0)
        self.assertEqual(ms.double_factorial_odd(0), 1.0)

    def test_gaussian_even_moment(self):
        s = 2.0
        self.assertEqual(ms.gaussian_even_moment(0, s), 1.0)
        self.assertEqual(ms.gaussian_even_moment(1, s), 0.0)
        self.assertEqual(ms.gaussian_even_moment(2, s), s**2)
        self.assertEqual(ms.gaussian_even_moment(4, s), 3.0 * s**4)

    def test_raw_and_central_moments(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        raw = ms.raw_moments(v, 3)
        self.assertTrue(np.allclose(raw[1:], [2.5, 7.5, 25.0]))
        mu, cent = ms.central_moments(v, 3)
        self.assertAlmostEqual(mu, 2.5)
        self.assertTrue(abs(cent[1]) < 1e-12)
        self.assertAlmostEqual(cent[2], 1.25)
        self.assertAlmostEqual(cent[3], 0.0)

    def test_cumulants_from_central_closed_forms(self):
        mu = 0.0
        C = np.zeros(9)
        C[2] = 2.0
        C[3] = 3.0
        C[4] = 5.0
        C[5] = 7.0
        C[6] = 11.0
        C[7] = 13.0
        C[8] = 17.0
        K = ms.cumulants_from_central(mu, C, 8)
        self.assertAlmostEqual(K[1], mu)
        self.assertAlmostEqual(K[2], C[2])
        self.assertAlmostEqual(K[3], C[3])
        self.assertAlmostEqual(K[4], C[4] - 3 * C[2] ** 2)
        self.assertAlmostEqual(K[5], C[5] - 10 * C[3] * C[2])
        self.assertAlmostEqual(K[6], C[6] - 15 * C[4] * C[2] + 30 * C[2] ** 3)
        self.assertAlmostEqual(
            K[7], C[7] - 21 * C[5] * C[2] - 35 * C[4] * C[3] + 210 * C[3] * C[2] ** 2
        )
        self.assertAlmostEqual(
            K[8],
            C[8]
            - 28 * C[6] * C[2]
            - 35 * C[4] ** 2
            + 420 * C[4] * C[2] ** 2
            - 630 * C[2] ** 4,
        )

    def _gauss_even_moment(self, order, sigma):
        if order % 2 == 1:
            return 0.0
        if order == 0:
            return 1.0
        n = order // 2
        df = 1.0
        k = 1
        while k <= (2 * n - 1):
            df *= float(k)
            k += 2
        return df * (sigma ** (2 * n))

    def _expected_raw_even(self, k, m_abs, sigma):
        acc = 0.0
        for i in range(0, k + 1, 2):
            acc += comb(k, i) * (m_abs**i) * self._gauss_even_moment(k - i, sigma)
        return acc

    def test_analytic_raw_marginal_closed_forms(self):
        m, s = ms.m_value, ms.SIGMA
        R = ms.analytic_raw_marginal(8, m, s)
        self.assertTrue(np.allclose(R[1::2], 0.0))
        self.assertAlmostEqual(R[2], m**2 + s**2)
        self.assertAlmostEqual(R[4], m**4 + 6 * m * m * s * s + 3 * s**4)
        self.assertAlmostEqual(
            R[6], m**6 + 15 * m**4 * s**2 + 45 * m**2 * s**4 + 15 * s**6
        )
        self.assertAlmostEqual(R[8], self._expected_raw_even(8, m, s))

    def test_analytic_cumulants_closed_forms(self):
        m, s = ms.m_value, ms.SIGMA
        K = ms.analytic_cumulants(8, m, s)
        self.assertTrue(np.allclose(K[1::2], 0.0))
        self.assertAlmostEqual(K[2], s**2 + m**2)
        self.assertAlmostEqual(K[4], -2.0 * m**4)
        self.assertAlmostEqual(K[6], 16.0 * m**6)
        self.assertAlmostEqual(K[8], -272.0 * m**8)

    def test_bootstrap_reproducible_and_shapes(self):
        # Use global test seed for determinism; just validate API and outputs
        data = np.random.randn(100, 2)
        means, errs = ms.bootstrap(data, n_boot=BOOT_N)
        self.assertEqual(means.shape, (2,))
        self.assertEqual(errs.shape, (2,))
        self.assertTrue(np.isfinite(means).all())
        self.assertTrue(np.isfinite(errs).all())

    def test_mh_parallel_api_invariants_and_seed(self):
        cfg = dict(
            n_chains=32,
            n_keep=5,
            burn_in=50,
            thin=3,
            seed=321,
            init_prop_std=0.3,
            adapt=True,
            adapt_window=20,
        )
        s1, a1 = ms.mh_parallel(**cfg)
        s2, a2 = ms.mh_parallel(**cfg)
        self.assertEqual(s1.shape, (cfg["n_chains"] * cfg["n_keep"], 2))
        self.assertTrue(np.isfinite(s1).all())
        self.assertTrue(0.0 <= a1 <= 1.0)
        self.assertTrue(np.allclose(s1, s2))
        self.assertAlmostEqual(a1, a2)
        # different seed ⇒ not equal
        cfg2 = dict(cfg, seed=cfg["seed"] + 1)
        s3, a3 = ms.mh_parallel(**cfg2)
        self.assertFalse(np.allclose(s1, s3))

    def test_torch_mh_parallel_api_invariants_and_seed(self):
        model = DummyEnergyModel("cpu")
        cfg = dict(
            n_chains=32,
            n_keep=5,
            burn_in=50,
            thin=3,
            seed=777,
            init_prop_std=0.3,
            adapt=True,
            adapt_window=20,
            model=model,
        )
        s1, a1 = ms.torch_mh_parallel(**cfg)
        s2, a2 = ms.torch_mh_parallel(**cfg)
        self.assertEqual(s1.shape, (cfg["n_chains"] * cfg["n_keep"], 2))
        self.assertTrue(torch.isfinite(s1).all())
        self.assertTrue(0.0 <= a1 <= 1.0)
        self.assertTrue(torch.allclose(s1, s2))
        self.assertAlmostEqual(a1, a2)
        s3, a3 = ms.torch_mh_parallel(**dict(cfg, seed=cfg["seed"] + 1))
        self.assertFalse(torch.allclose(s1, s3))

    def test_odd_moment_zero_for_symmetric_about_center(self):
        # symmetric around 0 along axis=0
        x = np.array(
            [[-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, 0.0, 0.0, 0.0, 3.0]], dtype=float
        ).T  # shape (5,2)
        # third central moment around 0 should be ~0 per column
        expected = np.mean((x - 0.0) ** 3, axis=0)
        out = ms.moment(x, order=3, axis=0)
        np.testing.assert_allclose(out, expected)
        np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-15)

    def test_bad_center_shape_raises(self):
        x = np.random.RandomState(1).randn(4, 3)
        bad_center = np.array([1.0, 2.0])  # not broadcastable to (4,3) along axis=0
        with self.assertRaises(ValueError):
            # Numpy will raise during broadcasting inside (x - center)
            ms.moment(x, order=2, axis=0, center=bad_center)


if __name__ == "__main__":
    unittest.main()
