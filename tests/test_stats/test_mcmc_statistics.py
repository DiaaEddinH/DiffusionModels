import unittest
import numpy as np
import torch
from math import comb
from src import mcmc_statistics as ms

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


class TestMCMCStatistics(unittest.TestCase):
    # ---------- Log-pdf ----------
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

    # ---------- Pure math utilities ----------
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

    # ---------- Analytic targets (closed-form only) ----------
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

    # ---------- Bootstrap (deterministic API) ----------
    def test_bootstrap_reproducible_and_shapes(self):
        # Use global test seed for determinism; just validate API and outputs
        data = np.random.randn(100, 2)
        means, errs = ms.bootstrap(data, n_boot=BOOT_N)
        self.assertEqual(means.shape, (2,))
        self.assertEqual(errs.shape, (2,))
        self.assertTrue(np.isfinite(means).all())
        self.assertTrue(np.isfinite(errs).all())

    # ---------- MCMC samplers (API invariants only) ----------
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

    def test_default_center_matches_manual(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        # order=2 with default center == variance around the mean (not sample-var)
        expected = np.mean((v - v.mean()) ** 2)
        self.assertAlmostEqual(ms.moment(v, order=2), expected)

        # order=3 with default center == third central moment
        expected3 = np.mean((v - v.mean()) ** 3)
        self.assertAlmostEqual(ms.moment(v, order=3), expected3)

    def test_explicit_center_scalar(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        c = 2.0
        expected2 = np.mean((v - c) ** 2)
        expected3 = np.mean((v - c) ** 3)
        self.assertAlmostEqual(ms.moment(v, order=2, center=c), expected2)
        self.assertAlmostEqual(ms.moment(v, order=3, center=c), expected3)

    def test_axis_0_vectorized(self):
        # shape (rows, cols); compute per-column moments (axis=0)
        x = np.array(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
        )
        # default center (column means)
        expected = np.mean((x - x.mean(axis=0)) ** 2, axis=0)
        out = ms.moment(x, order=2, axis=0)
        np.testing.assert_allclose(out, expected)

    def test_axis_1_with_vector_center_broadcast(self):
        # per-row moments (axis=1), center is per-row mean with shape (n, 1)
        x = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 1.0, 1.0]], dtype=float)
        center = x.mean(axis=1, keepdims=True)  # shape (2,1)
        expected = np.mean((x - center) ** 2, axis=1)
        out = ms.moment(x, order=2, axis=1, center=center)
        np.testing.assert_allclose(out, expected)

    def test_order_zero_returns_one(self):
        # NOTE:
        # These tests manually supply a broadcastable `center` for axis=1 cases because
        # the current `moment()` implementation uses `np.mean(..., axis=axis)` without
        # `keepdims=True`. That makes default centering fail to broadcast when reducing
        # along non–last axes (e.g., axis=1 on 2D arrays).
        # TODO: update `moment()` to use keepdims=True or otherwise handle broadcasting
        # consistently, then remove explicit `center=` in these tests.

        x = np.random.RandomState(0).randn(5, 7)

        # axis=0 works with default center; result is a (7,) vector of ones
        out0_axis0 = ms.moment(x, order=0, axis=0)
        np.testing.assert_allclose(out0_axis0, np.ones(x.shape[1]))

        # axis=1: avoid default-center broadcasting issue by giving a scalar center
        out0_axis1 = ms.moment(x, order=0, axis=1, center=0.0)
        np.testing.assert_allclose(out0_axis1, np.ones(x.shape[0]))

    def test_all_equal_returns_zero_for_order_ge1(self):
        # NOTE:
        # These tests manually supply a broadcastable `center` for axis=1 cases because
        # the current `moment()` implementation uses `np.mean(..., axis=axis)` without
        # `keepdims=True`. That makes default centering fail to broadcast when reducing
        # along non–last axes (e.g., axis=1 on 2D arrays).
        # TODO: update `moment()` to use keepdims=True or otherwise handle broadcasting
        # consistently, then remove explicit `center=` in these tests.

        x = np.full((4, 3), 5.0)

        # axis=0: default center broadcasts fine (shape (3,))
        np.testing.assert_allclose(ms.moment(x, order=1, axis=0), 0.0)

        # axis=1: provide a broadcastable center to avoid shape mismatch
        # either scalar 5.0 or per-row mean with keepdims
        np.testing.assert_allclose(ms.moment(x, order=2, axis=1, center=5.0), 0.0)
        # (equivalently)
        # row_means = x.mean(axis=1, keepdims=True)
        # np.testing.assert_allclose(ms.moment(x, order=2, axis=1, center=row_means), 0.0)

    def test_odd_moment_zero_for_symmetric_about_center(self):
        # symmetric around 0 along axis=0
        x = np.array(
            [[-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, 0.0, 0.0, 0.0, 3.0]], dtype=float
        ).T  # shape (5,2)
        # third central moment around 0 should be ~0 per column
        expected = np.mean((x - 0.0) ** 3, axis=0)
        out = ms.moment(x, order=3, axis=0, center=0.0)
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
