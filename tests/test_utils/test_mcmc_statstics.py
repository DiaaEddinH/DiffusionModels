import unittest
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import torch
from math import comb

from src import mcmc_statistics as ms


class DummyEnergyModel:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    # Return a simple stable log-density proportional to the true mixture
    def energy(self, x, t):
        # x: [N, 2]
        # Compute the same mixture logpdf in torch for testing
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
    # ---------------- Log-pdf tests ----------------
    def test_logpdf_component_at_mean(self):
        x0 = np.array([ms.MEAN1_X])
        x1 = np.array([ms.MEAN1_Y])
        lp = ms.logpdf_component_batch(x0, x1, ms.MEAN1_X, ms.MEAN1_Y)
        # At the mean of the component, quadratic term is zero
        self.assertTrue(np.allclose(lp, ms.LOG_NORM_2D))

    def test_logpdf_mixture_symmetry(self):
        # Evaluate mixture log-pdf at the two component means; they should be equal
        a = ms.logpdf_mixture_batch(np.array([ms.MEAN1_X]), np.array([ms.MEAN1_Y]))
        b = ms.logpdf_mixture_batch(np.array([ms.MEAN2_X]), np.array([ms.MEAN2_Y]))
        self.assertTrue(np.allclose(a, b))

    def _ref_logpdf_mixture_np(self, x0, x1):
        # Independent stable implementation (no calls into ms.logpdf_*)
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

    def test_logpdf_mixture_batch_numeric(self):
        # Check multiple points and extreme tails for numerical stability
        xs = np.array([ms.MEAN1_X, ms.MEAN2_X, 0.0, 5.0, -5.0])
        ys = np.array([ms.MEAN1_Y, ms.MEAN2_Y, 0.0, -5.0, 5.0])
        lp_ms = ms.logpdf_mixture_batch(xs, ys)
        lp_ref = self._ref_logpdf_mixture_np(xs, ys)
        self.assertTrue(np.allclose(lp_ms, lp_ref, rtol=1e-10, atol=1e-12))
        self.assertTrue(np.all(np.isfinite(lp_ms)))

    def test_logpdf_batch_calls_model(self):
        model = DummyEnergyModel(device="cpu")
        # Batch of points
        xs = np.array([ms.MEAN1_X, ms.MEAN2_X, 0.0, 3.0])
        ys = np.array([ms.MEAN1_Y, ms.MEAN2_Y, 0.0, -3.0])
        x0 = torch.tensor(xs, dtype=torch.float32)
        x1 = torch.tensor(ys, dtype=torch.float32)
        out = ms.logpdf_batch(x0, x1, model)
        self.assertTrue(torch.is_tensor(out))
        self.assertEqual(out.shape, (len(xs),))
        # Numeric agreement with reference NumPy implementation
        lp_ref = self._ref_logpdf_mixture_np(xs, ys)
        self.assertTrue(np.allclose(out.numpy(), lp_ref, rtol=1e-6, atol=1e-6))

    # ---------------- Moment/cumulant utilities ----------------
    def test_double_factorial_odd(self):
        # 1!! = 1, 3!! = 3, 5!! = 15
        self.assertEqual(ms.double_factorial_odd(1), 1.0)
        self.assertEqual(ms.double_factorial_odd(3), 3.0)
        self.assertEqual(ms.double_factorial_odd(5), 15.0)
        # edge case n < 1
        self.assertEqual(ms.double_factorial_odd(0), 1.0)

    def test_gaussian_even_moment(self):
        sigma = 2.0
        # 0th even moment = 1
        self.assertEqual(ms.gaussian_even_moment(0, sigma), 1.0)
        # Odd orders should be 0
        self.assertEqual(ms.gaussian_even_moment(1, sigma), 0.0)
        # 2nd moment of N(0, sigma^2) is sigma^2
        self.assertEqual(ms.gaussian_even_moment(2, sigma), sigma**2)
        # 4th moment = 3 sigma^4 for standard Gaussian; via double factorial formula
        m4 = ms.gaussian_even_moment(4, sigma)
        self.assertEqual(m4, 3.0 * sigma**4)

    def test_raw_and_central_moments(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        raw = ms.raw_moments(v, 3)
        # mean of v, mean of v^2, mean of v^3
        self.assertTrue(np.allclose(raw[1:], [2.5, 7.5, 25.0]))

        mu, cent = ms.central_moments(v, 3)
        self.assertAlmostEqual(mu, 2.5)
        # central 1st moment ~ 0, 2nd is variance, 3rd is third central moment
        self.assertTrue(abs(cent[1]) < 1e-12)
        self.assertAlmostEqual(cent[2], 1.25)
        self.assertAlmostEqual(cent[3], 0.0)

    def test_cumulants_from_central(self):
        # Use symmetric distribution around 0 to simplify: v in {-1,1}
        v = np.array([-1.0, 1.0, -1.0, 1.0])
        mu, C = ms.central_moments(v, 6)
        K = ms.cumulants_from_central(mu, C, 6)
        # For symmetric around mean=0: K1=0, K3=0
        self.assertAlmostEqual(K[1], 0.0)
        self.assertAlmostEqual(K[3], 0.0)
        # K2 equals variance
        self.assertAlmostEqual(K[2], C[2])
        # Check higher-order formulae are finite numbers
        self.assertTrue(np.isfinite(K[4]))
        self.assertTrue(np.isfinite(K[5]))
        self.assertTrue(np.isfinite(K[6]))

    def test_cumulants_from_central_maxk_1(self):
        mu = 3.14
        C = np.zeros(2)
        K = ms.cumulants_from_central(mu, C, 1)
        self.assertEqual(len(K), 2)
        self.assertAlmostEqual(K[1], mu)

    def test_cumulants_from_central_maxk_2(self):
        mu = 0.0
        C = np.zeros(3)
        C[2] = 2.5
        K = ms.cumulants_from_central(mu, C, 2)
        self.assertEqual(len(K), 3)
        self.assertAlmostEqual(K[2], 2.5)

    def test_cumulants_from_central_maxk_3(self):
        mu = 0.0
        C = np.zeros(4)
        C[3] = 7.0
        K = ms.cumulants_from_central(mu, C, 3)
        self.assertEqual(len(K), 4)
        self.assertAlmostEqual(K[3], 7.0)

    def test_cumulants_from_central_maxk_4(self):
        mu = 0.0
        C = np.zeros(5)
        C[2] = 2.0
        C[4] = 50.0
        K = ms.cumulants_from_central(mu, C, 4)
        self.assertEqual(len(K), 5)
        self.assertAlmostEqual(K[4], 50.0 - 3.0 * (2.0**2))

    def test_cumulants_from_central_maxk_5(self):
        mu = 0.0
        C = np.zeros(6)
        C[2] = 2.0
        C[3] = 3.0
        C[5] = 100.0
        K = ms.cumulants_from_central(mu, C, 5)
        self.assertEqual(len(K), 6)
        self.assertAlmostEqual(K[5], 100.0 - 10.0 * 3.0 * 2.0)

    def test_cumulants_from_central_maxk_6(self):
        mu = 0.0
        C = np.zeros(7)
        C[2] = 2.0
        C[4] = 5.0
        C[6] = 200.0
        K = ms.cumulants_from_central(mu, C, 6)
        self.assertEqual(len(K), 7)
        expected = 200.0 - 15.0 * 5.0 * 2.0 + 30.0 * (2.0**3)
        self.assertAlmostEqual(K[6], expected)

    def test_cumulants_from_central_maxk_7(self):
        mu = 0.0
        C = np.zeros(8)
        C[2] = 2.0
        C[3] = 3.0
        C[4] = 5.0
        C[5] = 7.0
        C[7] = 300.0
        K = ms.cumulants_from_central(mu, C, 7)
        self.assertEqual(len(K), 8)
        expected = 300.0 - 21.0 * 7.0 * 2.0 - 35.0 * 5.0 * 3.0 + 210.0 * 3.0 * (2.0**2)
        self.assertAlmostEqual(K[7], expected)

    def test_cumulants_from_central_maxk_8(self):
        mu = 0.0
        C = np.zeros(9)
        C[2] = 2.0
        C[4] = 5.0
        C[6] = 200.0
        C[8] = 1000.0
        K = ms.cumulants_from_central(mu, C, 8)
        self.assertEqual(len(K), 9)
        expected = (
            1000.0
            - 28.0 * 200.0 * 2.0
            - 35.0 * (5.0**2)
            + 420.0 * 5.0 * (2.0**2)
            - 630.0 * (2.0**4)
        )
        self.assertAlmostEqual(K[8], expected)

    # ---------------- Analytic targets ----------------
    def test_analytic_functions_consistency(self):
        max_k = 8
        m = ms.m_value
        s = ms.SIGMA
        R = ms.analytic_raw_marginal(max_k, m, s)
        K = ms.analytic_cumulants(max_k, m, s)
        # Odd raw moments should be zero by symmetry
        self.assertTrue(np.allclose(R[1::2], 0.0))
        # Check specific even raw moments against closed forms (no production helpers)
        self.assertAlmostEqual(R[2], m**2 + s**2)
        self.assertAlmostEqual(R[4], m**4 + 6*m*m*s*s + 3*s**4)
        self.assertAlmostEqual(R[6], m**6 + 15*m**4*s**2 + 45*m**2*s**4 + 15*s**6)
        # Odd cumulants should be zero
        self.assertTrue(np.allclose(K[1::2], 0.0))
        # Specific cumulants
        self.assertAlmostEqual(K[2], s**2 + m**2)
        self.assertAlmostEqual(K[4], -2.0 * m**4)
        self.assertAlmostEqual(K[6], 16.0 * m**6)
        self.assertAlmostEqual(K[8], -272.0 * m**8)

    def _gauss_even_moment(self, order, sigma):
        if order % 2 == 1:
            return 0.0
        if order == 0:
            return 1.0
        # (2n-1)!! * sigma^{2n}
        n = order // 2
        # compute (2n-1)!!
        df = 1.0
        k = 1
        while k <= (2 * n - 1):
            df *= float(k)
            k += 2
        return df * (sigma ** (2 * n))

    def _expected_raw_even(self, k, m_abs, sigma):
        # compute sum_{i even} C(k,i) m^i E[Z^{k-i}] with Z~N(0,sigma^2)
        acc = 0.0
        for i in range(0, k + 1, 2):
            acc += comb(k, i) * (m_abs**i) * self._gauss_even_moment(k - i, sigma)
        return acc

    def test_analytic_raw_marginal_maxk_1(self):
        R = ms.analytic_raw_marginal(1, ms.m_value, ms.SIGMA)
        self.assertEqual(len(R), 2)
        self.assertAlmostEqual(R[1], 0.0)

    def test_analytic_raw_marginal_maxk_2(self):
        m = ms.m_value
        s = ms.SIGMA
        R = ms.analytic_raw_marginal(2, m, s)
        self.assertEqual(len(R), 3)
        self.assertAlmostEqual(R[2], self._expected_raw_even(2, m, s))

    def test_analytic_raw_marginal_maxk_3(self):
        R = ms.analytic_raw_marginal(3, ms.m_value, ms.SIGMA)
        self.assertEqual(len(R), 4)
        self.assertAlmostEqual(R[3], 0.0)

    def test_analytic_raw_marginal_maxk_4(self):
        m = ms.m_value
        s = ms.SIGMA
        R = ms.analytic_raw_marginal(4, m, s)
        self.assertEqual(len(R), 5)
        self.assertAlmostEqual(R[4], self._expected_raw_even(4, m, s))

    def test_analytic_raw_marginal_maxk_5(self):
        R = ms.analytic_raw_marginal(5, ms.m_value, ms.SIGMA)
        self.assertEqual(len(R), 6)
        self.assertAlmostEqual(R[5], 0.0)

    def test_analytic_raw_marginal_maxk_6(self):
        m = ms.m_value
        s = ms.SIGMA
        R = ms.analytic_raw_marginal(6, m, s)
        self.assertEqual(len(R), 7)
        self.assertAlmostEqual(R[6], self._expected_raw_even(6, m, s))

    def test_analytic_raw_marginal_maxk_7(self):
        R = ms.analytic_raw_marginal(7, ms.m_value, ms.SIGMA)
        self.assertEqual(len(R), 8)
        self.assertAlmostEqual(R[7], 0.0)

    def test_analytic_raw_marginal_maxk_8(self):
        m = ms.m_value
        s = ms.SIGMA
        R = ms.analytic_raw_marginal(8, m, s)
        self.assertEqual(len(R), 9)
        self.assertAlmostEqual(R[8], self._expected_raw_even(8, m, s))

    def test_analytic_cumulants_maxk_1(self):
        K = ms.analytic_cumulants(1, ms.m_value, ms.SIGMA)
        self.assertEqual(len(K), 2)
        self.assertAlmostEqual(K[1], 0.0)

    def test_analytic_cumulants_maxk_2(self):
        K = ms.analytic_cumulants(2, ms.m_value, ms.SIGMA)
        self.assertEqual(len(K), 3)
        self.assertAlmostEqual(K[2], ms.SIGMA**2 + ms.m_value**2)

    def test_analytic_cumulants_maxk_3(self):
        K = ms.analytic_cumulants(3, ms.m_value, ms.SIGMA)
        self.assertEqual(len(K), 4)
        self.assertAlmostEqual(K[3], 0.0)

    def test_analytic_cumulants_maxk_4(self):
        m = ms.m_value
        K = ms.analytic_cumulants(4, m, ms.SIGMA)
        self.assertEqual(len(K), 5)
        self.assertAlmostEqual(K[4], -2.0 * (m**4))

    def test_analytic_cumulants_maxk_5(self):
        K = ms.analytic_cumulants(5, ms.m_value, ms.SIGMA)
        self.assertEqual(len(K), 6)
        self.assertAlmostEqual(K[5], 0.0)

    def test_analytic_cumulants_maxk_6(self):
        m = ms.m_value
        K = ms.analytic_cumulants(6, m, ms.SIGMA)
        self.assertEqual(len(K), 7)
        self.assertAlmostEqual(K[6], 16.0 * (m**6))

    def test_analytic_cumulants_maxk_7(self):
        K = ms.analytic_cumulants(7, ms.m_value, ms.SIGMA)
        self.assertEqual(len(K), 8)
        self.assertAlmostEqual(K[7], 0.0)

    def test_analytic_cumulants_maxk_8(self):
        m = ms.m_value
        K = ms.analytic_cumulants(8, m, ms.SIGMA)
        self.assertEqual(len(K), 9)
        self.assertAlmostEqual(K[8], -272.0 * (m**8))

    # ---------------- Bootstrap ----------------
    def test_bootstrap_reproducible(self):
        # Use local RNGs for determinism independent of global state
        rs = np.random.RandomState(42)
        data = rs.randn(100, 2)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        means1, errs1 = ms.bootstrap(data, n_boot=200, rng=rng1)
        means2, errs2 = ms.bootstrap(data, n_boot=200, rng=rng2)
        # Deterministic with same seed
        self.assertTrue(np.allclose(means1, means2))
        self.assertTrue(np.allclose(errs1, errs2))
        # Shape checks
        self.assertEqual(means1.shape, (2,))
        self.assertEqual(errs1.shape, (2,))
        # Mean of bootstrap means should be close to sample mean (loose tolerance)
        sample_mean = data.mean(axis=0)
        self.assertTrue(np.allclose(means1, sample_mean, atol=0.1))

    # ---------------- Misc moment ----------------
    def test_moment_function(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        # default center = mean
        m2 = ms.moment(v, 2)
        self.assertAlmostEqual(m2, np.mean((v - v.mean()) ** 2))
        # explicitly provided center
        m3_center_2 = ms.moment(v, 3, center=2.0)
        self.assertAlmostEqual(m3_center_2, np.mean((v - 2.0) ** 3))

    # ---------------- MCMC samplers ----------------
    def test_mh_parallel_moments_correctness(self):
        # Use enough samples for stable statistics while keeping runtime modest
        n_chains = 128
        n_keep = 24
        burn_in = 400
        thin = 5
        seed = 7
        samples, acc_rate = ms.mh_parallel(
            n_chains=n_chains,
            n_keep=n_keep,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            init_prop_std=0.3,
            adapt=True,
            adapt_window=50,
        )
        # Basic sanity
        self.assertEqual(samples.shape, (n_chains * n_keep, 2))
        self.assertTrue(0.0 <= acc_rate <= 1.0)
        # Acceptance rate within a sensible band (very loose)
        self.assertTrue(0.05 <= acc_rate <= 0.8)

        xs = samples[:, 0]
        ys = samples[:, 1]

        # Analytic targets for 1D marginals
        max_k = 3
        R = ms.analytic_raw_marginal(max_k, ms.m_value, ms.SIGMA)
        target_m2 = R[2]
        target_m3 = R[3]
        target_xy = - (ms.m_value ** 2)

        # Tolerances (3-6 sigma bands for the chosen sample size ~3k)
        tol_mean = 0.06
        tol_m2 = 0.05
        tol_m3 = 0.10
        tol_xy = 0.12

        # Means approximately zero (symmetry)
        self.assertLess(abs(float(np.mean(xs))), tol_mean)
        self.assertLess(abs(float(np.mean(ys))), tol_mean)

        # Second raw moment close to analytic σ^2 + m^2
        self.assertLess(abs(float(np.mean(xs ** 2)) - target_m2), tol_m2)
        self.assertLess(abs(float(np.mean(ys ** 2)) - target_m2), tol_m2)

        # Third raw moment approximately zero (symmetry)
        self.assertLess(abs(float(np.mean(xs ** 3)) - target_m3), tol_m3)
        self.assertLess(abs(float(np.mean(ys ** 3)) - target_m3), tol_m3)

        # Cross-moment E[XY] ~ -m^2 due to anti-correlation of mixture modes
        self.assertLess(abs(float(np.mean(xs * ys)) - target_xy), tol_xy)

    def test_torch_mh_parallel_moments_correctness(self):
        model = DummyEnergyModel(device="cpu")
        # Use same configuration as numpy version for parity
        n_chains = 128
        n_keep = 24
        burn_in = 400
        thin = 5
        seed = 123
        samples, acc_rate = ms.torch_mh_parallel(
            n_chains=n_chains,
            n_keep=n_keep,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
            init_prop_std=0.3,
            adapt=True,
            adapt_window=50,
            model=model,
        )
        self.assertEqual(samples.shape, (n_chains * n_keep, 2))
        self.assertTrue(0.0 <= acc_rate <= 1.0)
        self.assertTrue(0.05 <= acc_rate <= 0.8)

        xs = samples[:, 0].numpy()
        ys = samples[:, 1].numpy()

        # Analytic targets
        max_k = 3
        R = ms.analytic_raw_marginal(max_k, ms.m_value, ms.SIGMA)
        target_m2 = R[2]
        target_m3 = R[3]
        target_xy = - (ms.m_value ** 2)

        tol_mean = 0.06
        tol_m2 = 0.05
        tol_m3 = 0.10
        tol_xy = 0.12

        self.assertLess(abs(float(np.mean(xs))), tol_mean)
        self.assertLess(abs(float(np.mean(ys))), tol_mean)

        self.assertLess(abs(float(np.mean(xs ** 2)) - target_m2), tol_m2)
        self.assertLess(abs(float(np.mean(ys ** 2)) - target_m2), tol_m2)

        self.assertLess(abs(float(np.mean(xs ** 3)) - target_m3), tol_m3)
        self.assertLess(abs(float(np.mean(ys ** 3)) - target_m3), tol_m3)

        self.assertLess(abs(float(np.mean(xs * ys)) - target_xy), tol_xy)

    def test_mh_parallel_reproducible(self):
        cfg = dict(n_chains=32, n_keep=5, burn_in=50, thin=3, seed=321, init_prop_std=0.3, adapt=True, adapt_window=20)
        s1, a1 = ms.mh_parallel(**cfg)
        s2, a2 = ms.mh_parallel(**cfg)
        self.assertTrue(np.allclose(s1, s2))
        self.assertAlmostEqual(a1, a2)

    def test_torch_mh_parallel_reproducible(self):
        model = DummyEnergyModel(device="cpu")
        cfg = dict(n_chains=32, n_keep=5, burn_in=50, thin=3, seed=777, init_prop_std=0.3, adapt=True, adapt_window=20, model=model)
        s1, a1 = ms.torch_mh_parallel(**cfg)
        s2, a2 = ms.torch_mh_parallel(**cfg)
        self.assertTrue(torch.allclose(s1, s2))
        self.assertAlmostEqual(a1, a2)

    def _extract_chain_series(self, samples, n_chains, chain_idx=0):
        # samples shape: (n_chains*n_keep, 2). Extract x coordinate for a chain across kept steps
        # kept step j contributes block [j*n_chains : (j+1)*n_chains)
        n_total = samples.shape[0]
        n_keep = n_total // n_chains
        idxs = chain_idx + np.arange(n_keep) * n_chains
        return samples[idxs, 0]  # x coordinate

    def _lag1_corr(self, x):
        if len(x) < 2:
            return 0.0
        x0 = x[:-1]
        x1 = x[1:]
        x0 = x0 - np.mean(x0)
        x1 = x1 - np.mean(x1)
        denom = np.sqrt(np.mean(x0**2) * np.mean(x1**2))
        if denom == 0:
            return 0.0
        return float(np.mean(x0 * x1) / denom)

    def test_thinning_reduces_autocorr(self):
        # Compare autocorrelation with thin=1 vs thin=10
        cfg_lo = dict(n_chains=64, n_keep=20, burn_in=200, thin=1, seed=2024, init_prop_std=0.3, adapt=True, adapt_window=50)
        cfg_hi = dict(n_chains=64, n_keep=20, burn_in=200, thin=10, seed=2024, init_prop_std=0.3, adapt=True, adapt_window=50)
        s_lo, _ = ms.mh_parallel(**cfg_lo)
        s_hi, _ = ms.mh_parallel(**cfg_hi)
        # average lag1 correlation across a few chains
        acfs_lo = []
        acfs_hi = []
        for c in [0, 1, 2, 3, 4]:
            series_lo = self._extract_chain_series(s_lo, cfg_lo['n_chains'], chain_idx=c)
            series_hi = self._extract_chain_series(s_hi, cfg_hi['n_chains'], chain_idx=c)
            acfs_lo.append(self._lag1_corr(series_lo))
            acfs_hi.append(self._lag1_corr(series_hi))
        # Thinning should not increase lag-1 autocorrelation; allow small numerical wiggle
        self.assertLess(np.mean(acfs_hi), np.mean(acfs_lo) + 0.02)


if __name__ == "__main__":
    unittest.main()
