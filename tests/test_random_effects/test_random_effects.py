import numpy as np
from pathlib import Path

from diffusion_models.effects.random_effects import RandomEffectsAnalyser


def _make_npz(path: Path, val: np.ndarray, err: np.ndarray):
    np.savez(path, val=val, err=err)


def test_load_val_err_reads_multiple_files(tmp_path):
    # Create 3 files with simple 1D arrays
    vals = [np.array([1.0, 2.0]), np.array([1.5, 1.0]), np.array([0.5, 3.0])]
    errs = [np.array([0.1, 0.2]), np.array([0.2, 0.1]), np.array([0.3, 0.4])]

    files = []
    for i, (v, e) in enumerate(zip(vals, errs)):
        f = tmp_path / f"run_{i}.npz"
        _make_npz(f, v, e)
        files.append(str(f))

    analyser = RandomEffectsAnalyser.from_file_paths(files)
    Y_i, eps_i = analyser.Y_i, analyser.eps_i

    assert Y_i.shape == (3, 2)
    assert eps_i.shape == (3, 2)
    np.testing.assert_allclose(Y_i, np.stack(vals))
    np.testing.assert_allclose(eps_i, np.stack(errs))


def test_zero_heterogeneity_reduces_sys_to_zero():
    # All runs have identical Y → between-run variance tau^2 = 0 → sigma_sys_mean = 0
    Y = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    eps = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

    analyser = RandomEffectsAnalyser(Y, eps)
    Y_hat, sigma_stat, sigma_sys, sigma_tot = analyser.analyze()

    # Mean should equal the common value
    np.testing.assert_allclose(Y_hat, Y[0])
    # Systematic component should be zero (within floating tol)
    assert np.allclose(sigma_sys, 0.0)
    # Totals should be finite and positive
    assert np.all(np.isfinite(sigma_tot))
    assert np.all(sigma_tot > 0)


def test_multidimensional_support():
    # Verify behavior on 3D data (runs, h, w)
    Y = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [3.0, 5.0]],
        ]
    )
    eps = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
        ]
    )

    analyser = RandomEffectsAnalyser(Y, eps)
    Y_hat, sigma_stat, sigma_sys, sigma_tot = analyser.analyze()

    # Shapes should drop the run dimension
    assert Y_hat.shape == (2, 2)
    assert sigma_stat.shape == (2, 2)
    assert sigma_sys.shape == (2, 2)
    assert sigma_tot.shape == (2, 2)
    # Values should be finite
    assert np.all(np.isfinite(Y_hat))
    assert np.all(np.isfinite(sigma_stat))
    assert np.all(np.isfinite(sigma_sys))
    assert np.all(np.isfinite(sigma_tot))
