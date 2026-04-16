"""Tests verifying jaxwavelets matches pywt to machine precision."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt

import jaxwavelets as wt

WAVELETS = ["haar", "db2", "db4", "db8", "sym4", "sym8", "coif2", "coif4"]
MODES = ["symmetric", "reflect", "periodization"]
ATOL = 1e-14  # direct comparison: our output vs pywt output (near machine epsilon)
ATOL_RT = (
    1e-11  # roundtrip: idwt(dwt(x)) ≈ x (inherent transform precision, not our error)
)


# --- 1D forward ---


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("N", [2, 3, 4, 7, 8, 15, 16, 31, 32, 64])
@pytest.mark.parametrize("mode", MODES)
def test_dwt_matches_pywt(wavelet, N, mode):
    x_np = np.random.RandomState(0).randn(N)
    cA_jax, cD_jax = wt.dwt(jnp.array(x_np), wavelet, mode)
    cA_pywt, cD_pywt = pywt.dwt(x_np, wavelet, mode)
    np.testing.assert_allclose(np.array(cA_jax), cA_pywt, atol=ATOL)
    np.testing.assert_allclose(np.array(cD_jax), cD_pywt, atol=ATOL)


# --- 1D inverse ---


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("N", [2, 3, 4, 7, 8, 15, 16, 32])
@pytest.mark.parametrize("mode", MODES)
def test_idwt_matches_pywt(wavelet, N, mode):
    x_np = np.random.RandomState(0).randn(N)
    cA_pywt, cD_pywt = pywt.dwt(x_np, wavelet, mode)
    rec_pywt = pywt.idwt(cA_pywt, cD_pywt, wavelet, mode)
    rec_jax = wt.idwt(jnp.array(cA_pywt), jnp.array(cD_pywt), wavelet, mode)
    np.testing.assert_allclose(np.array(rec_jax[: len(rec_pywt)]), rec_pywt, atol=ATOL)


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("N", [2, 3, 4, 7, 8, 15, 16, 32])
@pytest.mark.parametrize("mode", MODES)
def test_perfect_reconstruction_1d(wavelet, N, mode):
    x = jnp.array(np.random.RandomState(0).randn(N))
    cA, cD = wt.dwt(x, wavelet, mode)
    rec = wt.idwt(cA, cD, wavelet, mode)[:N]
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL_RT)


# --- nD forward ---


@pytest.mark.parametrize("wavelet", ["haar", "db4", "sym4"])
@pytest.mark.parametrize("shape", [(16, 16), (15, 17), (8, 8, 8)])
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("mode", MODES)
def test_wavedecn_matches_pywt(wavelet, shape, level, mode):
    x_np = np.random.RandomState(0).randn(*shape)
    coeffs_jax = wt.wavedecn(jnp.array(x_np), wavelet, mode=mode, level=level)
    coeffs_pywt = pywt.wavedecn(x_np, wavelet, mode=mode, level=level)
    np.testing.assert_allclose(np.array(coeffs_jax.approx), coeffs_pywt[0], atol=ATOL)
    for jax_d, pywt_d in zip(coeffs_jax.details, coeffs_pywt[1:], strict=False):
        for key in jax_d:
            np.testing.assert_allclose(np.array(jax_d[key]), pywt_d[key], atol=ATOL)


# --- nD reconstruction ---


@pytest.mark.parametrize("wavelet", ["haar", "db4", "sym4"])
@pytest.mark.parametrize("shape", [(16, 16), (15, 17), (32, 32, 32), (15, 17, 19)])
@pytest.mark.parametrize("mode", MODES)
def test_perfect_reconstruction_nd(wavelet, shape, mode):
    x = jnp.array(np.random.RandomState(0).randn(*shape))
    coeffs = wt.wavedecn(x, wavelet, mode=mode)
    rec = wt.waverecn(coeffs, wavelet, mode=mode)
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL_RT)


@pytest.mark.parametrize("wavelet", ["haar", "db4"])
@pytest.mark.parametrize("shape", [(15, 17)])
def test_vmap_reconstruction_odd_shapes(wavelet, shape):
    """Regression: vmap over odd-shaped inputs exercises shape metadata trimming."""
    batch = jnp.stack(
        [jnp.array(np.random.RandomState(i).randn(*shape)) for i in range(3)]
    )
    f = partial(wt.wavedecn, wavelet=wavelet, level=2)
    g = partial(wt.waverecn, wavelet=wavelet)
    batch_rec = jax.vmap(lambda x: g(f(x)))(batch)
    np.testing.assert_allclose(np.array(batch_rec), np.array(batch), atol=ATOL_RT)


# --- JAX transforms ---


def test_grad():
    x = jnp.array(np.random.RandomState(0).randn(16, 16))
    g = jax.grad(lambda x: jnp.sum(wt.waverecn(wt.wavedecn(x, "db4"), "db4")))(x)
    np.testing.assert_allclose(np.array(g), np.ones_like(g), atol=ATOL_RT)


def test_vmap():
    batch = jnp.stack(
        [jnp.array(np.random.RandomState(i).randn(8, 8)) for i in range(4)]
    )
    batch_coeffs = jax.vmap(partial(wt.wavedecn, wavelet="haar", level=1))(batch)
    assert batch_coeffs.approx.shape == (4, 4, 4)
    batch_rec = jax.vmap(partial(wt.waverecn, wavelet="haar"))(batch_coeffs)
    np.testing.assert_allclose(np.array(batch_rec), np.array(batch), atol=ATOL_RT)


def test_jit_roundtrip():
    x = jnp.array(np.random.RandomState(0).randn(16, 16))
    roundtrip = jax.jit(
        lambda x: wt.waverecn(wt.wavedecn(x, "db4", level=2), "db4"),
    )
    np.testing.assert_allclose(np.array(roundtrip(x)), np.array(x), atol=ATOL_RT)
