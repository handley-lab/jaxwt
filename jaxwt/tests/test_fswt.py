"""Tests for fully separable wavelet decomposition."""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pywt
import pytest

from jaxwt._fswt import fswavedecn, fswaverecn

WAVELETS = ["haar", "db2", "db4"]
ATOL = 1e-14
ATOL_RT = 1e-11


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("shape", [(16, 16), (32, 16)])
@pytest.mark.parametrize("levels", [1, 2, (1, 2)])
def test_fswavedecn_approx_matches_pywt(wavelet, shape, levels):
    x_np = np.random.RandomState(0).randn(*shape)
    r_jax = fswavedecn(jnp.array(x_np), wavelet, levels=levels)
    r_pywt = pywt.fswavedecn(x_np, wavelet, levels=levels)
    np.testing.assert_allclose(np.array(r_jax.approx), r_pywt.approx, atol=ATOL)


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("shape", [(16, 16), (32, 16)])
def test_fswavedecn_coeffs_shape_matches_pywt(wavelet, shape):
    x_np = np.random.RandomState(0).randn(*shape)
    r_jax = fswavedecn(jnp.array(x_np), wavelet)
    r_pywt = pywt.fswavedecn(x_np, wavelet)
    assert r_jax.coeffs.shape == r_pywt.coeffs.shape


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("shape", [(16, 16), (15, 17), (32, 16)])
def test_fswaverecn_roundtrip(wavelet, shape):
    x = jnp.array(np.random.RandomState(0).randn(*shape))
    result = fswavedecn(x, wavelet)
    rec = fswaverecn(result)
    np.testing.assert_allclose(
        np.array(rec[tuple(slice(s) for s in shape)]), np.array(x), atol=ATOL_RT
    )


@pytest.mark.parametrize("wavelet", WAVELETS)
def test_fswavedecn_coeffs_values_match_pywt(wavelet):
    """Verify packed coefficient values match pywt, not just shapes."""
    x_np = np.random.RandomState(0).randn(16, 16)
    r_jax = fswavedecn(jnp.array(x_np), wavelet, levels=2)
    r_pywt = pywt.fswavedecn(x_np, wavelet, levels=2)
    np.testing.assert_allclose(np.array(r_jax.coeffs), r_pywt.coeffs, atol=ATOL)


def test_fswt_jit():
    x = jnp.array(np.random.RandomState(0).randn(16, 16))
    f = jax.jit(lambda x: fswaverecn(fswavedecn(x, "haar", levels=2)))
    np.testing.assert_allclose(np.array(f(x)), np.array(x), atol=ATOL_RT)
