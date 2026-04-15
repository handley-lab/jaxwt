"""Tests for stationary wavelet transform."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pywt
import pytest

from jaxwt._swt import swt, iswt, swtn, iswtn, swt2, iswt2

WAVELETS = ['haar', 'db2', 'db4', 'sym4']
ATOL = 1e-11


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [8, 16, 32, 64])
@pytest.mark.parametrize('level', [1, 2, 3])
def test_swt_matches_pywt(wavelet, N, level):
    if 2**level > N:
        pytest.skip("level too high for signal length")
    x_np = np.random.RandomState(0).randn(N)
    coeffs_jax = swt(jnp.array(x_np), wavelet, level=level)
    coeffs_pywt = pywt.swt(x_np, wavelet, level=level)
    for (cA_j, cD_j), (cA_p, cD_p) in zip(coeffs_jax, coeffs_pywt):
        np.testing.assert_allclose(np.array(cA_j), cA_p, atol=ATOL)
        np.testing.assert_allclose(np.array(cD_j), cD_p, atol=ATOL)


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [8, 16, 32])
def test_iswt_roundtrip(wavelet, N):
    x = jnp.array(np.random.RandomState(0).randn(N))
    level = min(3, int(np.log2(N)))
    coeffs = swt(x, wavelet, level=level)
    rec = iswt(coeffs, wavelet)
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL)


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [8, 16, 32])
def test_iswt_matches_pywt(wavelet, N):
    x_np = np.random.RandomState(0).randn(N)
    level = min(3, int(np.log2(N)))
    coeffs_pywt = pywt.swt(x_np, wavelet, level=level)
    rec_pywt = pywt.iswt(coeffs_pywt, wavelet)
    rec_jax = iswt([(jnp.array(a), jnp.array(d)) for a, d in coeffs_pywt], wavelet)
    np.testing.assert_allclose(np.array(rec_jax), rec_pywt, atol=ATOL)


def test_swt_trim_approx(wavelet='db2', N=16, level=2):
    x_np = np.random.RandomState(0).randn(N)
    coeffs_jax = swt(jnp.array(x_np), wavelet, level=level, trim_approx=True)
    coeffs_pywt = pywt.swt(x_np, wavelet, level=level, trim_approx=True)
    for j, p in zip(coeffs_jax, coeffs_pywt):
        np.testing.assert_allclose(np.array(j), p, atol=ATOL)


# --- nD SWT ---

@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('shape', [(8, 8), (16, 16)])
def test_swtn_matches_pywt(wavelet, shape):
    x_np = np.random.RandomState(0).randn(*shape)
    level = 2
    cj = swtn(jnp.array(x_np), wavelet, level=level)
    cp = pywt.swtn(x_np, wavelet, level=level)
    for jd, pd in zip(cj, cp):
        for key in jd:
            np.testing.assert_allclose(np.array(jd[key]), pd[key], atol=ATOL)


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('shape', [(8, 8), (16, 16)])
def test_iswtn_roundtrip(wavelet, shape):
    x = jnp.array(np.random.RandomState(0).randn(*shape))
    level = 2
    coeffs = swtn(x, wavelet, level=level)
    rec = iswtn(coeffs, wavelet)
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL)


@pytest.mark.parametrize('wavelet', ['haar', 'db2'])
def test_swt2_matches_pywt_swtn(wavelet):
    """swt2 is an alias for swtn with 2 axes — compare against pywt.swtn."""
    x_np = np.random.RandomState(0).randn(16, 16)
    cj = swt2(jnp.array(x_np), wavelet, level=2)
    cp = pywt.swtn(x_np, wavelet, level=2)
    for jd, pd in zip(cj, cp):
        for key in jd:
            np.testing.assert_allclose(np.array(jd[key]), pd[key], atol=ATOL)


@pytest.mark.parametrize('wavelet', ['haar', 'db2'])
def test_iswt2_roundtrip(wavelet):
    x = jnp.array(np.random.RandomState(0).randn(16, 16))
    coeffs = swt2(x, wavelet, level=2)
    rec = iswt2(coeffs, wavelet)
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL)


# --- 3D SWT ---

def test_swtn_3d_matches_pywt():
    x_np = np.random.RandomState(0).randn(8, 8, 8)
    cj = swtn(jnp.array(x_np), 'haar', level=1)
    cp = pywt.swtn(x_np, 'haar', level=1)
    for jd, pd in zip(cj, cp):
        for key in jd:
            np.testing.assert_allclose(np.array(jd[key]), pd[key], atol=ATOL)


def test_iswtn_3d_roundtrip():
    x = jnp.array(np.random.RandomState(0).randn(8, 8, 8))
    coeffs = swtn(x, 'haar', level=1)
    rec = iswtn(coeffs, 'haar')
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL)


def test_swtn_subset_axes():
    """Transform only 2 axes of a 3D array."""
    x_np = np.random.RandomState(0).randn(8, 8, 8)
    axes = (0, 2)
    cj = swtn(jnp.array(x_np), 'haar', level=1, axes=axes)
    cp = pywt.swtn(x_np, 'haar', level=1, axes=axes)
    for jd, pd in zip(cj, cp):
        for key in jd:
            np.testing.assert_allclose(np.array(jd[key]), pd[key], atol=ATOL)
