"""Tests for continuous wavelet transform."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pywt
import pytest

from jaxwt._cwt import cwt, wavefun, integrate_wavelet, central_frequency, as_wavelet

REAL_WAVELETS = ['morl', 'mexh', 'gaus1', 'gaus4', 'gaus8']
COMPLEX_WAVELETS = ['cgau1', 'cgau4', 'cgau8', 'cmor1.5-1.0', 'shan1.5-1.0', 'fbsp2-1.5-1.0']
ATOL = 1e-10


# --- Wavelet functions ---

@pytest.mark.parametrize('wavelet', REAL_WAVELETS + COMPLEX_WAVELETS)
def test_wavefun_matches_pywt(wavelet):
    psi_jax, x_jax = wavefun(wavelet, precision=8)
    psi_pywt, x_pywt = pywt.ContinuousWavelet(wavelet).wavefun(level=8)
    np.testing.assert_allclose(np.array(x_jax), x_pywt, atol=ATOL)
    np.testing.assert_allclose(np.array(psi_jax), psi_pywt, atol=ATOL)


# --- Integrated wavelets ---

@pytest.mark.parametrize('wavelet', REAL_WAVELETS + COMPLEX_WAVELETS)
def test_integrate_wavelet_matches_pywt(wavelet):
    int_psi_jax, x_jax = integrate_wavelet(wavelet, precision=8)
    int_psi_pywt, x_pywt = pywt.integrate_wavelet(wavelet, precision=8)
    np.testing.assert_allclose(np.array(int_psi_jax), int_psi_pywt, atol=ATOL)


# --- Central frequency ---

@pytest.mark.parametrize('wavelet', REAL_WAVELETS + COMPLEX_WAVELETS)
def test_central_frequency_matches_pywt(wavelet):
    cf_jax = central_frequency(wavelet, precision=12)
    cf_pywt = pywt.central_frequency(wavelet, precision=12)
    np.testing.assert_allclose(float(cf_jax), cf_pywt, rtol=1e-6)


# --- CWT ---

@pytest.mark.parametrize('wavelet', ['morl', 'mexh', 'gaus1'])
def test_cwt_real_matches_pywt(wavelet):
    x = np.random.RandomState(0).randn(128)
    scales = np.array([1., 2., 4., 8.])
    coef_jax, freq_jax = cwt(jnp.array(x), scales, wavelet)
    coef_pywt, freq_pywt = pywt.cwt(x, scales, wavelet)
    np.testing.assert_allclose(np.array(coef_jax), coef_pywt, atol=ATOL)
    np.testing.assert_allclose(np.array(freq_jax), freq_pywt, rtol=1e-6)


@pytest.mark.parametrize('wavelet', ['cmor1.5-1.0', 'cgau1'])
def test_cwt_complex_matches_pywt(wavelet):
    x = np.random.RandomState(0).randn(128)
    scales = np.array([1., 2., 4., 8.])
    coef_jax, freq_jax = cwt(jnp.array(x), scales, wavelet)
    coef_pywt, freq_pywt = pywt.cwt(x, scales, wavelet)
    np.testing.assert_allclose(np.array(coef_jax), coef_pywt, atol=ATOL)


def test_cwt_grad():
    x = jnp.array(np.random.RandomState(0).randn(64))
    scales = jnp.array([1., 2., 4.])
    g = jax.grad(lambda x: jnp.sum(jnp.abs(cwt(x, scales, 'morl')[0])))(x)
    assert g.shape == x.shape
