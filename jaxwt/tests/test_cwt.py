"""Tests for continuous wavelet transform."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pywt
import pytest

from jaxwt._cwt import cwt, wavefun, integrate_wavelet, central_frequency

REAL_WAVELETS = ['morl', 'mexh'] + [f'gaus{i}' for i in range(1, 9)]
COMPLEX_WAVELETS = [f'cgau{i}' for i in range(1, 9)] + ['cmor1.5-1.0', 'shan1.5-1.0', 'fbsp2-1.5-1.0']
ALL_WAVELETS = REAL_WAVELETS + COMPLEX_WAVELETS
ATOL = 1e-10


@pytest.mark.parametrize('wavelet', ALL_WAVELETS)
def test_wavefun_matches_pywt(wavelet):
    psi_jax, x_jax = wavefun(wavelet, precision=8)
    psi_pywt, x_pywt = pywt.ContinuousWavelet(wavelet).wavefun(level=8)
    np.testing.assert_allclose(np.array(x_jax), x_pywt, atol=ATOL)
    np.testing.assert_allclose(np.array(psi_jax), psi_pywt, atol=ATOL)


@pytest.mark.parametrize('wavelet', ALL_WAVELETS)
def test_integrate_wavelet_matches_pywt(wavelet):
    int_psi_jax, _ = integrate_wavelet(wavelet, precision=8)
    int_psi_pywt, _ = pywt.integrate_wavelet(wavelet, precision=8)
    np.testing.assert_allclose(np.array(int_psi_jax), int_psi_pywt, atol=ATOL)


@pytest.mark.parametrize('wavelet', ALL_WAVELETS)
def test_central_frequency_matches_pywt(wavelet):
    cf_jax = central_frequency(wavelet, precision=12)
    cf_pywt = pywt.central_frequency(wavelet, precision=12)
    np.testing.assert_allclose(float(cf_jax), cf_pywt, rtol=1e-6)


@pytest.mark.parametrize('wavelet', ['morl', 'mexh', 'gaus1', 'gaus4', 'gaus8'])
def test_cwt_real_matches_pywt(wavelet):
    x = np.random.RandomState(0).randn(128)
    scales = np.array([1., 2., 4., 8.])
    coef_jax, freq_jax = cwt(jnp.array(x), scales, wavelet)
    coef_pywt, freq_pywt = pywt.cwt(x, scales, wavelet)
    np.testing.assert_allclose(np.array(coef_jax), coef_pywt, atol=ATOL)
    np.testing.assert_allclose(np.array(freq_jax), freq_pywt, rtol=1e-6)


@pytest.mark.parametrize('wavelet', ['cmor1.5-1.0', 'cgau1', 'cgau4', 'shan1.5-1.0', 'fbsp2-1.5-1.0'])
def test_cwt_complex_matches_pywt(wavelet):
    x = np.random.RandomState(0).randn(128)
    scales = np.array([1., 2., 4., 8.])
    coef_jax, _ = cwt(jnp.array(x), scales, wavelet)
    coef_pywt, _ = pywt.cwt(x, scales, wavelet)
    np.testing.assert_allclose(np.array(coef_jax), coef_pywt, atol=ATOL)


def test_cwt_fft_method():
    x = np.random.RandomState(0).randn(128)
    scales = np.array([1., 2., 4.])
    coef_conv, _ = cwt(jnp.array(x), scales, 'morl', method='conv')
    coef_fft, _ = cwt(jnp.array(x), scales, 'morl', method='fft')
    np.testing.assert_allclose(np.array(coef_fft), np.array(coef_conv), atol=1e-10)


def test_cwt_grad():
    x = jnp.array(np.random.RandomState(0).randn(64))
    scales = jnp.array([1., 2., 4.])
    g = jax.grad(lambda x: jnp.sum(jnp.abs(cwt(x, scales, 'morl')[0])))(x)
    assert g.shape == x.shape
