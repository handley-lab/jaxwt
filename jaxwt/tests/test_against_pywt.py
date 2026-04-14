"""Tests verifying jaxwt matches pywt to machine precision."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pywt
import pytest
from functools import partial

import jaxwt


WAVELETS = ['haar', 'db2', 'db4', 'db8', 'sym4', 'sym8', 'coif2', 'coif4']
ATOL = 1e-11


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [4, 7, 8, 15, 16, 31, 32, 64])
def test_dwt_matches_pywt(wavelet, N):
    np.random.seed(0)
    x_np = np.random.randn(N)
    x_jnp = jnp.array(x_np)

    cA_jax, cD_jax = jaxwt.dwt(x_jnp, wavelet)
    cA_pywt, cD_pywt = pywt.dwt(x_np, wavelet)

    np.testing.assert_allclose(np.array(cA_jax), cA_pywt, atol=ATOL)
    np.testing.assert_allclose(np.array(cD_jax), cD_pywt, atol=ATOL)


@pytest.mark.parametrize('wavelet', WAVELETS)
@pytest.mark.parametrize('N', [4, 7, 8, 15, 16, 32])
def test_perfect_reconstruction_1d(wavelet, N):
    np.random.seed(0)
    x = jnp.array(np.random.randn(N))
    cA, cD = jaxwt.dwt(x, wavelet)
    rec = jaxwt.idwt(cA, cD, wavelet)
    np.testing.assert_allclose(np.array(rec[:N]), np.array(x), atol=ATOL)


@pytest.mark.parametrize('wavelet', ['haar', 'db4', 'sym4'])
@pytest.mark.parametrize('shape', [(16, 16), (15, 17), (8, 8, 8)])
@pytest.mark.parametrize('level', [1, 2])
def test_wavedecn_matches_pywt(wavelet, shape, level):
    np.random.seed(0)
    x_np = np.random.randn(*shape)
    x_jnp = jnp.array(x_np)

    coeffs_jax = jaxwt.wavedecn(x_jnp, wavelet, level=level)
    coeffs_pywt = pywt.wavedecn(x_np, wavelet, level=level)

    np.testing.assert_allclose(np.array(coeffs_jax.approx), coeffs_pywt[0], atol=ATOL)
    for jax_d, pywt_d in zip(coeffs_jax.details, coeffs_pywt[1:]):
        for key in jax_d:
            np.testing.assert_allclose(np.array(jax_d[key]), pywt_d[key], atol=ATOL)


@pytest.mark.parametrize('wavelet', ['haar', 'db4', 'sym4'])
@pytest.mark.parametrize('shape', [(16, 16), (15, 17), (8, 8, 8)])
def test_perfect_reconstruction_nd(wavelet, shape):
    np.random.seed(0)
    x = jnp.array(np.random.randn(*shape))
    coeffs = jaxwt.wavedecn(x, wavelet)
    rec = jaxwt.waverecn(coeffs, wavelet)
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL)


def test_grad():
    x = jnp.array(np.random.randn(16, 16))
    g = jax.grad(lambda x: jnp.sum(jaxwt.waverecn(jaxwt.wavedecn(x, 'db4'), 'db4')))(x)
    np.testing.assert_allclose(np.array(g), np.ones_like(g), atol=ATOL)


def test_vmap():
    batch = jnp.stack([jnp.array(np.random.randn(8, 8)) for _ in range(4)])
    f = partial(jaxwt.wavedecn, wavelet='haar', level=1)
    batch_coeffs = jax.vmap(f)(batch)
    assert batch_coeffs.approx.shape == (4, 4, 4)


def test_jit():
    x = jnp.array(np.random.randn(16, 16))
    f = jax.jit(jaxwt.wavedecn, static_argnames=['wavelet', 'mode', 'level'])
    coeffs = f(x, wavelet='db4', level=2)
    rec = jaxwt.waverecn(coeffs, 'db4')
    np.testing.assert_allclose(np.array(rec), np.array(x), atol=ATOL)
