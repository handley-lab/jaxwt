"""Tests for multiresolution analysis."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pytest

from jaxwt._mra import mra, imra, mra2, imra2, mran, imran

ATOL = 1e-11


@pytest.mark.parametrize('wavelet', ['haar', 'db2', 'db4'])
@pytest.mark.parametrize('N', [16, 32, 64])
def test_mra_additivity(wavelet, N):
    x = jnp.array(np.random.RandomState(0).randn(N))
    components = mra(x, wavelet, level=3)
    np.testing.assert_allclose(np.array(imra(components)), np.array(x), atol=ATOL)


@pytest.mark.parametrize('wavelet', ['haar', 'db2'])
@pytest.mark.parametrize('N', [16, 32])
def test_mra_component_count(wavelet, N):
    x = jnp.array(np.random.RandomState(0).randn(N))
    components = mra(x, wavelet, level=3)
    assert len(components) == 4  # 1 approx + 3 detail levels


@pytest.mark.parametrize('wavelet', ['haar', 'db2'])
@pytest.mark.parametrize('shape', [(16, 16), (32, 16)])
def test_mra2_additivity(wavelet, shape):
    x = jnp.array(np.random.RandomState(0).randn(*shape))
    components = mra2(x, wavelet, level=2)
    np.testing.assert_allclose(np.array(imra2(components)), np.array(x), atol=ATOL)


@pytest.mark.parametrize('wavelet', ['haar', 'db2'])
@pytest.mark.parametrize('shape', [(16, 16), (8, 8, 8)])
def test_mran_additivity(wavelet, shape):
    x = jnp.array(np.random.RandomState(0).randn(*shape))
    components = mran(x, wavelet, level=2)
    np.testing.assert_allclose(np.array(imran(components)), np.array(x), atol=ATOL)


def test_mra_component_shapes():
    x = jnp.array(np.random.RandomState(0).randn(32))
    components = mra(x, 'haar', level=3)
    for c in components:
        assert c.shape == x.shape
