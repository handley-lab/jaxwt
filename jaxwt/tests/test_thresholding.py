"""Tests for thresholding functions."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import pywt
import pytest

from jaxwt._thresholding import threshold, threshold_firm

ATOL = 1e-12


@pytest.mark.parametrize('mode', ['soft', 'hard', 'greater', 'less', 'garrote'])
def test_threshold_matches_pywt(mode):
    x_np = np.linspace(-4, 4, 17)
    r_jax = threshold(jnp.array(x_np), 2.0, mode)
    r_pywt = pywt.threshold(x_np, 2.0, mode)
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


@pytest.mark.parametrize('mode', ['soft', 'hard', 'garrote'])
def test_threshold_substitute(mode):
    x_np = np.linspace(-4, 4, 17)
    r_jax = threshold(jnp.array(x_np), 2.0, mode, substitute=-1.0)
    r_pywt = pywt.threshold(x_np, 2.0, mode, substitute=-1.0)
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


def test_threshold_firm_matches_pywt():
    x_np = np.linspace(-4, 4, 17)
    r_jax = threshold_firm(jnp.array(x_np), 1.0, 3.0)
    r_pywt = pywt.threshold_firm(x_np, 1.0, 3.0)
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


def test_threshold_jit():
    x = jnp.linspace(-4, 4, 17)
    f = jax.jit(lambda x: threshold(x, 2.0, 'soft'))
    np.testing.assert_allclose(np.array(f(x)), np.array(threshold(x, 2.0, 'soft')))


def test_threshold_grad():
    x = jnp.linspace(-4, 4, 17)
    g = jax.grad(lambda x: jnp.sum(threshold(x, 2.0, 'soft')))(x)
    assert g.shape == x.shape
