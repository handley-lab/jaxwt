"""Tests for thresholding functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pywt

from jaxwavelets._thresholding import (
    firm_threshold,
    garrote_threshold,
    hard_threshold,
    soft_threshold,
)

ATOL = 1e-12


def test_soft_threshold_matches_pywt():
    x_np = np.linspace(-4, 4, 17)
    r_jax = soft_threshold(jnp.array(x_np), 2.0)
    r_pywt = pywt.threshold(x_np, 2.0, "soft")
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


def test_hard_threshold_matches_pywt():
    x_np = np.linspace(-4, 4, 17)
    r_jax = hard_threshold(jnp.array(x_np), 2.0)
    r_pywt = pywt.threshold(x_np, 2.0, "hard")
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


def test_garrote_threshold_matches_pywt():
    x_np = np.linspace(-4, 4, 17)
    r_jax = garrote_threshold(jnp.array(x_np), 2.0)
    r_pywt = pywt.threshold(x_np, 2.0, "garrote")
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


def test_soft_threshold_substitute():
    x_np = np.linspace(-4, 4, 17)
    r_jax = soft_threshold(jnp.array(x_np), 2.0, substitute=-1.0)
    r_pywt = pywt.threshold(x_np, 2.0, "soft", substitute=-1.0)
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


def test_firm_threshold_matches_pywt():
    x_np = np.linspace(-4, 4, 17)
    r_jax = firm_threshold(jnp.array(x_np), 1.0, 3.0)
    r_pywt = pywt.threshold_firm(x_np, 1.0, 3.0)
    np.testing.assert_allclose(np.array(r_jax), r_pywt, atol=ATOL)


def test_soft_threshold_jit():
    x = jnp.linspace(-4, 4, 17)
    f = jax.jit(lambda x: soft_threshold(x, 2.0))
    np.testing.assert_allclose(np.array(f(x)), np.array(soft_threshold(x, 2.0)))


def test_soft_threshold_grad():
    x = jnp.linspace(-4, 4, 17)
    g = jax.grad(lambda x: jnp.sum(soft_threshold(x, 2.0)))(x)
    assert g.shape == x.shape


def test_firm_threshold_edge_cases():
    x = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    result = firm_threshold(x, 1.0, 3.0)
    np.testing.assert_allclose(np.array(result[:2]), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(float(result[2]), 0.0, atol=1e-12)
    np.testing.assert_allclose(float(result[-1]), 4.0, atol=1e-12)
