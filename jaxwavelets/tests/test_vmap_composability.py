"""Test JAX composability: vmap, grad, jit, and combinations."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxwavelets as wt


@pytest.fixture
def batch_1d():
    return jnp.stack(
        [jnp.array(np.random.RandomState(i).randn(32)) for i in range(4)]
    )


@pytest.fixture
def batch_2d():
    return jnp.stack(
        [jnp.array(np.random.RandomState(i).randn(16, 16)) for i in range(4)]
    )


ATOL_RT = 1e-11


def test_vmap_dwt_idwt(batch_1d):
    cA, cD = jax.vmap(partial(wt.dwt, wavelet="db4"))(batch_1d)
    rec = jax.vmap(partial(wt.idwt, wavelet="db4"))(cA, cD)
    np.testing.assert_allclose(
        np.array(rec[:, :32]), np.array(batch_1d), atol=ATOL_RT
    )


def test_vmap_dwt_periodization(batch_1d):
    f = partial(wt.dwt, wavelet="db4", mode="periodization")
    g = partial(wt.idwt, wavelet="db4", mode="periodization")
    cA, cD = jax.vmap(f)(batch_1d)
    rec = jax.vmap(g)(cA, cD)
    np.testing.assert_allclose(
        np.array(rec[:, :32]), np.array(batch_1d), atol=ATOL_RT
    )


def test_vmap_wavedecn_2d(batch_2d):
    f = partial(wt.wavedecn, wavelet="db4", level=2)
    g = partial(wt.waverecn, wavelet="db4")
    rec = jax.vmap(lambda x: g(f(x)))(batch_2d)
    np.testing.assert_allclose(
        np.array(rec), np.array(batch_2d), atol=ATOL_RT
    )


def test_grad_through_vmap(batch_2d):
    def loss(batch):
        f = partial(wt.wavedecn, wavelet="db4", level=2)
        g = partial(wt.waverecn, wavelet="db4")
        return jnp.sum(jax.vmap(lambda x: g(f(x)))(batch))

    grad = jax.grad(loss)(batch_2d)
    np.testing.assert_allclose(
        np.array(grad), np.ones_like(grad), atol=ATOL_RT
    )


def test_jit_vmap(batch_2d):
    f = jax.jit(jax.vmap(
        lambda x: wt.waverecn(
            wt.wavedecn(x, "db4", level=2), "db4"
        )
    ))
    np.testing.assert_allclose(
        np.array(f(batch_2d)), np.array(batch_2d), atol=ATOL_RT
    )


def test_vmap_grad(batch_2d):
    f = jax.vmap(jax.grad(
        lambda x: jnp.sum(wt.waverecn(
            wt.wavedecn(x, "db4", level=2), "db4"
        ))
    ))
    grad = f(batch_2d)
    np.testing.assert_allclose(
        np.array(grad), np.ones_like(grad), atol=ATOL_RT
    )


def test_tree_map():
    x = jnp.array(np.random.RandomState(0).randn(16, 16))
    coeffs = wt.wavedecn(x, "db4", level=2)
    doubled = jax.tree_util.tree_map(lambda a: 2 * a, coeffs)
    ratio = float(jnp.mean(doubled.approx / coeffs.approx))
    np.testing.assert_allclose(ratio, 2.0)
