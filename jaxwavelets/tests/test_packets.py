"""Tests for wavelet packets."""

import jax.numpy as jnp
import numpy as np
import pytest
import pywt

from jaxwavelets._packets import (
    wp_decompose,
    wp_decompose_nd,
    wp_reconstruct,
    wp_reconstruct_nd,
)

WAVELETS = ["haar", "db2", "db4"]
ATOL = 1e-14
ATOL_RT = 1e-11


# --- 1D leaf values match pywt ---


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("N", [16, 32, 64])
def test_wp_leaves_match_pywt(wavelet, N):
    x_np = np.random.RandomState(0).randn(N)
    leaves, _ = wp_decompose(jnp.array(x_np), wavelet, maxlevel=2)
    wp = pywt.WaveletPacket(x_np, wavelet, maxlevel=2)
    for path in leaves:
        np.testing.assert_allclose(np.array(leaves[path]), wp[path].data, atol=ATOL)


# --- 1D roundtrip ---


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("N", [16, 32])
def test_wp_roundtrip(wavelet, N):
    x = jnp.array(np.random.RandomState(0).randn(N))
    leaves, shapes = wp_decompose(x, wavelet, maxlevel=2)
    rec = wp_reconstruct(leaves, wavelet)
    np.testing.assert_allclose(np.array(rec[:N]), np.array(x), atol=ATOL_RT)


# --- nD leaf values ---


@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_wp_nd_leaves_match_pywt(wavelet):
    x_np = np.random.RandomState(0).randn(16, 16)
    mode = "symmetric"
    leaves, _ = wp_decompose_nd(jnp.array(x_np), wavelet, mode=mode, maxlevel=1)
    wp = pywt.WaveletPacketND(x_np, wavelet, mode=mode, maxlevel=1)
    for path in leaves:
        np.testing.assert_allclose(np.array(leaves[path]), wp[path].data, atol=ATOL)


# --- nD roundtrip ---


@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_wp_nd_roundtrip(wavelet):
    x = jnp.array(np.random.RandomState(0).randn(16, 16))
    axes = (0, 1)
    leaves, shapes = wp_decompose_nd(x, wavelet, maxlevel=1, axes=axes)
    rec = wp_reconstruct_nd(leaves, wavelet, axes=axes)
    np.testing.assert_allclose(np.array(rec[:16, :16]), np.array(x), atol=ATOL_RT)


# --- Leaf count ---


def test_wp_leaf_count():
    x = jnp.array(np.random.RandomState(0).randn(32))
    leaves, _ = wp_decompose(x, "haar", maxlevel=3)
    assert len(leaves) == 2**3


def test_wp_nd_leaf_count():
    x = jnp.array(np.random.RandomState(0).randn(16, 16))
    leaves, _ = wp_decompose_nd(x, "haar", maxlevel=2)
    assert len(leaves) == 4**2  # (2^ndim)^maxlevel
