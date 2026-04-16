"""Tests for filter utilities and partial DWT functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt

import jaxwavelets as wt

WAVELETS = ["haar", "db2", "db4", "db8", "sym4", "coif2"]
ATOL = 1e-14
ATOL_RT = 1e-11


# --- qmf ---


@pytest.mark.parametrize("wavelet", WAVELETS)
def test_qmf(wavelet):
    w = pywt.Wavelet(wavelet)
    result = wt.qmf(jnp.array(w.rec_lo))
    expected = pywt.qmf(w.rec_lo)
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)


# --- orthogonal_filter_bank ---


@pytest.mark.parametrize("wavelet", WAVELETS)
def test_orthogonal_filter_bank(wavelet):
    w = pywt.Wavelet(wavelet)
    bank_jax = wt.orthogonal_filter_bank(jnp.array(w.rec_lo))
    bank_pywt = pywt.orthogonal_filter_bank(w.rec_lo)
    for j, p in zip(bank_jax, bank_pywt, strict=False):
        np.testing.assert_allclose(np.array(j), p, atol=ATOL)


# --- downcoef ---


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("N", [8, 16, 32])
@pytest.mark.parametrize("part", ["a", "d"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_downcoef(wavelet, N, part, level):
    x_np = np.random.RandomState(0).randn(N)
    result = wt.downcoef(part, jnp.array(x_np), wavelet, level=level)
    expected = pywt.downcoef(part, x_np, wavelet, level=level)
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)


# --- upcoef ---


@pytest.mark.parametrize("wavelet", WAVELETS)
@pytest.mark.parametrize("part", ["a", "d"])
@pytest.mark.parametrize("level", [1, 2])
def test_upcoef(wavelet, part, level):
    x_np = np.random.RandomState(0).randn(16)
    cA, cD = pywt.dwt(x_np, wavelet)
    coeffs = cA if part == "a" else cD
    result = wt.upcoef(part, jnp.array(coeffs), wavelet, level=level)
    expected = pywt.upcoef(part, coeffs, wavelet, level=level)
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)


# --- JAX composability ---


def test_qmf_jit():
    f = jnp.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(jax.jit(wt.qmf)(f), wt.qmf(f))


def test_downcoef_jit():
    x = jnp.array(np.random.RandomState(0).randn(32))
    f = jax.jit(lambda x: wt.downcoef("a", x, "db4", level=2))
    np.testing.assert_allclose(
        np.array(f(x)), np.array(wt.downcoef("a", x, "db4", level=2))
    )


def test_downcoef_grad():
    x = jnp.array(np.random.RandomState(0).randn(32))
    g = jax.grad(lambda x: jnp.sum(wt.downcoef("a", x, "db4", level=2)))(x)
    assert g.shape == x.shape


def test_upcoef_jit():
    c = jnp.array(np.random.RandomState(0).randn(10))
    f = jax.jit(lambda c: wt.upcoef("a", c, "db4", level=2))
    np.testing.assert_allclose(
        np.array(f(c)), np.array(wt.upcoef("a", c, "db4", level=2))
    )


# --- upcoef take ---

# --- 2D wrappers ---


@pytest.mark.parametrize("wavelet", ["haar", "db4", "sym4"])
@pytest.mark.parametrize("shape", [(16, 16), (15, 17)])
def test_dwt2_matches_pywt(wavelet, shape):
    x_np = np.random.RandomState(0).randn(*shape)
    cA_j, (cH_j, cV_j, cD_j) = wt.dwt2(jnp.array(x_np), wavelet)
    cA_p, (cH_p, cV_p, cD_p) = pywt.dwt2(x_np, wavelet)
    np.testing.assert_allclose(np.array(cA_j), cA_p, atol=ATOL)
    np.testing.assert_allclose(np.array(cH_j), cH_p, atol=ATOL)
    np.testing.assert_allclose(np.array(cV_j), cV_p, atol=ATOL)
    np.testing.assert_allclose(np.array(cD_j), cD_p, atol=ATOL)


@pytest.mark.parametrize("wavelet", ["haar", "db4", "sym4"])
@pytest.mark.parametrize("shape", [(16, 16), (15, 17)])
def test_idwt2_roundtrip(wavelet, shape):
    x = jnp.array(np.random.RandomState(0).randn(*shape))
    coeffs = wt.dwt2(x, wavelet)
    rec = wt.idwt2(coeffs, wavelet)
    np.testing.assert_allclose(
        np.array(rec[: shape[0], : shape[1]]), np.array(x), atol=ATOL_RT
    )


@pytest.mark.parametrize("wavelet", ["haar", "db4", "sym4"])
@pytest.mark.parametrize("shape", [(16, 16), (15, 17)])
@pytest.mark.parametrize("level", [1, 2])
def test_wavedec2_matches_pywt(wavelet, shape, level):
    x_np = np.random.RandomState(0).randn(*shape)
    coeffs_j = wt.wavedec2(jnp.array(x_np), wavelet, level=level)
    coeffs_p = pywt.wavedec2(x_np, wavelet, level=level)
    np.testing.assert_allclose(np.array(coeffs_j[0]), coeffs_p[0], atol=ATOL)
    for j_detail, p_detail in zip(coeffs_j[1:], coeffs_p[1:], strict=False):
        cH_j, cV_j, cD_j = j_detail
        cH_p, cV_p, cD_p = p_detail
        np.testing.assert_allclose(np.array(cH_j), cH_p, atol=ATOL)
        np.testing.assert_allclose(np.array(cV_j), cV_p, atol=ATOL)
        np.testing.assert_allclose(np.array(cD_j), cD_p, atol=ATOL)


@pytest.mark.parametrize("wavelet", ["haar", "db4", "sym4"])
@pytest.mark.parametrize("shape", [(16, 16), (15, 17)])
def test_waverec2_roundtrip(wavelet, shape):
    x = jnp.array(np.random.RandomState(0).randn(*shape))
    coeffs = wt.wavedec2(x, wavelet)
    rec = wt.waverec2(coeffs, wavelet)
    # waverec2 may produce larger output for odd shapes (matches pywt behavior)
    np.testing.assert_allclose(
        np.array(rec[tuple(slice(s) for s in shape)]), np.array(x), atol=ATOL_RT
    )


# --- upcoef take ---


@pytest.mark.parametrize("wavelet", WAVELETS)
def test_upcoef_take(wavelet):
    x_np = np.random.RandomState(0).randn(16)
    cA, cD = pywt.dwt(x_np, wavelet)
    result = wt.upcoef("a", jnp.array(cA), wavelet, take=len(x_np))
    expected = pywt.upcoef("a", cA, wavelet, take=len(x_np))
    np.testing.assert_allclose(np.array(result), expected, atol=ATOL)
