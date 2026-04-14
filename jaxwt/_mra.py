"""Multiresolution analysis."""
import jax.numpy as jnp
from jaxwt._multidim import (wavedecn, waverecn, WaveletCoeffs,
                              wavedec2, waverec2)


def _mra_nd(data, wavelet, mode, level, axes):
    """Core MRA: decompose, then reconstruct each component with others zeroed."""
    wc = wavedecn(data, wavelet, mode, level, axes)
    nc = 1 + len(wc.details)  # approx + detail levels
    result = []
    for j in range(nc):
        if j == 0:
            approx = wc.approx
            details = tuple({k: jnp.zeros_like(v) for k, v in d.items()} for d in wc.details)
        else:
            approx = jnp.zeros_like(wc.approx)
            details = tuple(
                (d if i == j - 1 else {k: jnp.zeros_like(v) for k, v in d.items()})
                for i, d in enumerate(wc.details)
            )
        tmp = WaveletCoeffs(approx, details, wc.shapes, wc.axes)
        rec = waverecn(tmp, wavelet, mode)
        result.append(rec[tuple(slice(s) for s in data.shape)])
    return result


def mra(data, wavelet, mode='symmetric', level=None, axis=-1):
    """1D multiresolution analysis. Returns list of arrays summing to data."""
    return _mra_nd(data, wavelet, mode, level, axes=(axis,))


def imra(mra_coeffs):
    """Inverse 1D MRA: sum of components."""
    return sum(mra_coeffs)


def mra2(data, wavelet, mode='symmetric', level=None, axes=(-2, -1)):
    """2D multiresolution analysis."""
    return _mra_nd(data, wavelet, mode, level, axes)


def imra2(mra_coeffs):
    """Inverse 2D MRA: sum of components."""
    return sum(mra_coeffs)


def mran(data, wavelet, mode='symmetric', level=None, axes=None):
    """nD multiresolution analysis."""
    return _mra_nd(data, wavelet, mode, level, axes)


def imran(mra_coeffs):
    """Inverse nD MRA: sum of components."""
    return sum(mra_coeffs)
