"""Multiresolution analysis."""

import jax.numpy as jnp

from jaxwavelets._multidim import WaveletCoeffs, wavedecn, waverecn


def _mra_nd(data, wavelet, mode, level, axes):
    """Core MRA: decompose, then reconstruct each component with others zeroed."""
    wc = wavedecn(data, wavelet, mode, level, axes)
    nc = 1 + len(wc.details)  # approx + detail levels
    result = []
    for j in range(nc):
        if j == 0:
            approx = wc.approx
            details = tuple(
                {k: jnp.zeros_like(v) for k, v in d.items()} for d in wc.details
            )
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


def mra(data, wavelet, mode="symmetric", level=None, axis=-1):
    """1D multiresolution analysis.

    Parameters
    ----------
    data : array
        Input array.
    wavelet : str or Wavelet
        Wavelet to use.
    mode : str
        Signal extension mode. Default 'symmetric'.
    level : int, optional
        Decomposition level. Default is the maximum useful level.
    axis : int
        Axis along which to compute the MRA. Default -1.

    Returns
    -------
    list of array
        Components whose sum reconstructs the input data.
    """
    return _mra_nd(data, wavelet, mode, level, axes=(axis,))


def imra(mra_coeffs):
    """Inverse 1D multiresolution analysis.

    Parameters
    ----------
    mra_coeffs : list of array
        MRA components from :func:`mra`.

    Returns
    -------
    array
        Sum of all components.
    """
    return sum(mra_coeffs)


def mra2(data, wavelet, mode="symmetric", level=None, axes=(-2, -1)):
    """2D multiresolution analysis.

    Parameters
    ----------
    data : array
        2D input array.
    wavelet : str or Wavelet
        Wavelet to use.
    mode : str
        Signal extension mode. Default 'symmetric'.
    level : int, optional
        Decomposition level. Default is the maximum useful level.
    axes : tuple of int
        Axes for the 2D transform. Default ``(-2, -1)``.

    Returns
    -------
    list of array
        Components whose sum reconstructs the input data.
    """
    return _mra_nd(data, wavelet, mode, level, axes)


def imra2(mra_coeffs):
    """Inverse 2D multiresolution analysis.

    Parameters
    ----------
    mra_coeffs : list of array
        MRA components from :func:`mra2`.

    Returns
    -------
    array
        Sum of all components.
    """
    return sum(mra_coeffs)


def mran(data, wavelet, mode="symmetric", level=None, axes=None):
    """N-dimensional multiresolution analysis.

    Parameters
    ----------
    data : array
        Input array.
    wavelet : str or Wavelet
        Wavelet to use.
    mode : str
        Signal extension mode. Default 'symmetric'.
    level : int, optional
        Decomposition level. Default is the maximum useful level.
    axes : sequence of int, optional
        Axes over which to compute the MRA. Default is all axes.

    Returns
    -------
    list of array
        Components whose sum reconstructs the input data.
    """
    return _mra_nd(data, wavelet, mode, level, axes)


def imran(mra_coeffs):
    """Inverse n-dimensional multiresolution analysis.

    Parameters
    ----------
    mra_coeffs : list of array
        MRA components from :func:`mran`.

    Returns
    -------
    array
        Sum of all components.
    """
    return sum(mra_coeffs)
