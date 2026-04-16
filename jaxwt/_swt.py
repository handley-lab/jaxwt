"""Stationary (undecimated) wavelet transform — algorithme à trous."""

import math
from itertools import product

import jax
import jax.numpy as jnp
from jaxwt._dwt import idwt
from jaxwt._multidim import idwtn
from jaxwt._filters import get_wavelet, Wavelet


def swt_max_level(input_len):
    """Maximum SWT decomposition level for a given signal length.

    Parameters
    ----------
    input_len : int
        Length of the input signal.

    Returns
    -------
    int
        Maximum decomposition level.
    """
    return int(math.floor(math.log2(input_len)))


def _dilate(filt, dilation):
    """Insert dilation-1 zeros between filter taps."""
    if dilation == 1:
        return filt
    out = jnp.zeros(len(filt) + (len(filt) - 1) * (dilation - 1))
    return out.at[::dilation].set(filt)


def _swt_1d(data, w, level, start_level):
    """Core 1D SWT forward along last axis."""
    L = w.dec_lo.shape[0]
    a = data
    coeffs = []
    for j in range(start_level, start_level + level):
        lo, hi = _dilate(w.dec_lo, 2**j), _dilate(w.dec_hi, 2**j)
        F = lo.shape[0]
        xp = jnp.pad(a, (F - 1, F - 1), mode="wrap")
        s = (L // 2) * 2**j
        cA = jnp.convolve(xp, lo, mode="valid")[s : s + a.shape[0]]
        cD = jnp.convolve(xp, hi, mode="valid")[s : s + a.shape[0]]
        coeffs.append((cA, cD))
        a = cA
    return coeffs


def _swt_axis(data, w, level, start_level, axis):
    """1D SWT along one axis via structural vmap."""
    data = jnp.moveaxis(data, axis, -1)
    shape = data.shape[:-1]
    flat = data.reshape(-1, data.shape[-1])
    results = [_swt_1d(flat[i], w, level, start_level) for i in range(flat.shape[0])]
    out = []
    for lev in range(level):
        cA = jnp.stack([r[lev][0] for r in results]).reshape(shape + (-1,))
        cD = jnp.stack([r[lev][1] for r in results]).reshape(shape + (-1,))
        out.append((jnp.moveaxis(cA, -1, axis), jnp.moveaxis(cD, -1, axis)))
    return out


# --- 1D API ---


def swt(data, wavelet, level=None, start_level=0, trim_approx=False, norm=False):
    """Multilevel 1D stationary (undecimated) wavelet transform.

    Parameters
    ----------
    data : array
        1D input signal.
    wavelet : str or Wavelet
        Wavelet to use.
    level : int, optional
        Number of decomposition levels. Default is the maximum level.
    start_level : int
        Starting decomposition level. Default 0.
    trim_approx : bool
        If True, return ``[cA_n, cD_n, ..., cD_1]`` instead of
        ``[(cA_n, cD_n), ..., (cA_1, cD_1)]``. Default False.
    norm : bool
        If True, rescale filters by ``1/sqrt(2)`` for energy
        preservation. Default False.

    Returns
    -------
    list
        Coefficients from coarsest to finest level. Format depends on
        ``trim_approx``.
    """
    w = get_wavelet(wavelet)
    if norm:
        w = Wavelet(*(f / jnp.sqrt(2) for f in w))
    if level is None:
        level = swt_max_level(data.shape[0])
    coeffs = _swt_1d(data, w, level, start_level)
    coeffs.reverse()
    if trim_approx:
        return [coeffs[0][0]] + [cD for _, cD in coeffs]
    return coeffs


def iswt(coeffs, wavelet, norm=False):
    """Multilevel 1D inverse stationary wavelet transform.

    Parameters
    ----------
    coeffs : list
        SWT coefficients as returned by :func:`swt`.
    wavelet : str or Wavelet
        Wavelet to use.
    norm : bool
        If True, rescale filters by ``sqrt(2)`` to undo normalisation.
        Default False.

    Returns
    -------
    array
        Reconstructed signal.
    """
    w = get_wavelet(wavelet)
    if norm:
        w = Wavelet(*(f * jnp.sqrt(2) for f in w))
    trim_approx = not isinstance(coeffs[0], (tuple, list))
    output = coeffs[0] if trim_approx else coeffs[0][0]
    detail_coeffs = coeffs[1:] if trim_approx else coeffs
    num_levels = len(detail_coeffs)
    for j in range(num_levels, 0, -1):
        step_size = 2 ** (j - 1)
        cD = detail_coeffs[-j] if trim_approx else detail_coeffs[-j][1]
        N = output.shape[0]
        for first in range(step_size):
            indices = jnp.arange(first, N, step_size)
            even_idx, odd_idx = indices[0::2], indices[1::2]
            x1 = idwt(output[even_idx], cD[even_idx], w, "periodization")
            x2 = jnp.roll(idwt(output[odd_idx], cD[odd_idx], w, "periodization"), 1)
            output = output.at[indices].set((x1 + x2) / 2)
    return output


# --- nD API ---


def swtn(data, wavelet, level, start_level=0, axes=None, trim_approx=False, norm=False):
    """Multilevel n-dimensional stationary wavelet transform.

    Parameters
    ----------
    data : array
        Input array.
    wavelet : str or Wavelet
        Wavelet to use.
    level : int
        Number of decomposition levels.
    start_level : int
        Starting decomposition level. Default 0.
    axes : sequence of int, optional
        Axes over which to compute the SWT. Default is all axes.
    trim_approx : bool
        If True, separate the final approximation from the detail
        dictionaries. Default False.
    norm : bool
        If True, rescale filters for energy preservation. Default False.

    Returns
    -------
    list
        Subband coefficient dictionaries from coarsest to finest level.
    """
    if axes is None:
        axes = tuple(range(data.ndim))
    else:
        axes = tuple(axes)
    w = get_wavelet(wavelet)
    if norm:
        w = Wavelet(*(f / jnp.sqrt(2) for f in w))
    ndim = len(axes)
    ret = []
    for i in range(start_level, start_level + level):
        coeffs = [("", data)]
        for axis in axes:
            new_coeffs = []
            for subband, x in coeffs:
                pair = _swt_axis(x, w, level=1, start_level=i, axis=axis)
                cA, cD = pair[0]
                new_coeffs.extend([(subband + "a", cA), (subband + "d", cD)])
            coeffs = new_coeffs
        coeffs = dict(coeffs)
        ret.append(coeffs)
        data = coeffs["a" * ndim]
        if trim_approx:
            coeffs.pop("a" * ndim)
    if trim_approx:
        ret.append(data)
    ret.reverse()
    return ret


def swt2(
    data, wavelet, level, start_level=0, axes=(-2, -1), trim_approx=False, norm=False
):
    """Multilevel 2D stationary wavelet transform.

    Parameters
    ----------
    data : array
        2D input array.
    wavelet : str or Wavelet
        Wavelet to use.
    level : int
        Number of decomposition levels.
    start_level : int
        Starting decomposition level. Default 0.
    axes : tuple of int
        Axes for the 2D transform. Default ``(-2, -1)``.
    trim_approx : bool
        If True, separate the final approximation. Default False.
    norm : bool
        If True, rescale filters for energy preservation. Default False.

    Returns
    -------
    list
        Subband coefficient dictionaries from coarsest to finest level.
    """
    return swtn(data, wavelet, level, start_level, axes, trim_approx, norm)


def iswtn(coeffs, wavelet, axes=None, norm=False):
    """Multilevel n-dimensional inverse stationary wavelet transform.

    Parameters
    ----------
    coeffs : list
        SWT coefficients as returned by :func:`swtn`.
    wavelet : str or Wavelet
        Wavelet to use.
    axes : sequence of int, optional
        Axes over which the transform was computed. Default inferred
        from coefficient keys.
    norm : bool
        If True, rescale filters to undo normalisation. Default False.

    Returns
    -------
    array
        Reconstructed array.
    """
    w = get_wavelet(wavelet)
    if norm:
        w = Wavelet(*(f * jnp.sqrt(2) for f in w))
    trim_approx = not isinstance(coeffs[0], dict)
    if axes is not None:
        axes = tuple(axes)
    if trim_approx:
        output = coeffs[0]
        detail_list = coeffs[1:]
    else:
        first_dict = coeffs[0]
        n = len(axes) if axes is not None else len(next(iter(first_dict)))
        output = first_dict["a" * n]
        detail_list = coeffs
    if axes is None:
        axes = tuple(range(output.ndim))
    ndim_transform = len(axes)
    num_levels = len(detail_list)

    for j in range(num_levels):
        step_size = int(2 ** (num_levels - j - 1))
        if trim_approx:
            details = detail_list[j]
        else:
            details = {
                k: v for k, v in detail_list[j].items() if k != "a" * ndim_transform
            }

        for firsts in product(*[range(step_size)] * ndim_transform):
            approx = output.copy()
            indices = [slice(None)] * output.ndim
            even_indices = [slice(None)] * output.ndim
            odd_indices = [slice(None)] * output.ndim
            for first, ax in zip(firsts, axes):
                sh = output.shape[ax]
                indices[ax] = slice(first, sh, step_size)

            output = output.at[tuple(indices)].set(0)
            ntransforms = 0
            for odds in product(*[(0, 1)] * ndim_transform):
                odd_even_slices = [slice(None)] * output.ndim
                for o, first, ax in zip(odds, firsts, axes):
                    sh = output.shape[ax]
                    if o:
                        odd_even_slices[ax] = slice(
                            first + step_size, sh, 2 * step_size
                        )
                    else:
                        odd_even_slices[ax] = slice(first, sh, 2 * step_size)

                details_slice = {
                    k: v[tuple(odd_even_slices)] for k, v in details.items()
                }
                details_slice["a" * ndim_transform] = approx[tuple(odd_even_slices)]
                x = idwtn(details_slice, w, "periodization", axes=axes)
                for o, ax in zip(odds, axes):
                    if o:
                        x = jnp.roll(x, 1, axis=ax)
                output = output.at[tuple(indices)].add(x)
                ntransforms += 1
            output = output.at[tuple(indices)].divide(ntransforms)
    return output


def iswt2(coeffs, wavelet, axes=(-2, -1), norm=False):
    """Multilevel 2D inverse stationary wavelet transform.

    Parameters
    ----------
    coeffs : list
        SWT coefficients as returned by :func:`swt2`.
    wavelet : str or Wavelet
        Wavelet to use.
    axes : tuple of int
        Axes for the 2D transform. Default ``(-2, -1)``.
    norm : bool
        If True, rescale filters to undo normalisation. Default False.

    Returns
    -------
    array
        Reconstructed 2D array.
    """
    return iswtn(coeffs, wavelet, axes, norm)
