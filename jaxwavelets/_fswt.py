"""Fully separable wavelet decomposition."""

import jax
import jax.numpy as jnp

from jaxwavelets._dwt import dwt_max_level
from jaxwavelets._filters import get_wavelet
from jaxwavelets._multidim import _dwt_axis, _idwt_axis


def _wavedec_axis(data, w, mode, level, axis):
    """1D multilevel DWT along one axis. Returns [cA_n, cD_n, ..., cD_1]."""
    coeffs = []
    a = data
    for _ in range(level):
        a, d = _dwt_axis(a, w, mode, axis)
        coeffs.append(d)
    coeffs.append(a)
    coeffs.reverse()
    return coeffs


def _waverec_axis(coeffs, w, mode, axis):
    """1D multilevel IDWT along one axis from [cA_n, cD_n, ..., cD_1]."""
    a = coeffs[0]
    for d in coeffs[1:]:
        slices = [slice(None)] * a.ndim
        slices[axis] = slice(d.shape[axis])
        a = _idwt_axis(a[tuple(slices)], d, w, mode, axis)
    return a


class FswavedecnResult:
    """Result of a fully separable wavelet decomposition.

    Parameters
    ----------
    coeffs : array
        Concatenated coefficient array along each transform axis.
    coeff_slices : tuple of tuple of slice
        Slices to extract each subband along each axis.
    axes : tuple of int
        Axes along which the transform was computed.
    wavelet : Wavelet
        Wavelet used for the decomposition.
    mode : str
        Signal extension mode used.

    Notes
    -----
    ``coeffs`` is JAX-traced; remaining fields are static metadata.
    Registered as a JAX pytree node.
    """

    __slots__ = ("coeffs", "coeff_slices", "axes", "wavelet", "mode")

    def __init__(self, coeffs, coeff_slices, axes, wavelet, mode):
        self.coeffs = coeffs
        self.coeff_slices = coeff_slices
        self.axes = tuple(axes)
        self.wavelet = wavelet
        self.mode = mode

    @property
    def approx(self):
        slices = [slice(None)] * self.coeffs.ndim
        for ax_idx, ax in enumerate(self.axes):
            slices[ax] = self.coeff_slices[ax_idx][0]
        return self.coeffs[tuple(slices)]


jax.tree_util.register_pytree_node(
    FswavedecnResult,
    lambda r: ((r.coeffs,), (r.coeff_slices, r.axes, r.wavelet, r.mode)),
    lambda aux, children: FswavedecnResult(children[0], aux[0], aux[1], aux[2], aux[3]),
)


def fswavedecn(data, wavelet, mode="symmetric", levels=None, axes=None):
    """Fully separable n-dimensional wavelet decomposition.

    Parameters
    ----------
    data : array
        Input array.
    wavelet : str or Wavelet
        Wavelet to use.
    mode : str
        Signal extension mode. Default 'symmetric'.
    levels : int or list of int, optional
        Decomposition level(s) per axis. Default is the maximum useful
        level for each axis.
    axes : sequence of int, optional
        Axes over which to decompose. Default is all axes.

    Returns
    -------
    FswavedecnResult
        Decomposition result containing concatenated coefficients and
        metadata for reconstruction.
    """
    axes = tuple(range(data.ndim)) if axes is None else tuple(axes)
    w = get_wavelet(wavelet)
    if levels is None:
        levels = [dwt_max_level(data.shape[ax], w.dec_lo.shape[0]) for ax in axes]
    elif isinstance(levels, int):
        levels = [levels] * len(axes)

    coeff_slices = []
    arr = data
    for ax, lev in zip(axes, levels, strict=False):
        coeffs = _wavedec_axis(arr, w, mode, lev, ax)
        shapes = [c.shape[ax] for c in coeffs]
        offsets = [0]
        for s in shapes:
            offsets.append(offsets[-1] + s)
        coeff_slices.append(
            tuple(slice(offsets[i], offsets[i + 1]) for i in range(len(shapes)))
        )
        arr = jnp.concatenate(coeffs, axis=ax)

    return FswavedecnResult(arr, tuple(coeff_slices), axes, w, mode)


def fswaverecn(result):
    """Fully separable inverse wavelet reconstruction.

    Parameters
    ----------
    result : FswavedecnResult
        Decomposition result from :func:`fswavedecn`.

    Returns
    -------
    array
        Reconstructed array.
    """
    arr = result.coeffs
    for ax_idx, ax in enumerate(result.axes):
        slices_base = [slice(None)] * arr.ndim
        coeffs = []
        for sl in result.coeff_slices[ax_idx]:
            slices_base[ax] = sl
            coeffs.append(arr[tuple(slices_base)])
        slices_base[ax] = slice(None)
        arr = _waverec_axis(coeffs, result.wavelet, result.mode, ax)
    return arr
