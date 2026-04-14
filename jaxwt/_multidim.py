"""nD wavelet transforms."""
from typing import NamedTuple
import jax.numpy as jnp
from jaxwt._dwt import dwt, idwt, dwt_max_level
from jaxwt._filters import get_wavelet


class WaveletCoeffs(NamedTuple):
    """Multilevel wavelet decomposition coefficients."""
    approx: jnp.ndarray
    details: tuple
    shapes: tuple  # original shapes at each level for reconstruction


def dwtn(data, wavelet, mode='symmetric', axes=None):
    """Single-level nD DWT. Returns dict of subbands keyed by 'a'/'d' strings."""
    if axes is None:
        axes = tuple(range(data.ndim))
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    coeffs = [('', data)]
    for axis in axes:
        new_coeffs = []
        for key, x in coeffs:
            cA, cD = _dwt_axis(x, w, mode, axis)
            new_coeffs.append((key + 'a', cA))
            new_coeffs.append((key + 'd', cD))
        coeffs = new_coeffs
    return dict(sorted(coeffs))


def idwtn(coeffs, wavelet, mode='symmetric', axes=None):
    """Single-level nD IDWT."""
    keys = sorted(coeffs.keys())
    ndim = len(keys[0])
    if axes is None:
        axes = tuple(range(ndim))
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    # Reverse axis order for reconstruction
    for i, axis in enumerate(reversed(axes)):
        axis_idx = ndim - 1 - i
        pairs = {}
        for key, arr in coeffs.items():
            base = key[:axis_idx] + key[axis_idx + 1:]
            if base not in pairs:
                pairs[base] = {}
            pairs[base][key[axis_idx]] = arr
        coeffs = {}
        for base, pair in sorted(pairs.items()):
            coeffs[base] = _idwt_axis(pair['a'], pair['d'], w, mode, axis)
    return coeffs['']


def wavedecn(data, wavelet, mode='symmetric', level=None, axes=None):
    """Multilevel nD DWT."""
    if axes is None:
        axes = tuple(range(data.ndim))
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    if level is None:
        level = dwt_max_level(
            min(data.shape[ax] for ax in axes),
            w.dec_lo.shape[0],
        )
    detail_dicts = []
    shapes = []
    a = data
    for _ in range(level):
        shapes.append(a.shape)
        coeffs = dwtn(a, w, mode, axes)
        a_key = 'a' * len(axes)
        a = coeffs.pop(a_key)
        detail_dicts.append(coeffs)
    detail_dicts.reverse()
    shapes.reverse()
    return WaveletCoeffs(approx=a, details=tuple(detail_dicts), shapes=tuple(shapes))


def waverecn(coeffs, wavelet, mode='symmetric', axes=None):
    """Multilevel nD IDWT."""
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    a = coeffs.approx
    for detail_dict, target_shape in zip(coeffs.details, coeffs.shapes):
        # Match approx shape to detail shape (truncate if off by 1)
        d_coeff = next(iter(detail_dict.values()))
        a = a[tuple(slice(s) for s in d_coeff.shape)]
        a_key = 'a' * len(next(iter(detail_dict.keys())))
        all_coeffs = {a_key: a, **detail_dict}
        if axes is None:
            axes = tuple(range(a.ndim))
        a = idwtn(all_coeffs, w, mode, axes)
        # Trim to original shape at this level
        a = a[tuple(slice(s) for s in target_shape)]
    return a


def _dwt_axis(x, wavelet, mode, axis):
    """Apply 1D DWT along a single axis of an nD array."""
    x = jnp.moveaxis(x, axis, -1)
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    cA_list = []
    cD_list = []
    for i in range(x_flat.shape[0]):
        a, d = dwt(x_flat[i], wavelet, mode)
        cA_list.append(a)
        cD_list.append(d)
    cA = jnp.stack(cA_list).reshape(shape[:-1] + (-1,))
    cD = jnp.stack(cD_list).reshape(shape[:-1] + (-1,))
    return jnp.moveaxis(cA, -1, axis), jnp.moveaxis(cD, -1, axis)


def _idwt_axis(cA, cD, wavelet, mode, axis):
    """Apply 1D IDWT along a single axis of an nD array."""
    cA = jnp.moveaxis(cA, axis, -1)
    cD = jnp.moveaxis(cD, axis, -1)
    shape = cA.shape
    cA_flat = cA.reshape(-1, shape[-1])
    cD_flat = cD.reshape(-1, shape[-1])
    rec_list = []
    for i in range(cA_flat.shape[0]):
        rec_list.append(idwt(cA_flat[i], cD_flat[i], wavelet, mode))
    rec = jnp.stack(rec_list).reshape(shape[:-1] + (-1,))
    return jnp.moveaxis(rec, -1, axis)
