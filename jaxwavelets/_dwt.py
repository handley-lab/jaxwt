"""Core 1D discrete wavelet transform."""

import math

import jax
import jax.numpy as jnp

from jaxwavelets._filters import get_wavelet


def dwt_max_level(data_len, filter_len):
    """Maximum useful decomposition level for a given data and filter length.

    Parameters
    ----------
    data_len : int
        Length of the input signal.
    filter_len : int
        Length of the wavelet filter.

    Returns
    -------
    int
        Maximum decomposition level.
    """
    return int(math.floor(math.log2(data_len / (filter_len - 1))))


def dwt(x, wavelet, mode="symmetric"):
    """1D discrete wavelet transform.

    Parameters
    ----------
    x : array
        1D input signal.
    wavelet : str or Wavelet
        Wavelet to use.
    mode : str
        Signal extension mode. Default 'symmetric'.

    Returns
    -------
    cA : array
        Approximation coefficients.
    cD : array
        Detail coefficients.
    """
    w = get_wavelet(wavelet)
    F = w.dec_lo.shape[0]
    if mode == "periodization":
        if x.shape[0] % 2:
            x = jnp.concatenate([x, x[-1:]])
        xp = jnp.pad(x, (F // 2 - 1, F // 2 - 1), mode="wrap")
        M = int(math.ceil(x.shape[0] / 2))
        return jnp.convolve(xp, w.dec_lo, mode="valid")[::2][:M], jnp.convolve(
            xp, w.dec_hi, mode="valid"
        )[::2][:M]
    xp = jnp.pad(x, (F - 2, F - 1), mode=mode)
    return jnp.convolve(xp, w.dec_lo, mode="valid")[::2], jnp.convolve(
        xp, w.dec_hi, mode="valid"
    )[::2]


def idwt(cA, cD, wavelet, mode="symmetric"):
    """1D inverse discrete wavelet transform.

    Parameters
    ----------
    cA : array
        Approximation coefficients.
    cD : array
        Detail coefficients.
    wavelet : str or Wavelet
        Wavelet to use.
    mode : str
        Signal extension mode. Default 'symmetric'.

    Returns
    -------
    array
        Reconstructed signal.
    """
    w = get_wavelet(wavelet)
    if mode == "periodization":
        return _upc_per(cA, w.rec_lo) + _upc_per(cD, w.rec_hi)
    return _upc(cA, w.rec_lo) + _upc(cD, w.rec_hi)


def downcoef(part, data, wavelet, mode="symmetric", level=1):
    """Partial DWT: extract a single subband at the given decomposition level.

    Parameters
    ----------
    part : str
        Subband to compute: 'a' for approximation, 'd' for detail.
    data : array
        1D input signal.
    wavelet : str or Wavelet
        Wavelet to use.
    mode : str
        Signal extension mode. Default 'symmetric'.
    level : int
        Decomposition level. Default 1.

    Returns
    -------
    array
        Coefficients of the requested subband.

    Notes
    -----
    'a' applies the low-pass filter ``level`` times. 'd' applies the
    low-pass filter ``level - 1`` times, then the high-pass filter once.
    """
    w = get_wavelet(wavelet)
    for _ in range(level - 1):
        data, _ = dwt(data, w, mode)
    return {"a": lambda: dwt(data, w, mode)[0], "d": lambda: dwt(data, w, mode)[1]}[
        part
    ]()


def upcoef(part, coeffs, wavelet, level=1, take=0):
    """Partial inverse DWT: reconstruct from a single subband.

    Parameters
    ----------
    part : str
        Subband type: 'a' for approximation, 'd' for detail.
    coeffs : array
        1D coefficient array.
    wavelet : str or Wavelet
        Wavelet to use.
    level : int
        Number of reconstruction levels. Default 1.
    take : int
        If positive, extract a centred window of this length from the
        result. Must be static (determines output shape). Default 0.

    Returns
    -------
    array
        Reconstructed signal.

    Notes
    -----
    'a' uses ``rec_lo`` at every level. 'd' uses ``rec_hi`` first,
    then ``rec_lo`` for subsequent levels.
    """
    w = get_wavelet(wavelet)
    first_filter = {"a": w.rec_lo, "d": w.rec_hi}[part]
    rec = _upcoef_step(coeffs, first_filter)
    for _ in range(level - 1):
        rec = _upcoef_step(rec, w.rec_lo)
    if take > 0:
        start = rec.shape[0] // 2 - take // 2
        rec = jax.lax.dynamic_slice(rec, (start,), (take,))
    return rec


def _upcoef_step(c, f):
    """Single upsample-by-2 then full convolve."""
    return jnp.convolve(jnp.zeros(2 * c.shape[0] - 1).at[::2].set(c), f)


def _upc(c, f):
    """Upsample-convolve: even/odd filter splitting."""
    e = jnp.convolve(c, f[::2], mode="valid")
    o = jnp.convolve(c, f[1::2], mode="valid")
    return jnp.stack([e, o], axis=1).reshape(-1)


def _upc_per(c, f):
    """Upsample-convolve for periodization: circular convolution."""
    F, L = f.shape[0], 2 * c.shape[0]
    u = jnp.zeros(L).at[::2].set(c)
    z = jnp.convolve(jnp.pad(u, (F - 1, F - 1), mode="wrap"), f, mode="valid")
    return jnp.roll(z[:L], -(F // 2 - 1))
