"""Core 1D discrete wavelet transform."""
import jax.numpy as jnp
from jaxwt._filters import get_wavelet

# pywt mode -> jnp.pad mode
_PAD_MODES = {
    'symmetric': 'symmetric',
    'reflect': 'reflect',
    'zero': 'constant',
    'periodic': 'wrap',
}


def dwt_max_level(data_len, filter_len):
    """Maximum useful decomposition level."""
    return int(jnp.floor(jnp.log2(data_len / (filter_len - 1))))


def _pad_periodization(x, pad_left, pad_right):
    """Periodization padding: promote odd length to even, then wrap."""
    if x.shape[0] % 2:
        x = jnp.concatenate([x, x[-1:]])
    return jnp.pad(x, (pad_left, pad_right), mode='wrap')


def _pad(x, pad_left, pad_right, mode):
    """Pad signal according to boundary mode."""
    if mode == 'periodization':
        return _pad_periodization(x, pad_left, pad_right)
    return jnp.pad(x, (pad_left, pad_right), mode=_PAD_MODES[mode])


def dwt(x, wavelet, mode='symmetric'):
    """1D discrete wavelet transform.

    Parameters
    ----------
    x : 1D array
    wavelet : str or Wavelet NamedTuple
    mode : str

    Returns
    -------
    cA, cD : approximation and detail coefficients
    """
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet
    F = w.dec_lo.shape[0]

    if mode == 'periodization':
        N = x.shape[0]
        x_padded = _pad(x, F // 2, F // 2, mode)
        cA = jnp.convolve(x_padded, w.dec_lo, mode='valid')[::2][:int(jnp.ceil(N / 2))]
        cD = jnp.convolve(x_padded, w.dec_hi, mode='valid')[::2][:int(jnp.ceil(N / 2))]
    else:
        x_padded = _pad(x, F - 2, F - 1, mode)
        cA = jnp.convolve(x_padded, w.dec_lo, mode='valid')[::2]
        cD = jnp.convolve(x_padded, w.dec_hi, mode='valid')[::2]

    return cA, cD


def idwt(cA, cD, wavelet, mode='symmetric'):
    """1D inverse discrete wavelet transform.

    Parameters
    ----------
    cA : approximation coefficients (or None-like zeros)
    cD : detail coefficients (or None-like zeros)
    wavelet : str or Wavelet NamedTuple
    mode : str

    Returns
    -------
    Reconstructed signal.
    """
    w = get_wavelet(wavelet) if isinstance(wavelet, str) else wavelet

    def _idwt_component(coeffs, rec_filter):
        filt_even = rec_filter[::2]
        filt_odd = rec_filter[1::2]
        out_even = jnp.convolve(coeffs, filt_even, mode='valid')
        out_odd = jnp.convolve(coeffs, filt_odd, mode='valid')
        result = jnp.empty(2 * out_even.shape[0])
        return result.at[::2].set(out_even).at[1::2].set(out_odd)

    rec = _idwt_component(cA, w.rec_lo) + _idwt_component(cD, w.rec_hi)
    return rec
