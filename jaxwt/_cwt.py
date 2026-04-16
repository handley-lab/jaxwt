"""Continuous wavelet transform.

All internal computation uses separate real/imaginary arrays (no complex JAX
arrays) matching pywt's C implementation. Complex arrays are formed only at
the final output boundary for API compatibility.
"""
import math
import re
from typing import NamedTuple

import jax.numpy as jnp


class ContinuousWavelet(NamedTuple):
    name: str
    lower_bound: float
    upper_bound: float
    complex_cwt: bool


def as_wavelet(wavelet):
    return wavelet if isinstance(wavelet, ContinuousWavelet) else _wavelet_from_name(wavelet)


_WAVELET_SPECS = {
    'morl': (-8., 8., False),
    'mexh': (-8., 8., False),
    **{f'gaus{i}': (-5., 5., False) for i in range(1, 9)},
    **{f'cgau{i}': (-5., 5., True) for i in range(1, 9)},
}

_PARAM_FAMILIES = {'cmor': (-8., 8.), 'shan': (-20., 20.), 'fbsp': (-20., 20.)}


def _wavelet_from_name(name):
    if name in _WAVELET_SPECS:
        return ContinuousWavelet(name, *_WAVELET_SPECS[name])
    for prefix, bounds in _PARAM_FAMILIES.items():
        if name.startswith(prefix):
            return ContinuousWavelet(name, *bounds, True)
    return _WAVELET_SPECS[name]  # KeyError for unsupported names


def _parse_params(name, prefix):
    return tuple(float(p) if '.' in p else int(p) for p in name[len(prefix):].split('-'))


# --- Real wavelet functions ---

def _morl(x):
    return jnp.cos(5. * x) * jnp.exp(-.5 * x**2)


def _mexh(x):
    return (1. - x**2) * jnp.exp(-.5 * x**2) * 2. / (jnp.sqrt(3.) * jnp.sqrt(jnp.sqrt(jnp.pi)))


def _gaus(x, n):
    e = jnp.exp(-x**2)
    s = jnp.sqrt(jnp.pi / 2.)
    return {
        1: lambda: -2.*x*e / jnp.sqrt(s),
        2: lambda: -2.*(2.*x**2 - 1.)*e / jnp.sqrt(3.*s),
        3: lambda: -4.*(-2.*x**3 + 3.*x)*e / jnp.sqrt(15.*s),
        4: lambda: 4.*(-12.*x**2 + 4.*x**4 + 3.)*e / jnp.sqrt(105.*s),
        5: lambda: 8.*(-4.*x**5 + 20.*x**3 - 15.*x)*e / jnp.sqrt(945.*s),
        6: lambda: -8.*(8.*x**6 - 60.*x**4 + 90.*x**2 - 15.)*e / jnp.sqrt(10395.*s),
        7: lambda: -16.*(-8.*x**7 + 84.*x**5 - 210.*x**3 + 105.*x)*e / jnp.sqrt(135135.*s),
        8: lambda: 16.*(16.*x**8 - 224.*x**6 + 840.*x**4 - 840.*x**2 + 105.)*e / jnp.sqrt(2027025.*s),
    }[n]()


# --- Complex wavelet functions (return (real, imag) tuples) ---

def _cgau(x, n):
    c, s, e = jnp.cos(x), jnp.sin(x), jnp.exp(-x**2)
    norms = [2., 10., 76., 764., 9496., 140152., 2390480., 46206736.]
    d = jnp.sqrt(norms[n-1] * jnp.sqrt(jnp.pi / 2.))
    polys = {
        1: (-2.*x*c - s, 2.*x*s - c),
        2: (4.*x**2*c + 4.*x*s - 3.*c, -4.*x**2*s + 4.*x*c + 3.*s),
        3: (-8.*x**3*c - 12.*x**2*s + 18.*x*c + 7.*s, 8.*x**3*s - 12.*x**2*c - 18.*x*s + 7.*c),
        4: (16.*x**4*c + 32.*x**3*s - 72.*x**2*c - 56.*x*s + 25.*c, -16.*x**4*s + 32.*x**3*c + 72.*x**2*s - 56.*x*c - 25.*s),
        5: (-32.*x**5*c - 80.*x**4*s + 240.*x**3*c + 280.*x**2*s - 250.*x*c - 81.*s, 32.*x**5*s - 80.*x**4*c - 240.*x**3*s + 280.*x**2*c + 250.*x*s - 81.*c),
        6: (64.*x**6*c + 192.*x**5*s - 720.*x**4*c - 1120.*x**3*s + 1500.*x**2*c + 972.*x*s - 331.*c, -64.*x**6*s + 192.*x**5*c + 720.*x**4*s - 1120.*x**3*c - 1500.*x**2*s + 972.*x*c + 331.*s),
        7: (-128.*x**7*c - 448.*x**6*s + 2016.*x**5*c + 3920.*x**4*s - 7000.*x**3*c - 6804.*x**2*s + 4634.*x*c + 1303.*s, 128.*x**7*s - 448.*x**6*c - 2016.*x**5*s + 3920.*x**4*c + 7000.*x**3*s - 6804.*x**2*c - 4634.*x*s + 1303.*c),
        8: (256.*x**8*c + 1024.*x**7*s - 5376.*x**6*c - 12544.*x**5*s + 28000.*x**4*c + 36288.*x**3*s - 37072.*x**2*c - 20848.*x*s + 5937.*c, -256.*x**8*s + 1024.*x**7*c + 5376.*x**6*s - 12544.*x**5*c - 28000.*x**4*s + 36288.*x**3*c + 37072.*x**2*s - 20848.*x*c - 5937.*s),
    }
    re_part, im_part = polys[n]
    return re_part * e / d, im_part * e / d


def _cmor(x, fb, fc):
    envelope = jnp.exp(-x**2 / fb) / jnp.sqrt(jnp.pi * fb)
    return jnp.cos(2 * jnp.pi * fc * x) * envelope, jnp.sin(2 * jnp.pi * fc * x) * envelope


def _shan(x, fb, fc):
    carrier_r = jnp.cos(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    carrier_i = jnp.sin(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    sinc_z = jnp.sinc(fb * x)
    return carrier_r * sinc_z, carrier_i * sinc_z


def _fbsp(x, m, fb, fc):
    carrier_r = jnp.cos(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    carrier_i = jnp.sin(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    sinc_z = jnp.sinc(x * fb / m)**m
    return carrier_r * sinc_z, carrier_i * sinc_z


# --- Dispatcher ---

def _psi(w, x):
    """Returns array for real wavelets, (array, array) for complex."""
    name = w.name
    if name == 'morl': return _morl(x)
    if name == 'mexh': return _mexh(x)
    if name.startswith('gaus'): return _gaus(x, int(name[4:]))
    if name.startswith('cgau'): return _cgau(x, int(name[4:]))
    if name.startswith('cmor'):
        fb, fc = _parse_params(name, 'cmor')
        return _cmor(x, fb, fc)
    if name.startswith('shan'):
        fb, fc = _parse_params(name, 'shan')
        return _shan(x, fb, fc)
    if name.startswith('fbsp'):
        m, fb, fc = _parse_params(name, 'fbsp')
        return _fbsp(x, m, fb, fc)


# --- Utility functions ---

def wavefun(wavelet, precision=8):
    """Sample wavelet function. Returns (psi, x) or ((psi_r, psi_i), x) for complex."""
    w = as_wavelet(wavelet)
    x = jnp.linspace(w.lower_bound, w.upper_bound, 2**precision)
    psi = _psi(w, x)
    if w.complex_cwt:
        return psi[0] + 1j * psi[1], x  # combine for public API
    return psi, x


def integrate_wavelet(wavelet, precision=8):
    """Integrate wavelet function. Returns (int_psi, x) — int_psi may be complex for public API."""
    w = as_wavelet(wavelet)
    x = jnp.linspace(w.lower_bound, w.upper_bound, 2**precision)
    step = x[1] - x[0]
    psi = _psi(w, x)
    if w.complex_cwt:
        return jnp.cumsum(psi[0]) * step + 1j * jnp.cumsum(psi[1]) * step, x
    return jnp.cumsum(psi) * step, x


def _integrate_wavelet_real_imag(wavelet, precision=8):
    """Internal: returns ((int_r, int_i), x) for complex, (int_psi, x) for real."""
    w = as_wavelet(wavelet)
    x = jnp.linspace(w.lower_bound, w.upper_bound, 2**precision)
    step = x[1] - x[0]
    psi = _psi(w, x)
    if w.complex_cwt:
        return (jnp.cumsum(psi[0]) * step, jnp.cumsum(psi[1]) * step), x
    return jnp.cumsum(psi) * step, x


def central_frequency(wavelet, precision=8):
    """Central frequency via FFT peak."""
    psi, x = wavefun(wavelet, precision)  # complex combination is fine here — one-off
    domain = x[-1] - x[0]
    index = jnp.argmax(jnp.abs(jnp.fft.fft(psi))[1:]) + 2
    index = jnp.where(index > psi.shape[0] / 2, psi.shape[0] - index + 2, index)
    return 1. / (domain / (index - 1))


def scale2frequency(wavelet, scale, precision=8):
    return central_frequency(wavelet, precision) / scale


# --- CWT ---

def cwt(data, scales, wavelet, sampling_period=1., method='conv', precision=12):
    """1D continuous wavelet transform. All internal computation is real-only."""
    w = as_wavelet(wavelet)
    int_psi_raw, x = _integrate_wavelet_real_imag(w, precision)

    step = x[1] - x[0]
    width = x[-1] - x[0]
    N = data.shape[0]

    out_r, out_i = [], []
    for scale in list(scales):
        j = jnp.arange(scale * width + 1) / (scale * step)
        j = jnp.floor(j).astype(jnp.int32)
        j = j[j < (int_psi_raw[0] if w.complex_cwt else int_psi_raw).shape[0]]

        if w.complex_cwt:
            # Conjugate: negate imaginary part
            kernel_r = int_psi_raw[0][j][::-1]
            kernel_i = -int_psi_raw[1][j][::-1]

            if method == 'fft':
                n = N + kernel_r.shape[0] - 1
                size = 1 << int(math.ceil(math.log2(n)))
                data_fft = jnp.fft.rfft(data, size)
                kr_fft = jnp.fft.rfft(kernel_r, size)
                ki_fft = jnp.fft.rfft(kernel_i, size)
                conv_r = jnp.fft.irfft(data_fft * kr_fft, size)[:n]
                conv_i = jnp.fft.irfft(data_fft * ki_fft, size)[:n]
            else:
                conv_r = jnp.convolve(data, kernel_r)
                conv_i = jnp.convolve(data, kernel_i)

            coef_r = -jnp.sqrt(scale) * jnp.diff(conv_r)
            coef_i = -jnp.sqrt(scale) * jnp.diff(conv_i)
        else:
            kernel = int_psi_raw[j][::-1]
            if method == 'fft':
                n = N + kernel.shape[0] - 1
                size = 1 << int(math.ceil(math.log2(n)))
                conv = jnp.fft.irfft(jnp.fft.rfft(data, size) * jnp.fft.rfft(kernel, size), size)[:n]
            else:
                conv = jnp.convolve(data, kernel)
            coef_r = -jnp.sqrt(scale) * jnp.diff(conv)
            coef_i = None

        d = (coef_r.shape[0] - N) / 2.
        if d > 0:
            lo, hi = math.floor(d), coef_r.shape[0] - math.ceil(d)
            coef_r = coef_r[lo:hi]
            if coef_i is not None:
                coef_i = coef_i[lo:hi]

        out_r.append(coef_r)
        if coef_i is not None:
            out_i.append(coef_i)

    freqs = scale2frequency(w, jnp.asarray(scales), precision) / sampling_period
    if w.complex_cwt:
        return jnp.stack(out_r) + 1j * jnp.stack(out_i), freqs
    return jnp.stack(out_r), freqs
