"""Continuous wavelet transform.

All internal computation uses separate real/imaginary arrays (no complex JAX
arrays) matching pywt's C implementation. Complex arrays are formed only at
the final output boundary for API compatibility.

Two-phase design: prepare_cwt() builds the kernel bank (static/trace-time),
apply_cwt() runs the transform (JIT-compatible).
"""
import math
from typing import NamedTuple

import jax
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
    return _WAVELET_SPECS[name]


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


def _psi(w, x):
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


# --- Public utility functions ---

def wavefun(wavelet, precision=8):
    w = as_wavelet(wavelet)
    x = jnp.linspace(w.lower_bound, w.upper_bound, 2**precision)
    psi = _psi(w, x)
    if w.complex_cwt:
        return psi[0] + 1j * psi[1], x
    return psi, x


def integrate_wavelet(wavelet, precision=8):
    w = as_wavelet(wavelet)
    x = jnp.linspace(w.lower_bound, w.upper_bound, 2**precision)
    step = x[1] - x[0]
    psi = _psi(w, x)
    if w.complex_cwt:
        return jnp.cumsum(psi[0]) * step + 1j * jnp.cumsum(psi[1]) * step, x
    return jnp.cumsum(psi) * step, x


def central_frequency(wavelet, precision=8):
    psi, x = wavefun(wavelet, precision)
    domain = x[-1] - x[0]
    index = jnp.argmax(jnp.abs(jnp.fft.fft(psi))[1:]) + 2
    index = jnp.where(index > psi.shape[0] / 2, psi.shape[0] - index + 2, index)
    return 1. / (domain / (index - 1))


def scale2frequency(wavelet, scale, precision=8):
    return central_frequency(wavelet, precision) / scale


# --- Two-phase CWT: prepare + apply ---

class CWTKernelBank:
    """Precomputed CWT kernels. kernels/freqs are traced; lengths/scales/method are static."""
    __slots__ = ('kernels_r', 'kernels_i', 'kernel_lengths', 'scales_sqrt', 'complex_cwt', 'method', 'freqs')

    def __init__(self, kernels_r, kernels_i, kernel_lengths, scales_sqrt, complex_cwt, method, freqs):
        self.kernels_r = kernels_r
        self.kernels_i = kernels_i
        self.kernel_lengths = kernel_lengths
        self.scales_sqrt = scales_sqrt
        self.complex_cwt = complex_cwt
        self.method = method
        self.freqs = freqs


jax.tree_util.register_pytree_node(
    CWTKernelBank,
    lambda b: ((b.kernels_r, b.kernels_i, b.freqs), (b.kernel_lengths, b.scales_sqrt, b.complex_cwt, b.method)),
    lambda aux, children: CWTKernelBank(children[0], children[1], aux[0], aux[1], aux[2], aux[3], children[2]),
)


def prepare_cwt(scales, wavelet, sampling_period=1., method='conv', precision=12):
    """Build CWT kernel bank. Pure JAX, all shapes static from scale/wavelet/precision."""
    w = as_wavelet(wavelet)
    # Derive step/width analytically from wavelet bounds (Python floats, no tracing)
    n_samples = 2**precision
    step = (w.upper_bound - w.lower_bound) / (n_samples - 1)
    width = w.upper_bound - w.lower_bound

    # Evaluate and integrate wavelet function
    x = jnp.linspace(w.lower_bound, w.upper_bound, n_samples)
    psi = _psi(w, x)
    if w.complex_cwt:
        int_r = jnp.cumsum(psi[0]) * step
        int_i = jnp.cumsum(psi[1]) * step
    else:
        int_psi = jnp.cumsum(psi) * step

    # Build per-scale kernels
    kernels_r, kernels_i, lengths = [], [], []
    for scale in scales:
        scale = float(scale)
        j = jnp.floor(jnp.arange(scale * width + 1) / (scale * step)).astype(jnp.int32)
        j = j[j < n_samples]
        L = int(j.shape[0])
        lengths.append(L)
        if w.complex_cwt:
            kernels_r.append(int_r[j][::-1])
            kernels_i.append(-int_i[j][::-1])  # conjugate
        else:
            kernels_r.append(int_psi[j][::-1])

    Lmax = max(lengths)
    pad_r = jnp.stack([jnp.pad(k, (0, Lmax - k.shape[0])) for k in kernels_r])
    pad_i = jnp.stack([jnp.pad(k, (0, Lmax - k.shape[0])) for k in kernels_i]) if w.complex_cwt else jnp.empty((0,))

    freqs = scale2frequency(w, jnp.asarray(scales), precision) / sampling_period

    return CWTKernelBank(
        kernels_r=pad_r,
        kernels_i=pad_i,
        kernel_lengths=tuple(lengths),
        scales_sqrt=tuple(math.sqrt(float(s)) for s in scales),
        complex_cwt=w.complex_cwt,
        method=method,
        freqs=freqs,
    )


def apply_cwt(data, bank):
    """Apply precomputed CWT kernel bank to data. Fully JIT-compatible."""
    N = data.shape[0]
    Lmax = bank.kernels_r.shape[1]
    n_scales = bank.kernels_r.shape[0]

    if bank.method == 'fft':
        fft_size = 1 << int(math.ceil(math.log2(N + Lmax - 1)))
        data_fft = jnp.fft.rfft(data, fft_size)

    out_r, out_i = [], []
    for i in range(n_scales):
        L = bank.kernel_lengths[i]
        scale_sqrt = bank.scales_sqrt[i]
        conv_len = N + L - 1
        d = (conv_len - 1 - N) / 2.
        lo = math.floor(d)
        hi = conv_len - 1 - math.ceil(d)

        if bank.method == 'fft':
            conv_r = jnp.fft.irfft(data_fft * jnp.fft.rfft(bank.kernels_r[i], fft_size), fft_size)
        else:
            conv_r = jnp.convolve(data, bank.kernels_r[i])

        coef_r = -scale_sqrt * jnp.diff(conv_r[:conv_len])
        out_r.append(coef_r[lo:hi])

        if bank.complex_cwt:
            if bank.method == 'fft':
                conv_i = jnp.fft.irfft(data_fft * jnp.fft.rfft(bank.kernels_i[i], fft_size), fft_size)
            else:
                conv_i = jnp.convolve(data, bank.kernels_i[i])
            coef_i = -scale_sqrt * jnp.diff(conv_i[:conv_len])
            out_i.append(coef_i[lo:hi])

    if bank.complex_cwt:
        return jnp.stack(out_r) + 1j * jnp.stack(out_i), bank.freqs
    return jnp.stack(out_r), bank.freqs


def cwt(data, scales, wavelet, sampling_period=1., method='conv', precision=12):
    """1D continuous wavelet transform. Convenience wrapper around prepare_cwt + apply_cwt."""
    bank = prepare_cwt(scales, wavelet, sampling_period, method, precision)
    return apply_cwt(data, bank)
