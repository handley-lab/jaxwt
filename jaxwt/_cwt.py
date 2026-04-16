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
    """Continuous wavelet specification.

    Parameters
    ----------
    name : str
        Wavelet family name (e.g. 'morl', 'cmor1.5-1.0').
    lower_bound : float
        Lower bound of the wavelet's effective support.
    upper_bound : float
        Upper bound of the wavelet's effective support.
    complex_cwt : bool
        Whether the wavelet is complex-valued.
    """

    name: str
    lower_bound: float
    upper_bound: float
    complex_cwt: bool


def as_wavelet(wavelet):
    """Return a ContinuousWavelet, looking up by name if needed.

    Parameters
    ----------
    wavelet : str or ContinuousWavelet
        Wavelet name or existing ContinuousWavelet.

    Returns
    -------
    ContinuousWavelet
    """
    return (
        wavelet
        if isinstance(wavelet, ContinuousWavelet)
        else _wavelet_from_name(wavelet)
    )


_WAVELET_SPECS = {
    "morl": (-8.0, 8.0, False),
    "mexh": (-8.0, 8.0, False),
    **{f"gaus{i}": (-5.0, 5.0, False) for i in range(1, 9)},
    **{f"cgau{i}": (-5.0, 5.0, True) for i in range(1, 9)},
}

_PARAM_FAMILIES = {"cmor": (-8.0, 8.0), "shan": (-20.0, 20.0), "fbsp": (-20.0, 20.0)}


def _wavelet_from_name(name):
    if name in _WAVELET_SPECS:
        return ContinuousWavelet(name, *_WAVELET_SPECS[name])
    for prefix, bounds in _PARAM_FAMILIES.items():
        if name.startswith(prefix):
            return ContinuousWavelet(name, *bounds, True)
    return _WAVELET_SPECS[name]


def _parse_params(name, prefix):
    return tuple(
        float(p) if "." in p else int(p) for p in name[len(prefix) :].split("-")
    )


# --- Real wavelet functions ---


def _morl(x):
    return jnp.cos(5.0 * x) * jnp.exp(-0.5 * x**2)


def _mexh(x):
    return (
        (1.0 - x**2)
        * jnp.exp(-0.5 * x**2)
        * 2.0
        / (jnp.sqrt(3.0) * jnp.sqrt(jnp.sqrt(jnp.pi)))
    )


def _gaus(x, n):
    e = jnp.exp(-(x**2))
    s = jnp.sqrt(jnp.pi / 2.0)
    return {
        1: lambda: -2.0 * x * e / jnp.sqrt(s),
        2: lambda: -2.0 * (2.0 * x**2 - 1.0) * e / jnp.sqrt(3.0 * s),
        3: lambda: -4.0 * (-2.0 * x**3 + 3.0 * x) * e / jnp.sqrt(15.0 * s),
        4: lambda: 4.0 * (-12.0 * x**2 + 4.0 * x**4 + 3.0) * e / jnp.sqrt(105.0 * s),
        5: lambda: (
            8.0 * (-4.0 * x**5 + 20.0 * x**3 - 15.0 * x) * e / jnp.sqrt(945.0 * s)
        ),
        6: lambda: (
            -8.0
            * (8.0 * x**6 - 60.0 * x**4 + 90.0 * x**2 - 15.0)
            * e
            / jnp.sqrt(10395.0 * s)
        ),
        7: lambda: (
            -16.0
            * (-8.0 * x**7 + 84.0 * x**5 - 210.0 * x**3 + 105.0 * x)
            * e
            / jnp.sqrt(135135.0 * s)
        ),
        8: lambda: (
            16.0
            * (16.0 * x**8 - 224.0 * x**6 + 840.0 * x**4 - 840.0 * x**2 + 105.0)
            * e
            / jnp.sqrt(2027025.0 * s)
        ),
    }[n]()


# --- Complex wavelet functions (return (real, imag) tuples) ---


def _cgau(x, n):
    c, s, e = jnp.cos(x), jnp.sin(x), jnp.exp(-(x**2))
    norms = [2.0, 10.0, 76.0, 764.0, 9496.0, 140152.0, 2390480.0, 46206736.0]
    d = jnp.sqrt(norms[n - 1] * jnp.sqrt(jnp.pi / 2.0))
    polys = {
        1: (-2.0 * x * c - s, 2.0 * x * s - c),
        2: (
            4.0 * x**2 * c + 4.0 * x * s - 3.0 * c,
            -4.0 * x**2 * s + 4.0 * x * c + 3.0 * s,
        ),
        3: (
            -8.0 * x**3 * c - 12.0 * x**2 * s + 18.0 * x * c + 7.0 * s,
            8.0 * x**3 * s - 12.0 * x**2 * c - 18.0 * x * s + 7.0 * c,
        ),
        4: (
            16.0 * x**4 * c
            + 32.0 * x**3 * s
            - 72.0 * x**2 * c
            - 56.0 * x * s
            + 25.0 * c,
            -16.0 * x**4 * s
            + 32.0 * x**3 * c
            + 72.0 * x**2 * s
            - 56.0 * x * c
            - 25.0 * s,
        ),
        5: (
            -32.0 * x**5 * c
            - 80.0 * x**4 * s
            + 240.0 * x**3 * c
            + 280.0 * x**2 * s
            - 250.0 * x * c
            - 81.0 * s,
            32.0 * x**5 * s
            - 80.0 * x**4 * c
            - 240.0 * x**3 * s
            + 280.0 * x**2 * c
            + 250.0 * x * s
            - 81.0 * c,
        ),
        6: (
            64.0 * x**6 * c
            + 192.0 * x**5 * s
            - 720.0 * x**4 * c
            - 1120.0 * x**3 * s
            + 1500.0 * x**2 * c
            + 972.0 * x * s
            - 331.0 * c,
            -64.0 * x**6 * s
            + 192.0 * x**5 * c
            + 720.0 * x**4 * s
            - 1120.0 * x**3 * c
            - 1500.0 * x**2 * s
            + 972.0 * x * c
            + 331.0 * s,
        ),
        7: (
            -128.0 * x**7 * c
            - 448.0 * x**6 * s
            + 2016.0 * x**5 * c
            + 3920.0 * x**4 * s
            - 7000.0 * x**3 * c
            - 6804.0 * x**2 * s
            + 4634.0 * x * c
            + 1303.0 * s,
            128.0 * x**7 * s
            - 448.0 * x**6 * c
            - 2016.0 * x**5 * s
            + 3920.0 * x**4 * c
            + 7000.0 * x**3 * s
            - 6804.0 * x**2 * c
            - 4634.0 * x * s
            + 1303.0 * c,
        ),
        8: (
            256.0 * x**8 * c
            + 1024.0 * x**7 * s
            - 5376.0 * x**6 * c
            - 12544.0 * x**5 * s
            + 28000.0 * x**4 * c
            + 36288.0 * x**3 * s
            - 37072.0 * x**2 * c
            - 20848.0 * x * s
            + 5937.0 * c,
            -256.0 * x**8 * s
            + 1024.0 * x**7 * c
            + 5376.0 * x**6 * s
            - 12544.0 * x**5 * c
            - 28000.0 * x**4 * s
            + 36288.0 * x**3 * c
            + 37072.0 * x**2 * s
            - 20848.0 * x * c
            - 5937.0 * s,
        ),
    }
    re_part, im_part = polys[n]
    return re_part * e / d, im_part * e / d


def _cmor(x, fb, fc):
    envelope = jnp.exp(-(x**2) / fb) / jnp.sqrt(jnp.pi * fb)
    return jnp.cos(2 * jnp.pi * fc * x) * envelope, jnp.sin(
        2 * jnp.pi * fc * x
    ) * envelope


def _shan(x, fb, fc):
    carrier_r = jnp.cos(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    carrier_i = jnp.sin(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    sinc_z = jnp.sinc(fb * x)
    return carrier_r * sinc_z, carrier_i * sinc_z


def _fbsp(x, m, fb, fc):
    carrier_r = jnp.cos(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    carrier_i = jnp.sin(2 * jnp.pi * fc * x) * jnp.sqrt(fb)
    sinc_z = jnp.sinc(x * fb / m) ** m
    return carrier_r * sinc_z, carrier_i * sinc_z


def _psi(w, x):
    name = w.name
    if name == "morl":
        return _morl(x)
    if name == "mexh":
        return _mexh(x)
    if name.startswith("gaus"):
        return _gaus(x, int(name[4:]))
    if name.startswith("cgau"):
        return _cgau(x, int(name[4:]))
    if name.startswith("cmor"):
        fb, fc = _parse_params(name, "cmor")
        return _cmor(x, fb, fc)
    if name.startswith("shan"):
        fb, fc = _parse_params(name, "shan")
        return _shan(x, fb, fc)
    if name.startswith("fbsp"):
        m, fb, fc = _parse_params(name, "fbsp")
        return _fbsp(x, m, fb, fc)


# --- Public utility functions ---


def wavefun(wavelet, precision=8):
    """Evaluate the wavelet function on a grid.

    Parameters
    ----------
    wavelet : str or ContinuousWavelet
        Wavelet to evaluate.
    precision : int
        Grid resolution as ``2**precision`` points. Default 8.

    Returns
    -------
    psi : array
        Wavelet function values (complex if the wavelet is complex).
    x : array
        Grid points.
    """
    w = as_wavelet(wavelet)
    x = jnp.linspace(w.lower_bound, w.upper_bound, 2**precision)
    psi = _psi(w, x)
    if w.complex_cwt:
        return psi[0] + 1j * psi[1], x
    return psi, x


def integrate_wavelet(wavelet, precision=8):
    """Compute the running integral of the wavelet function.

    Parameters
    ----------
    wavelet : str or ContinuousWavelet
        Wavelet to integrate.
    precision : int
        Grid resolution as ``2**precision`` points. Default 8.

    Returns
    -------
    int_psi : array
        Cumulative integral of the wavelet function.
    x : array
        Grid points.
    """
    w = as_wavelet(wavelet)
    x = jnp.linspace(w.lower_bound, w.upper_bound, 2**precision)
    step = x[1] - x[0]
    psi = _psi(w, x)
    if w.complex_cwt:
        return jnp.cumsum(psi[0]) * step + 1j * jnp.cumsum(psi[1]) * step, x
    return jnp.cumsum(psi) * step, x


def central_frequency(wavelet, precision=8):
    """Central frequency of a wavelet.

    Parameters
    ----------
    wavelet : str or ContinuousWavelet
        Wavelet to analyse.
    precision : int
        Grid resolution as ``2**precision`` points. Default 8.

    Returns
    -------
    float
        Central frequency in cycles per unit.
    """
    psi, x = wavefun(wavelet, precision)
    domain = x[-1] - x[0]
    index = jnp.argmax(jnp.abs(jnp.fft.fft(psi))[1:]) + 2
    index = jnp.where(index > psi.shape[0] / 2, psi.shape[0] - index + 2, index)
    return 1.0 / (domain / (index - 1))


def scale2frequency(wavelet, scale, precision=8):
    """Convert wavelet scale to frequency.

    Parameters
    ----------
    wavelet : str or ContinuousWavelet
        Wavelet to use.
    scale : float or array
        Scale(s) to convert.
    precision : int
        Grid resolution as ``2**precision`` points. Default 8.

    Returns
    -------
    float or array
        Frequency corresponding to each scale.
    """
    return central_frequency(wavelet, precision) / scale


# --- Two-phase CWT: prepare + apply ---


class CWTKernelBank:
    """Precomputed CWT convolution kernels.

    Parameters
    ----------
    kernels_r : array
        Real parts of the padded kernel matrix, shape ``(n_scales, Lmax)``.
    kernels_i : array
        Imaginary parts (empty if the wavelet is real).
    kernel_lengths : tuple of int
        Unpadded kernel length per scale (static).
    scales_sqrt : tuple of float
        Square root of each scale (static).
    complex_cwt : bool
        Whether the wavelet is complex (static).
    method : str
        Convolution method, 'conv' or 'fft' (static).
    freqs : array
        Pseudo-frequencies corresponding to each scale.

    Notes
    -----
    ``kernels_r``, ``kernels_i``, and ``freqs`` are JAX-traced;
    remaining fields are static. Registered as a JAX pytree node.
    """

    __slots__ = (
        "kernels_r",
        "kernels_i",
        "kernel_lengths",
        "scales_sqrt",
        "complex_cwt",
        "method",
        "freqs",
    )

    def __init__(
        self,
        kernels_r,
        kernels_i,
        kernel_lengths,
        scales_sqrt,
        complex_cwt,
        method,
        freqs,
    ):
        self.kernels_r = kernels_r
        self.kernels_i = kernels_i
        self.kernel_lengths = kernel_lengths
        self.scales_sqrt = scales_sqrt
        self.complex_cwt = complex_cwt
        self.method = method
        self.freqs = freqs


jax.tree_util.register_pytree_node(
    CWTKernelBank,
    lambda b: (
        (b.kernels_r, b.kernels_i, b.freqs),
        (b.kernel_lengths, b.scales_sqrt, b.complex_cwt, b.method),
    ),
    lambda aux, children: CWTKernelBank(
        children[0], children[1], aux[0], aux[1], aux[2], aux[3], children[2]
    ),
)


def prepare_cwt(scales, wavelet, sampling_period=1.0, method="conv", precision=12):
    """Build a CWT kernel bank for use with :func:`apply_cwt`.

    Parameters
    ----------
    scales : sequence of float
        Wavelet scales.
    wavelet : str or ContinuousWavelet
        Wavelet to use.
    sampling_period : float
        Sampling period of the input signal. Default 1.
    method : str
        Convolution method: 'conv' (direct) or 'fft'. Default 'conv'.
    precision : int
        Wavelet integration resolution as ``2**precision`` points.
        Default 12.

    Returns
    -------
    CWTKernelBank
        Precomputed kernel bank.
    """
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
    pad_i = (
        jnp.stack([jnp.pad(k, (0, Lmax - k.shape[0])) for k in kernels_i])
        if w.complex_cwt
        else jnp.empty((0,))
    )

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
    """Apply a precomputed CWT kernel bank to data.

    Parameters
    ----------
    data : array
        1D input signal.
    bank : CWTKernelBank
        Kernel bank from :func:`prepare_cwt`.

    Returns
    -------
    coefs : array
        CWT coefficient matrix, shape ``(n_scales, n_samples)``.
        Complex if the wavelet is complex.
    freqs : array
        Pseudo-frequencies for each scale.
    """
    N = data.shape[0]
    Lmax = bank.kernels_r.shape[1]
    n_scales = bank.kernels_r.shape[0]

    if bank.method == "fft":
        fft_size = 1 << int(math.ceil(math.log2(N + Lmax - 1)))
        data_fft = jnp.fft.rfft(data, fft_size)

    out_r, out_i = [], []
    for i in range(n_scales):
        L = bank.kernel_lengths[i]
        scale_sqrt = bank.scales_sqrt[i]
        conv_len = N + L - 1
        d = (conv_len - 1 - N) / 2.0
        lo = math.floor(d)
        hi = conv_len - 1 - math.ceil(d)

        if bank.method == "fft":
            conv_r = jnp.fft.irfft(
                data_fft * jnp.fft.rfft(bank.kernels_r[i], fft_size), fft_size
            )
        else:
            conv_r = jnp.convolve(data, bank.kernels_r[i])

        coef_r = -scale_sqrt * jnp.diff(conv_r[:conv_len])
        out_r.append(coef_r[lo:hi])

        if bank.complex_cwt:
            if bank.method == "fft":
                conv_i = jnp.fft.irfft(
                    data_fft * jnp.fft.rfft(bank.kernels_i[i], fft_size), fft_size
                )
            else:
                conv_i = jnp.convolve(data, bank.kernels_i[i])
            coef_i = -scale_sqrt * jnp.diff(conv_i[:conv_len])
            out_i.append(coef_i[lo:hi])

    if bank.complex_cwt:
        return jnp.stack(out_r) + 1j * jnp.stack(out_i), bank.freqs
    return jnp.stack(out_r), bank.freqs


def cwt(data, scales, wavelet, sampling_period=1.0, method="conv", precision=12):
    """1D continuous wavelet transform.

    Parameters
    ----------
    data : array
        1D input signal.
    scales : sequence of float
        Wavelet scales.
    wavelet : str or ContinuousWavelet
        Wavelet to use.
    sampling_period : float
        Sampling period. Default 1.
    method : str
        Convolution method: 'conv' or 'fft'. Default 'conv'.
    precision : int
        Wavelet integration resolution as ``2**precision`` points.
        Default 12.

    Returns
    -------
    coefs : array
        CWT coefficient matrix, shape ``(n_scales, n_samples)``.
    freqs : array
        Pseudo-frequencies for each scale.
    """
    bank = prepare_cwt(scales, wavelet, sampling_period, method, precision)
    return apply_cwt(data, bank)
