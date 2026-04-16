"""jaxwavelets: JAX-native wavelet transforms."""

from jaxwavelets._cwt import (
    ContinuousWavelet,
    CWTKernelBank,
    apply_cwt,
    central_frequency,
    cwt,
    integrate_wavelet,
    prepare_cwt,
    scale2frequency,
    wavefun,
)
from jaxwavelets._dwt import downcoef, dwt, dwt_max_level, idwt, upcoef
from jaxwavelets._filters import Wavelet, get_wavelet, orthogonal_filter_bank, qmf
from jaxwavelets._fswt import FswavedecnResult, fswavedecn, fswaverecn
from jaxwavelets._mra import imra, imra2, imran, mra, mra2, mran
from jaxwavelets._multidim import (
    WaveletCoeffs,
    dwt2,
    dwtn,
    idwt2,
    idwtn,
    wavedec2,
    wavedecn,
    waverec2,
    waverecn,
)
from jaxwavelets._packets import (
    wp_decompose,
    wp_decompose_nd,
    wp_reconstruct,
    wp_reconstruct_nd,
)
from jaxwavelets._swt import iswt, iswt2, iswtn, swt, swt2, swt_max_level, swtn
from jaxwavelets._thresholding import (
    firm_threshold,
    garrote_threshold,
    hard_threshold,
    soft_threshold,
)
