"""jaxwt: JAX-native wavelet transforms."""

from jaxwt._filters import Wavelet, get_wavelet, qmf, orthogonal_filter_bank
from jaxwt._dwt import dwt, idwt, dwt_max_level, downcoef, upcoef
from jaxwt._multidim import (
    dwtn,
    idwtn,
    wavedecn,
    waverecn,
    WaveletCoeffs,
    dwt2,
    idwt2,
    wavedec2,
    waverec2,
)
from jaxwt._swt import swt, iswt, swtn, iswtn, swt2, iswt2, swt_max_level
from jaxwt._thresholding import threshold, threshold_firm
from jaxwt._fswt import fswavedecn, fswaverecn, FswavedecnResult
from jaxwt._mra import mra, imra, mra2, imra2, mran, imran
from jaxwt._cwt import (
    cwt,
    prepare_cwt,
    apply_cwt,
    wavefun,
    integrate_wavelet,
    central_frequency,
    scale2frequency,
    ContinuousWavelet,
    CWTKernelBank,
)
from jaxwt._packets import (
    wp_decompose,
    wp_reconstruct,
    wp_decompose_nd,
    wp_reconstruct_nd,
)
