API Reference
=============

Discrete Wavelet Transform
--------------------------

.. autofunction:: jaxwavelets.dwt
.. autofunction:: jaxwavelets.idwt
.. autofunction:: jaxwavelets.dwt2
.. autofunction:: jaxwavelets.idwt2
.. autofunction:: jaxwavelets.dwtn
.. autofunction:: jaxwavelets.idwtn

Multilevel DWT
--------------

.. autofunction:: jaxwavelets.wavedec2
.. autofunction:: jaxwavelets.waverec2
.. autofunction:: jaxwavelets.wavedecn
.. autofunction:: jaxwavelets.waverecn

.. autoclass:: jaxwavelets.WaveletCoeffs
   :members:

Stationary Wavelet Transform
-----------------------------

.. autofunction:: jaxwavelets.swt
.. autofunction:: jaxwavelets.iswt
.. autofunction:: jaxwavelets.swt2
.. autofunction:: jaxwavelets.iswt2
.. autofunction:: jaxwavelets.swtn
.. autofunction:: jaxwavelets.iswtn

Continuous Wavelet Transform
-----------------------------

.. autofunction:: jaxwavelets.cwt
.. autofunction:: jaxwavelets.prepare_cwt
.. autofunction:: jaxwavelets.apply_cwt
.. autofunction:: jaxwavelets.wavefun
.. autofunction:: jaxwavelets.integrate_wavelet
.. autofunction:: jaxwavelets.central_frequency
.. autofunction:: jaxwavelets.scale2frequency

.. autoclass:: jaxwavelets.ContinuousWavelet
   :members:

.. autoclass:: jaxwavelets.CWTKernelBank
   :members:

Fully Separable DWT
-------------------

.. autofunction:: jaxwavelets.fswavedecn
.. autofunction:: jaxwavelets.fswaverecn

.. autoclass:: jaxwavelets.FswavedecnResult
   :members:

Multiresolution Analysis
------------------------

.. autofunction:: jaxwavelets.mra
.. autofunction:: jaxwavelets.imra
.. autofunction:: jaxwavelets.mra2
.. autofunction:: jaxwavelets.imra2
.. autofunction:: jaxwavelets.mran
.. autofunction:: jaxwavelets.imran

Wavelet Packets
---------------

.. autofunction:: jaxwavelets.wp_decompose
.. autofunction:: jaxwavelets.wp_reconstruct
.. autofunction:: jaxwavelets.wp_decompose_nd
.. autofunction:: jaxwavelets.wp_reconstruct_nd

Thresholding
------------

.. autofunction:: jaxwavelets.soft_threshold
.. autofunction:: jaxwavelets.hard_threshold
.. autofunction:: jaxwavelets.garrote_threshold
.. autofunction:: jaxwavelets.firm_threshold

Utilities
---------

.. autofunction:: jaxwavelets.dwt_max_level
.. autofunction:: jaxwavelets.downcoef
.. autofunction:: jaxwavelets.upcoef
.. autofunction:: jaxwavelets.qmf
.. autofunction:: jaxwavelets.orthogonal_filter_bank
.. autofunction:: jaxwavelets.get_wavelet
.. autofunction:: jaxwavelets.swt_max_level

.. autoclass:: jaxwavelets.Wavelet
   :members:
