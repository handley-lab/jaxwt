jaxwavelets
=====

JAX-native wavelet transforms. Differentiable, JIT-compilable, GPU-ready.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api


Installation
------------

.. code-block:: bash

   pip install jax jaxlib jaxwavelets


Quick Start
-----------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import jaxwavelets

   # Decompose
   x = jnp.ones((64, 64))
   coeffs = jaxwavelets.wavedecn(x, 'db4', level=3)

   # Reconstruct
   rec = jaxwavelets.waverecn(coeffs, 'db4')

   # Batch via vmap
   from functools import partial
   batch_transform = jax.vmap(partial(jaxwavelets.wavedecn, wavelet='db4', level=3))

   # Differentiate
   grad = jax.grad(lambda x: jnp.sum(jaxwavelets.waverecn(jaxwavelets.wavedecn(x, 'db4'), 'db4')))(x)
