# jaxwavelets

[![PyPI](https://img.shields.io/pypi/v/jaxwavelets)](https://pypi.org/project/jaxwavelets/)
[![CI](https://github.com/handley-lab/jaxwavelets/actions/workflows/ci.yml/badge.svg)](https://github.com/handley-lab/jaxwavelets/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/jaxwavelets)](https://pypi.org/project/jaxwavelets/)
[![codecov](https://codecov.io/gh/handley-lab/jaxwavelets/graph/badge.svg)](https://codecov.io/gh/handley-lab/jaxwavelets)

Extending [PyWavelets](https://pywavelets.readthedocs.io/) to [JAX](https://jax.readthedocs.io/). Differentiable, JIT-compilable, GPU-ready wavelet transforms.

Built on the mathematical foundations of PyWavelets and validated against it to machine precision. jaxwavelets brings the full PyWavelets API to JAX, enabling automatic differentiation, GPU acceleration, and composability with `jax.vmap`, `jax.jit`, and `jax.pmap`.

## Features

| Transform | Functions |
|---|---|
| Discrete wavelet | `dwt`, `idwt`, `dwt2`, `idwt2`, `dwtn`, `idwtn` |
| Multilevel | `wavedec2`, `waverec2`, `wavedecn`, `waverecn` |
| Stationary (undecimated) | `swt`, `iswt`, `swt2`, `iswt2`, `swtn`, `iswtn` |
| Continuous | `cwt`, `prepare_cwt`, `apply_cwt` |
| Fully separable | `fswavedecn`, `fswaverecn` |
| Multiresolution analysis | `mra`, `imra`, `mra2`, `imra2`, `mran`, `imran` |
| Wavelet packets | `wp_decompose`, `wp_reconstruct`, `wp_decompose_nd`, `wp_reconstruct_nd` |
| Thresholding | `soft_threshold`, `hard_threshold`, `garrote_threshold`, `firm_threshold` |
| Utilities | `downcoef`, `upcoef`, `qmf`, `orthogonal_filter_bank` |

**Wavelets:** haar, db1-20, sym2-20, coif1-5, plus continuous wavelets (Morlet, Mexican hat, Gaussian 1-8, complex Gaussian 1-8, complex Morlet, Shannon, frequency B-spline).

## Usage

```python
import jax
import jax.numpy as jnp
import jaxwavelets as wt

# Decompose and reconstruct
x = jnp.ones((64, 64))
coeffs = wt.wavedecn(x, 'db4', level=3)
rec = wt.waverecn(coeffs, 'db4')

# Batch via vmap
from functools import partial
batch = jnp.ones((10, 64, 64))
batch_coeffs = jax.vmap(partial(wt.wavedecn, wavelet='db4', level=3))(batch)

# Differentiate through the transform
grad = jax.grad(lambda x: jnp.sum(wt.waverecn(wt.wavedecn(x, 'db4'), 'db4')))(x)

# JIT-compile for speed
fast = jax.jit(wt.wavedecn, static_argnames=['wavelet', 'mode', 'level'])
coeffs = fast(x, wavelet='db4', level=3)
```

## Performance

JIT-compiled jaxwavelets on CPU vs PyWavelets C:

```
Transform                       pywt         jaxwavelets (JIT)    ratio
--------------------------------------------------------------------------
dwt 1D (N=4096)                  0.011ms       0.023ms       2.1x
wavedecn 1D (N=4096)             0.065ms       0.046ms       0.7x  ← faster
dwt2 (256x256)                   0.608ms       0.287ms       0.5x  ← faster
wavedecn 2D level=3              0.755ms       0.363ms       0.5x  ← faster
swt 1D level=3 (N=1024)          0.023ms       0.025ms       1.1x
cwt morl 6 scales (N=512)        0.316ms       0.139ms       0.4x  ← faster
cwt cmor 6 scales (N=512)        0.615ms       0.254ms       0.4x  ← faster
```

On top of this, jaxwavelets supports `jax.grad`, `jax.vmap`, `jax.pmap`, and GPU acceleration.

## Installation

```bash
pip install jaxwavelets
```

No runtime dependency on PyWavelets. Filter coefficients are pre-extracted.

## Testing

```bash
pip install pywt pytest
pytest jaxwavelets/tests/
```

1189 tests verify numerical agreement with PyWavelets to machine precision.

## Composability

Every function operates on a single example. Batching, differentiation, compilation, and distribution compose naturally via JAX transforms:

```python
import jaxwavelets as wt

# Batch over examples
jax.vmap(partial(wt.wavedecn, wavelet='db4'))(batch_of_fields)

# Per-example gradients
jax.vmap(jax.grad(loss_fn))(batch)

# Distribute across devices
jax.pmap(partial(wt.wavedecn, wavelet='db4'))(sharded_data)

# Nest arbitrarily
jax.jit(jax.vmap(jax.grad(
    lambda x: jnp.sum(wt.waverecn(wt.wavedecn(x, 'db4'), 'db4'))
)))(batch)
```

Coefficients are JAX pytrees, so `jax.tree_util.tree_map` works directly on them.

## Design

- **Pure JAX** — no numpy, no C extensions
- **Single-example functions** — compose with `jax.vmap`/`jax.pmap`/`jax.grad`/`jax.jit`
- **Pytree coefficients** — all outputs are JAX-compatible pytrees
- **Validated against PyWavelets** — machine-precision numerical agreement

## Acknowledgements

jaxwavelets extends the [PyWavelets](https://pywavelets.readthedocs.io/) library to JAX. PyWavelets provides the mathematical reference implementation and filter coefficient database used for validation.
