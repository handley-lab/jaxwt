# jaxwavelets

JAX-native wavelet transforms extending PyWavelets to JAX.

## When to use

Use when the user needs wavelet transforms in JAX — differentiable, JIT-compilable, GPU-ready. Triggers on: 'wavelet', 'jaxwavelets', 'wavedec', 'waverec', 'dwt', 'idwt', 'cwt', 'swt', 'wavelet packet', 'multiresolution', 'MRA'.

## Quick reference

```python
import jaxwavelets as wt

# DWT
cA, cD = wt.dwt(x, 'db4')
x_rec = wt.idwt(cA, cD, 'db4')

# Multilevel nD
coeffs = wt.wavedecn(x, 'db4', level=3)
x_rec = wt.waverecn(coeffs, 'db4')

# 2D convenience
coeffs = wt.wavedec2(x, 'db4', level=3)
x_rec = wt.waverec2(coeffs, 'db4')

# SWT (shift-invariant)
coeffs = wt.swt(x, 'db4', level=3)
x_rec = wt.iswt(coeffs, 'db4')

# CWT (two-phase for JIT)
bank = wt.prepare_cwt(scales, 'morl')
coefs, freqs = wt.apply_cwt(data, bank)  # JIT-safe

# Composability
import jax
from functools import partial
jax.vmap(partial(wt.wavedecn, wavelet='db4', level=3))(batch)
jax.grad(lambda x: jnp.sum(wt.waverecn(wt.wavedecn(x, 'db4'), 'db4')))(x)
jax.jit(wt.wavedecn, static_argnames=['wavelet', 'mode', 'level'])(x, wavelet='db4')
```

## Available wavelets

- **Discrete:** haar, db1-20, sym2-20, coif1-5
- **Continuous:** morl, mexh, gaus1-8, cgau1-8, cmor, shan, fbsp

## Key design points

- Single-example functions — user applies `vmap`/`pmap`/`grad`/`jit` from outside
- No complex arithmetic internally (GPU-safe)
- No numpy dependency at runtime
- Coefficients are JAX pytrees (`WaveletCoeffs`, `CWTKernelBank`)
- Modes: `'symmetric'`, `'reflect'`, `'periodization'`
- CWT needs `prepare_cwt()` outside JIT, `apply_cwt()` inside JIT
