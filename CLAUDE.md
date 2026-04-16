# jaxwavelets

JAX-native wavelet transforms.

## Philosophy

- **Clean, lightweight, elegant code.** No overengineering. No defensive programming. No input validation, no guards, no try/except. Let it crash. This is scientific code, not web development.
- **Implement from the mathematical definition**, not as a port of another library. pywt is a numerical reference for testing, not a specification.
- **Single-example functions.** Users compose with `jax.vmap`, `jax.pmap`, `jax.grad`, `jax.jit` from outside. The library does not jit or batch internally.
- **Structural vmap is permitted** for separable nD transforms (applying a 1D operation along each axis of a single example). This is part of the nD algorithm, not batching.
- **NamedTuples for simple state** (e.g. `Wavelet` filter bank). Custom pytree registration when static metadata must be separated from traced array data (e.g. `WaveletCoeffs` where reconstruction shapes are static aux_data).
- **No pywt runtime dependency.** Filter coefficients are extracted at build time.

## Testing

```bash
pytest jaxwavelets/tests/test_against_pywt.py
```

Requires `pywt` as a test dependency. Tests verify numerical match with pywt to `atol=1e-11` on CPU float64.

## Generating filter coefficients

```bash
python scripts/extract_filters.py > jaxwavelets/_filters.py
```
