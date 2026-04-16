# jaxwavelets

Extending PyWavelets to JAX. `import jaxwavelets as wt`.

## Philosophy

- **Clean, lightweight, elegant code.** No overengineering. No defensive programming. No input validation, no guards, no try/except. Let it crash. This is scientific code, not web development.
- **Implement from the mathematical definition**, not as a port of another library. PyWavelets is a numerical reference for testing, not a specification.
- **Single-example functions.** Users compose with `jax.vmap`, `jax.pmap`, `jax.grad`, `jax.jit` from outside. The library does not jit or batch internally.
- **Structural vmap is permitted** for separable nD transforms (applying a 1D operation along each axis of a single example). This is part of the nD algorithm, not batching.
- **NamedTuples for simple state** (e.g. `Wavelet` filter bank). Custom pytree registration when static metadata must be separated from traced array data (e.g. `WaveletCoeffs` where reconstruction shapes are static aux_data).
- **No PyWavelets runtime dependency.** Filter coefficients are extracted at build time.
- **No complex arithmetic internally.** Separate real/imaginary arrays for GPU performance.
- **No numpy in library code.** Pure JAX throughout.

## Conventions

- Package alias: `import jaxwavelets as wt`
- Wavelet naming: `cA`, `cD`, `cH`, `cV` (standard wavelet convention, not snake_case)
- Linting: ruff with N-series ignores for wavelet naming

## Testing

```bash
pytest jaxwavelets/tests/
```

Requires `pywt` as a test dependency. Tests verify numerical match with PyWavelets to `atol=1e-14` (direct comparisons) and `atol=1e-11` (roundtrips).

## Generating filter coefficients

```bash
python scripts/extract_filters.py > jaxwavelets/_filters.py
```

## Version management

```bash
python scripts/bump_version.py patch  # bump version
git tag v0.1.1 && git push --tags     # publish to PyPI via CI
```
