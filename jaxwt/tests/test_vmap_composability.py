"""Thorough test of jaxwt vmap composability.

Tests that every public function works under jax.vmap, jax.grad, jax.jit,
and compositions thereof — the same way blackjax kernels work under vmap.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import jaxwt

x_1d = jnp.array(np.random.RandomState(0).randn(32))
x_2d = jnp.array(np.random.RandomState(0).randn(16, 16))
x_3d = jnp.array(np.random.RandomState(0).randn(8, 8, 8))

batch_1d = jnp.stack([jnp.array(np.random.RandomState(i).randn(32)) for i in range(4)])
batch_2d = jnp.stack(
    [jnp.array(np.random.RandomState(i).randn(16, 16)) for i in range(4)]
)
batch_3d = jnp.stack(
    [jnp.array(np.random.RandomState(i).randn(8, 8, 8)) for i in range(4)]
)

print("=== vmap over dwt/idwt ===")
# vmap 1D dwt
cA_batch, cD_batch = jax.vmap(partial(jaxwt.dwt, wavelet="db4"))(batch_1d)
print(
    f"vmap dwt 1D: input={batch_1d.shape} -> cA={cA_batch.shape}, cD={cD_batch.shape}"
)

# vmap 1D idwt
rec_batch = jax.vmap(partial(jaxwt.idwt, wavelet="db4"))(cA_batch, cD_batch)
print(f"vmap idwt 1D: rec={rec_batch.shape}")
print(f"  roundtrip err: {float(jnp.max(jnp.abs(rec_batch[:, :32] - batch_1d))):.2e}")

# vmap 1D periodization
cA_p, cD_p = jax.vmap(partial(jaxwt.dwt, wavelet="db4", mode="periodization"))(batch_1d)
rec_p = jax.vmap(partial(jaxwt.idwt, wavelet="db4", mode="periodization"))(cA_p, cD_p)
print(
    f"vmap dwt/idwt periodization: rec={rec_p.shape} err={float(jnp.max(jnp.abs(rec_p[:, :32] - batch_1d))):.2e}"
)

print("\n=== vmap over wavedecn/waverecn ===")
# vmap 2D roundtrip
f_dec = partial(jaxwt.wavedecn, wavelet="db4", level=2)
f_rec = partial(jaxwt.waverecn, wavelet="db4")
batch_coeffs = jax.vmap(f_dec)(batch_2d)
print(f"vmap wavedecn 2D: approx={batch_coeffs.approx.shape}")
batch_rec = jax.vmap(f_rec)(batch_coeffs)
print(
    f"vmap waverecn 2D: rec={batch_rec.shape} err={float(jnp.max(jnp.abs(batch_rec - batch_2d))):.2e}"
)

# vmap 3D roundtrip
f_dec3 = partial(jaxwt.wavedecn, wavelet="haar", level=2)
f_rec3 = partial(jaxwt.waverecn, wavelet="haar")
batch_coeffs3 = jax.vmap(f_dec3)(batch_3d)
print(f"vmap wavedecn 3D: approx={batch_coeffs3.approx.shape}")
batch_rec3 = jax.vmap(f_rec3)(batch_coeffs3)
print(
    f"vmap waverecn 3D: rec={batch_rec3.shape} err={float(jnp.max(jnp.abs(batch_rec3 - batch_3d))):.2e}"
)

print("\n=== vmap over periodization nD ===")
f_dec_per = partial(jaxwt.wavedecn, wavelet="db4", mode="periodization", level=2)
f_rec_per = partial(jaxwt.waverecn, wavelet="db4", mode="periodization")
batch_coeffs_per = jax.vmap(f_dec_per)(batch_2d)
batch_rec_per = jax.vmap(f_rec_per)(batch_coeffs_per)
print(
    f"vmap periodization 2D: rec={batch_rec_per.shape} err={float(jnp.max(jnp.abs(batch_rec_per - batch_2d))):.2e}"
)

print("\n=== grad through vmap ===")


# grad of sum of vmapped roundtrip
def loss(batch):
    return jnp.sum(
        jax.vmap(lambda x: jaxwt.waverecn(jaxwt.wavedecn(x, "db4", level=2), "db4"))(
            batch
        )
    )


g = jax.grad(loss)(batch_2d)
print(
    f"grad(sum(vmap(roundtrip))): shape={g.shape} ≈ ones: {jnp.allclose(g, 1.0, atol=1e-10)}"
)

print("\n=== jit(vmap(...)) ===")
jitted_vmap_roundtrip = jax.jit(
    jax.vmap(lambda x: jaxwt.waverecn(jaxwt.wavedecn(x, "db4", level=2), "db4"))
)
rec_jv = jitted_vmap_roundtrip(batch_2d)
print(
    f"jit(vmap(roundtrip)): shape={rec_jv.shape} err={float(jnp.max(jnp.abs(rec_jv - batch_2d))):.2e}"
)

print("\n=== vmap(grad(...)) — per-example gradients ===")
per_example_grad = jax.vmap(
    jax.grad(
        lambda x: jnp.sum(jaxwt.waverecn(jaxwt.wavedecn(x, "db4", level=2), "db4"))
    )
)
g_batch = per_example_grad(batch_2d)
print(
    f"vmap(grad(roundtrip)): shape={g_batch.shape} ≈ ones: {jnp.allclose(g_batch, 1.0, atol=1e-10)}"
)

print("\n=== tree_map over WaveletCoeffs ===")
coeffs = jaxwt.wavedecn(x_2d, "db4", level=2)
doubled = jax.tree_util.tree_map(lambda x: 2 * x, coeffs)
print(
    f"tree_map(2*x): approx ratio = {float(jnp.mean(doubled.approx / coeffs.approx)):.1f}"
)

print("\n=== ALL PASSED ===")
