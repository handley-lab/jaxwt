"""Benchmark: JIT'd jaxwavelets vs pywt C on CPU.

Run with: python benchmarks/benchmark_vs_pywt.py

Measures execution time excluding JIT compilation overhead.
Each transform is warmed up (5 calls), then timed over 200 repetitions.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"
import time

import jax
import jax.numpy as jnp
import numpy as np
import pywt
from wt._cwt import apply_cwt, prepare_cwt
from wt._swt import swt

import jaxwavelets as wt

jax.config.update("jax_platform_name", "cpu")


def bench(fn, n_warmup=5, n_repeat=200):
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return np.mean(times), np.std(times)


x1d = np.random.RandomState(0).randn(4096)
x2d = np.random.RandomState(0).randn(256, 256)
x1d_j = jnp.array(x1d)
x2d_j = jnp.array(x2d)
x_swt = jnp.array(x1d[:1024])
x_cwt = jnp.array(x1d[:512])

dwt_jit = jax.jit(lambda x: wt.dwt(x, "db4"))
wavedecn_1d_jit = jax.jit(lambda x: wt.wavedecn(x, "db4"))
dwt2_jit = jax.jit(lambda x: wt.dwt2(x, "db4"))
wavedecn_2d_jit = jax.jit(lambda x: wt.wavedecn(x, "db4", level=3))
swt_jit = jax.jit(lambda x: swt(x, "db4", level=3))

bank_morl = prepare_cwt((1.0, 2.0, 4.0, 8.0, 16.0, 32.0), "morl")
bank_cmor = prepare_cwt((1.0, 2.0, 4.0, 8.0, 16.0, 32.0), "cmor1.5-1.0")
apply_jit = jax.jit(apply_cwt)

tests = [
    ("dwt 1D (N=4096)", lambda: pywt.dwt(x1d, "db4"), lambda: dwt_jit(x1d_j)),
    (
        "wavedecn 1D (N=4096)",
        lambda: pywt.wavedecn(x1d.reshape(-1), "db4"),
        lambda: wavedecn_1d_jit(x1d_j),
    ),
    ("dwt2 (256x256)", lambda: pywt.dwt2(x2d, "db4"), lambda: dwt2_jit(x2d_j)),
    (
        "wavedecn 2D level=3",
        lambda: pywt.wavedecn(x2d, "db4", level=3),
        lambda: wavedecn_2d_jit(x2d_j),
    ),
    (
        "swt 1D level=3 (N=1024)",
        lambda: pywt.swt(np.array(x1d[:1024]), "db4", level=3),
        lambda: swt_jit(x_swt),
    ),
    (
        "cwt morl 6 scales (N=512)",
        lambda: pywt.cwt(
            np.array(x1d[:512]), np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]), "morl"
        ),
        lambda: apply_jit(x_cwt, bank_morl),
    ),
    (
        "cwt cmor 6 scales (N=512)",
        lambda: pywt.cwt(
            np.array(x1d[:512]),
            np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
            "cmor1.5-1.0",
        ),
        lambda: apply_jit(x_cwt, bank_cmor),
    ),
]

print(f"{'Transform':<30} {'pywt':>16} {'jaxwavelets (JIT)':>16} {'ratio':>8}")
print("-" * 74)

for name, fn_pywt, fn_jax in tests:
    m_p, s_p = bench(fn_pywt)
    m_j, s_j = bench(fn_jax)
    ratio = m_j / m_p
    print(f"{name:<30} {m_p:7.3f}±{s_p:5.3f}ms {m_j:7.3f}±{s_j:5.3f}ms {ratio:7.1f}x")
