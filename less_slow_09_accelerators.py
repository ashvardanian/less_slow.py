#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Accelerators — narrow types, and when other silicon starts to pay.

A matrix product is the same arithmetic whatever the element type, so cost
ought to track the width. It does not. NumPy's `float16` is nearly a thousand
times slower than its `float32` — narrower, and no BLAS kernel behind it, so
it falls to a generic loop. Narrowing a type without a kernel is a
pessimization.

Which is the opening for a library that has those kernels: NumKong's bfloat16
runs several times faster than NumPy's float32 on the same CPU, at a fraction
of a degree of angular error. Width does not predict speed; kernel coverage
does.

Then the GPU, where the honest answer is a crossover rather than a number.
Compute grows with n³ and host-device transfer with n², so their ratio grows
with n: below roughly n=1500 the accelerator loses outright, and above it the
advantage compounds. Accuracy is reported as the angle between each computed
row and the exact one, which is scale-free — a row uniformly rescaled scores
zero error, a row pointing the wrong way does not.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from less_slow_00_shared import (
    MatmulQuantizationScales,
    cublas_matmul,
    gpu_matmul_usable,
    ml_dtypes,
    nk,
    nvmath,
)

# region: Numeric Types

# ? Before any question about speed, one about correctness. NumPy accumulates
# ? an int8 dot product *in int8*, so a long one wraps silently — the result is
# ? not approximate, it is wrong, and often the wrong sign. Nothing raises.
# ?
# ? This is the cheapest lesson in the chapter and the one most likely to be
# ? already in your code: quantize a model to int8, dot the vectors, and NumPy
# ? hands back garbage with total confidence.

# region: Integer Overflow

VECTOR_DIMS = 1536  # long enough that an int8 accumulator must wrap


@pytest.mark.benchmark(group="09-accelerators-integer-overflow")
def test_lowprec_int8_dot_numpy(benchmark):
    """NumPy keeps the accumulator in int8, so a long dot product overflows."""
    rng = np.random.default_rng(42)
    first = rng.integers(-128, 127, VECTOR_DIMS, dtype=np.int8)
    second = rng.integers(-128, 127, VECTOR_DIMS, dtype=np.int8)
    exact = int(first.astype(np.int64) @ second.astype(np.int64))

    def kernel():
        return int(np.dot(first, second))

    result = benchmark(kernel)
    benchmark.extra_info["exact"] = exact
    benchmark.extra_info["reported"] = result
    # ! Asserting the *wrong* answer, so this fails the day NumPy widens it.
    assert result != exact


@pytest.mark.benchmark(group="09-accelerators-integer-overflow")
def test_lowprec_int8_dot_numkong(benchmark):
    """NumKong widens the accumulator, so the same inputs come out right."""
    rng = np.random.default_rng(42)
    first = rng.integers(-128, 127, VECTOR_DIMS, dtype=np.int8)
    second = rng.integers(-128, 127, VECTOR_DIMS, dtype=np.int8)
    exact = int(first.astype(np.int64) @ second.astype(np.int64))

    def kernel():
        return int(nk.dot(first, second))

    result = benchmark(kernel)
    assert result == exact


# endregion: Integer Overflow

# region: Matrix Multiplication by Type

# ? A matrix product is the same arithmetic whatever the element type, so the
# ? cost ought to follow the width: half the bits, half the work. It does not.
# ? What decides the speed is whether anyone wrote a kernel for that type.
# ?
# ? Integer inputs throughout, so every type below represents the data exactly
# ? and the only variable is the kernel. Accuracy gets its own table further
# ? down, where it is the point rather than a confound.

MATRIX_SIDE = 1024
THREADS = 8


def _integer_operands(dtype, side: int = MATRIX_SIDE):
    """Small integers, exactly representable in every type compared here."""
    rng = np.random.default_rng(42)
    left = rng.integers(0, 10, size=(side, side)).astype(dtype)
    right = rng.integers(0, 10, size=(side, side)).astype(dtype)
    return left, np.asfortranarray(right)


@pytest.mark.benchmark(group="09-accelerators-dtype-numpy")
@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.float16, np.int32, np.int16]
)
def test_dtype_numpy(benchmark, dtype):
    """NumPy across types — BLAS covers two of these five."""
    left, right = _integer_operands(dtype)

    def kernel():
        return left @ right

    assert benchmark(kernel).shape == (MATRIX_SIDE, MATRIX_SIDE)


# ? Half precision is the row worth staring at. It is narrower than float32,
# ? and it is the slowest of all five — there is no BLAS kernel for it, so
# ? NumPy falls back to a generic loop. Narrowing the type made it slower.
# ?
# ? Which is the opening for a library that does have those kernels. NumKong
# ? implements bfloat16 and e4m3 directly, and its batched kernels release the
# ? GIL, so they scale across threads on a free-threaded build.


def _float_operands(side: int = MATRIX_SIDE):
    """Well-conditioned floats, with an exact float64 product to score against."""
    rng = np.random.default_rng(0)
    left = rng.standard_normal((side, side), dtype=np.float32)
    right = rng.standard_normal((side, side), dtype=np.float32)
    reference = np.empty((side, side), dtype=np.float64)
    # ! `dots_packed` computes A @ B.T, so the reference must too.
    np.matmul(left.astype(np.float64), right.T.astype(np.float64), out=reference)
    return left, right, reference


def _row_angles(product, reference):
    """Angle between each computed row and the exact one, in degrees.

    Cosine similarity, which is scale-free by construction: a row uniformly
    off by 1000x scores 0°, a row pointing the wrong way scores badly. That is
    the right question for quantized arithmetic, where per-row scaling is
    undone afterwards and only direction survives.
    """
    dots = np.einsum("ij,ij->i", product, reference)
    norms = np.linalg.norm(product, axis=1) * np.linalg.norm(reference, axis=1)
    usable = np.isfinite(norms) & (norms > 0)
    cosine = np.where(usable, dots / np.where(usable, norms, 1.0), -1.0)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


@pytest.fixture(scope="module")
def matmul_threads():
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(int, range(THREADS)))  # warm the threads
        yield pool


@pytest.mark.benchmark(group="09-accelerators-dtype-numkong")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "e4m3"])
def test_dtype_numkong(benchmark, dtype_name, matmul_threads):
    """NumKong across types — narrower really is faster when a kernel exists."""
    left, right, reference = _float_operands()
    limit = ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max
    if dtype_name == "e4m3":
        # ! `.astype` turns anything past 448 into NaN rather than saturating,
        # ! so the clip is mandatory, not defensive.
        scale = 100.0
        cast = lambda m: np.clip(m * scale, -limit, limit).astype(  # noqa: E731
            ml_dtypes.float8_e4m3fn
        )
        undo = scale * scale
    else:
        target = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}[dtype_name]
        cast = lambda m: m.astype(target)  # noqa: E731
        undo = 1.0

    packed = nk.dots_pack(cast(right))
    query = cast(left)
    probe = np.asarray(nk.dots_packed(query, packed))
    out = nk.Tensor(np.empty(probe.shape, dtype=probe.dtype))
    span = (MATRIX_SIDE + THREADS - 1) // THREADS

    def kernel():
        # ? Packing is setup; only the product is timed. NumKong has no thread
        # ? pool of its own — you slice the rows and bring your own.
        list(
            matmul_threads.map(
                lambda index: nk.dots_packed(
                    query,
                    packed,
                    out=out,
                    start_row=index * span,
                    end_row=min(MATRIX_SIDE, (index + 1) * span),
                ),
                range(THREADS),
            )
        )
        return out

    benchmark(kernel)
    angles = _row_angles(np.asarray(out).astype(np.float64) / undo, reference)
    benchmark.extra_info.update(
        angle_p50=float(np.median(angles)), angle_max=float(angles.max())
    )
    assert angles.max() < 5.0


# ? Intel Xeon 4 · CPython 3.14t · 1024×1024, NumKong on 8 threads
# ? Ratios are against NumPy float32, the type most of this would otherwise
# ? be written in. Note this table reports throughput, so bigger is better —
# ? the reverse of every other table in the file.
# ?
# ? NumPy, where BLAS covers two of the five:
# ?   float32   0.3859 TF/s    1.00x     exact
# ?   float64   0.2977 TF/s    0.77x     exact
# ?   int32     0.0043 TF/s    0.01x     exact
# ?   int16     0.0041 TF/s    0.01x     exact
# ?   float16   0.0004 TF/s   0.001x     exact
# ?
# ? NumKong, which has kernels for the narrow types NumPy lacks:
# ?   bfloat16  1.4667 TF/s    3.80x    0.135°
# ?   e4m3      1.2394 TF/s    3.21x    2.142°
# ?   float32   0.1021 TF/s    0.26x     exact
# ?
# ? Read down the NumPy block and it falls off a cliff — nearly four orders
# ? of magnitude from top to bottom. float16 is 965x slower than float32,
# ? the same arithmetic in
# ? half the bits, because there is no BLAS kernel for it and NumPy drops to a
# ? generic loop. Narrowing a type without a kernel behind it is a
# ? pessimization, and the integer rows say the same thing more quietly.
# ?
# ? Read down the NumKong block and the order inverts: its narrowest types are
# ? its fastest, and its float32 — the one type OpenBLAS also implements —
# ? loses by 4x. Neither library is faster than the other. Each is faster
# ? where someone wrote the kernel, which is the whole lesson.
# ?
# ? bfloat16 is the row to take away: 3.8x NumPy's float32 on the same silicon
# ? at a seventh of a degree of error. It spends its 16 bits on exponent
# ? rather than mantissa, which is why it survives activations that overflow
# ? float16, and why the industry trains in it. e4m3 halves the width again
# ? for 16x the angle and no more speed — at this size the kernel is already
# ? saturated, so the extra bits were free.

# endregion: Matrix Multiplication by Type

# region: Scaling Laws

# ? Everything above ran on one machine. Adding a GPU introduces a second
# ? cost that has nothing to do with arithmetic: the operands have to get
# ? there and the answer has to come back.
# ?
# ? That transfer grows with the *area* of the matrices while the arithmetic
# ? grows with their volume, so the two costs scale differently and the
# ? comparison has no single answer — only a crossover.

SCALING_SIDES = [512, 1024, 2048, 4096]


@pytest.mark.benchmark(group="09-accelerators-scaling-cpu")
@pytest.mark.parametrize("side", SCALING_SIDES)
def test_scaling_cpu(benchmark, side: int):
    """NumPy float32 on the CPU — compute-bound, so cost grows with n³."""
    left, right = _integer_operands(np.float32, side)

    def kernel():
        return left @ right

    benchmark.extra_info["gflop"] = 2 * side**3 / 1e9
    assert benchmark(kernel).shape == (side, side)


@pytest.mark.skipif(not gpu_matmul_usable, reason="no usable cuBLAS GPU")
@pytest.mark.benchmark(group="09-accelerators-scaling-gpu")
@pytest.mark.parametrize("side", SCALING_SIDES)
def test_scaling_gpu(benchmark, side: int):
    """cuBLASLt e4m3 — transfer-bound from NumPy, so cost grows with n²."""
    limit = ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max
    left, right = _integer_operands(np.float32, side)
    query = np.clip(left, -limit, limit).astype(ml_dtypes.float8_e4m3fn)
    packed = np.asfortranarray(
        np.clip(right, -limit, limit).astype(ml_dtypes.float8_e4m3fn)
    )
    scales = MatmulQuantizationScales(a=1.0, b=1.0)
    options = {"result_type": nvmath.CudaDataType.CUDA_R_16BF}

    def kernel():
        return cublas_matmul(query, packed, quantization_scales=scales, options=options)

    benchmark.extra_info["gflop"] = 2 * side**3 / 1e9
    assert np.asarray(benchmark(kernel)).shape == (side, side)


# ? Intel Xeon 4 and NVIDIA H100 · float32 on the CPU against e4m3 on the GPU
# ?
# ?      n        CPU        GPU    GPU wins by  CPU growth   GPU growth
# ?    512     0.3 ms     7.2 ms       0.05x
# ?   1024     3.7 ms     7.2 ms       0.52x        11.1x        1.0x
# ?   2048    27.2 ms     7.1 ms       3.84x         7.3x        1.0x
# ?   4096   182.5 ms    28.5 ms       6.39x         6.7x        4.0x
# ?
# ? Read the last two columns. Doubling n multiplies the CPU's work by eight
# ? and its time by seven to eleven; it multiplies the GPU's bytes by four and
# ? its time by one to four. Compute scales with n³, transfer with n², so the
# ? ratio between them scales with n — and the GPU wins asymptotically however
# ? slow the interconnect. It only has to be given enough work to amortize.
# ?
# ? Below the crossover near n=1500 the GPU is 20x *slower*, which is the same
# ? shape as the first chapter of this file: `np.sin` loses to `math.sin` on
# ? one scalar and wins enormously on an array. Different hardware, identical
# ? lesson — fixed overhead is only amortized by volume.
# ?
# ? Note what the GPU column is not. An H100 does roughly 1'500 TFLOP/s of
# ? e4m3, and this measures under 5 even at n=4096: over 99% of that time is
# ? the answer being copied back to host memory, not tensor cores. nvmath
# ? accepts device-resident arrays only from CuPy or torch, so from NumPy the
# ? ceiling is structural. The scaling argument survives it — n² beats n³
# ? eventually, however large the constant.

# endregion: Scaling Laws

# endregion: Numeric Types
