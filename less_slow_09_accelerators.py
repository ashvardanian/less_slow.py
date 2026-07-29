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

# region: What NumPy Gives You

# ? Multiplying two matrices is the same arithmetic whatever the elements are,
# ? so halving the width of a number ought to halve the work. Ask NumPy for
# ? float16 and it takes about a thousand times longer than float32. Same
# ? operation, same machine, narrower type, three orders of magnitude slower.
# ?
# ? Speed here is not a property of the type. It is a property of whether a
# ? kernel exists for it.
# ?
# ? Inputs are small integers, exactly representable in every type compared,
# ? so nothing varies but the kernel. Four matrix sizes, because one size
# ? answers the question wrong.

MATRIX_SIDES = [512, 1024, 2048, 4096]
THREADS = 8


def _integer_operands(dtype, side: int = MATRIX_SIDES[0]):
    """Small integers, exactly representable in every type compared here."""
    rng = np.random.default_rng(42)
    left = rng.integers(0, 10, size=(side, side)).astype(dtype)
    right = rng.integers(0, 10, size=(side, side)).astype(dtype)
    return left, np.asfortranarray(right)


@pytest.mark.benchmark(group="09-accelerators-numpy-dtypes")
@pytest.mark.parametrize("side", MATRIX_SIDES)
@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.float16, np.int32, np.int16]
)
def test_dtype_numpy(benchmark, dtype, side):
    """NumPy across types — BLAS covers two of these five."""
    if dtype in (np.float16, np.int32, np.int16) and side > 1024:
        # ! Without a BLAS kernel these run a generic loop, and one repetition
        # ! of the larger size takes minutes. Their absence from the big column
        # ! is the measurement.
        pytest.skip("no kernel; a single repetition would take minutes")
    left, right = _integer_operands(dtype, side)

    def kernel():
        return left @ right

    assert benchmark(kernel).shape == (side, side)


@pytest.mark.benchmark(group="09-accelerators-numpy-int8")
def test_lowprec_int8_dot_numpy(benchmark):
    """NumPy keeps the accumulator in int8, so a long dot product overflows."""
    rng = np.random.default_rng(42)
    first = rng.integers(-128, 127, MATRIX_SIDES[0], dtype=np.int8)
    second = rng.integers(-128, 127, MATRIX_SIDES[0], dtype=np.int8)
    exact = int(first.astype(np.int64) @ second.astype(np.int64))

    def kernel():
        return int(np.dot(first, second))

    result = benchmark(kernel)
    benchmark.extra_info["exact"] = exact
    benchmark.extra_info["reported"] = result
    # ! Asserting the *wrong* answer, so this fails the day NumPy widens it.
    assert result != exact


# ? Intel Xeon 4 • TFLOP/s, bigger is better
# ?
# ?                        512       1024       2048       4096
# ?   float32            0.831      0.567      0.591      0.631
# ?   float64            0.386      0.386      0.310      0.408
# ?   int32              0.005      0.004          -          -
# ?   float16            0.000      0.000          -          -
# ?
# ? BLAS covers two of the five. Outside them NumPy runs a generic loop, and
# ? the empty cells are benchmarks skipping themselves because one repetition
# ? takes minutes — an absence that is itself the measurement.
# ?
# ? Speed is only half of it. NumPy accumulates an int8 dot product *in int8*,
# ? so a long one wraps silently: not approximate, wrong, and often the wrong
# ? sign. Nothing raises.

# endregion: What NumPy Gives You

# region: What NumKong Adds

# ? The gap is historical, not technical. BLAS was standardized in 1979 and
# ? covers the types that existed then. bfloat16 was published in 2018, e4m3
# ? in 2022. NumPy hands the array to a library that has never heard of them,
# ? and no future version of that library will.
# ?
# ? NumKong writes those kernels: same arithmetic, same silicon, SIMD for the
# ? types BLAS predates. It also widens the int8 accumulator, which was always
# ? a choice rather than a limit.
# ?
# ? Its kernels release the GIL but it ships no thread pool, so the caller
# ? slices the rows and supplies the threads.


@pytest.mark.benchmark(group="09-accelerators-numkong-int8")
def test_lowprec_int8_dot_numkong(benchmark):
    """NumKong widens the accumulator, so the same inputs come out right."""
    rng = np.random.default_rng(42)
    first = rng.integers(-128, 127, MATRIX_SIDES[0], dtype=np.int8)
    second = rng.integers(-128, 127, MATRIX_SIDES[0], dtype=np.int8)
    exact = int(first.astype(np.int64) @ second.astype(np.int64))

    def kernel():
        return int(nk.dot(first, second))

    result = benchmark(kernel)
    assert result == exact


# ? Accuracy below is the angle between each computed row and the exact one,
# ? not a relative error. Quantized arithmetic undoes its scale afterwards, so
# ? a row uniformly off by a constant is recoverable and a row pointing the
# ? wrong way is not — cosine similarity separates those, and a plain norm
# ? does not. In numerical linear algebra this is the angle between vectors;
# ? in the quantization literature the usual alternative is SQNR, which is
# ? scale-sensitive and would hide the distinction.


def _float_operands(side: int = MATRIX_SIDES[0]):
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


@pytest.mark.benchmark(group="09-accelerators-numkong-dtypes")
@pytest.mark.parametrize("side", MATRIX_SIDES)
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16", "e4m3"])
def test_dtype_numkong(benchmark, dtype_name, side, matmul_threads):
    """NumKong across types — narrower really is faster when a kernel exists."""
    left, right, reference = _float_operands(side)
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
    span = (side + THREADS - 1) // THREADS

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
                    end_row=min(side, (index + 1) * span),
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


# ? Intel Xeon 4 • 8 threads • TFLOP/s, and median per-row angle against an
# ? exact float64 product
# ?
# ?                        512       1024       2048       4096      angle
# ?   e4m3               0.792      1.674      2.080      2.022     2.142°
# ?   bfloat16           0.766      1.499      1.846      1.761     0.135°
# ?   float32 → 64       0.102      0.134      0.112      0.108     0.000°
# ?
# ? Roughly triple NumPy's throughput on the narrow types. The last row looks
# ? like a 5x loss against OpenBLAS but is not the same operation: NumKong
# ? multiplies float32 and accumulates into float64, returning a float64
# ? matrix. Against an exact reference OpenBLAS lands at 3.4e-07 and NumKong
# ? at 5.1e-16 — the difference between carrying float32 error and carrying
# ? float64 error through 1024 additions. The wider accumulator is the price.
# ?
# ? bfloat16 is the row to take away. It spends its 16 bits on exponent rather
# ? than mantissa, which is why it survives activations that overflow float16
# ? and why the industry trains in it. e4m3 halves the width again for 16x the
# ? angle and no more speed — the kernel is already saturated.

# endregion: What NumKong Adds

# region: What the GPU Adds

# ? A GPU is a faster processor at the end of a bus, and the bus charges by
# ? the byte. So "is the GPU faster" has no answer: a matrix of side n carries
# ? n² numbers across and buys n³ arithmetic, meaning the toll is fixed while
# ? the payoff grows. There is a size below which the trip does not pay, and
# ? the only useful question is where it falls.
# ?
# ? e4m3 tensor cores arrived with Hopper. Below them sit block-scaled formats
# ? — MXFP8 sharing one exponent per 32 values, NVFP4 per 16 — which need
# ? Blackwell: narrower elements, with a shared exponent buying back the range
# ? they lose.


@pytest.mark.benchmark(group="09-accelerators-scaling-cpu")
@pytest.mark.parametrize("side", MATRIX_SIDES)
def test_scaling_cpu(benchmark, side: int):
    """NumPy float32 on the CPU — compute-bound, so cost grows with n³."""
    left, right = _integer_operands(np.float32, side)

    def kernel():
        return left @ right

    benchmark.extra_info["gflop"] = 2 * side**3 / 1e9
    assert benchmark(kernel).shape == (side, side)


@pytest.mark.skipif(not gpu_matmul_usable, reason="no usable cuBLAS GPU")
@pytest.mark.benchmark(group="09-accelerators-scaling-gpu")
@pytest.mark.parametrize("side", MATRIX_SIDES)
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


# ? Intel Xeon 4 against NVIDIA H100 • TFLOP/s, the whole chain assembled
# ?
# ?                           512       1024       2048       4096
# ?   NumPy    float32      0.831      0.567      0.591      0.631
# ?   NumPy    float16      0.000      0.000          -          -
# ?   NumKong  bfloat16     0.766      1.499      1.846      1.761
# ?   NumKong  e4m3         0.792      1.674      2.080      2.022
# ?   nvmath   e4m3         0.038      0.301      2.457      4.677
# ?
# ? Read the last row across, not down. At 512 the accelerator is 22x *slower*
# ? than NumPy on the CPU; by 4096 it is 7x faster and ahead of everything.
# ? Compute scales with n³ and transfer with n², so their ratio scales with n.
# ?
# ? Each tier wins a band. NumPy is the floor and drops out entirely without a
# ? kernel. NumKong owns the middle, where matrices are too small to pay for a
# ? bus trip. The GPU takes everything above roughly 1500 and keeps pulling
# ? away. An accelerator is not fast or slow — it is fast above a size.

# endregion: What the GPU Adds

# endregion: Numeric Types
