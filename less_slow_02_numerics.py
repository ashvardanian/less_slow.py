#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Numerics — when the standard library beats NumPy, and picking a factorization.

Two lessons that pull in opposite directions. NumPy is not a faster `math`:
on a single scalar its dispatch machinery makes `np.sin` several times slower
than `math.sin`, and it only wins once the input is an array large enough to
amortize that overhead. This crossover is the same shape as the CPU-versus-GPU
one in the accelerators chapter, at a smaller scale.

Then linear algebra, where the win is algorithmic rather than mechanical. A
rank-k factorization turns an O(n²m) product into O(knm), and the right
factorization is the most constrained one your matrix admits: SVD is optimal
by Eckart-Young-Mirsky, QR is cheaper, and Cholesky cheaper still if the
matrix is symmetric positive-definite.
"""
import math

import numpy as np
import pytest

from less_slow_00_shared import numba_installed

if numba_installed:
    import numba

# region: Standard Library vs NumPy

# ? Numerical computing is a core subject in high-performance computing (HPC)
# ? research and graduate studies, yet its foundational concepts are more
# ? accessible than they seem. Let's start with one of the most basic
# ? operations — computing the __sine__ of a number.


def f64_sine_math(x: float) -> float:
    return math.sin(x)


def f64_sine_numpy(x: float) -> float:
    return np.sin(x)


# ? NumPy is the de-facto standard for numerical computing in Python, and
# ? it's known for its speed and simplicity. However, it's not always the
# ? fastest option for simple operations like sine, cosine, or exponentials.
# ?
# ? NumPy lacks hot-path optimizations if the input is a single scalar value,
# ? and it's often slower than the standard math library for such cases:
# ?
# ? Intel Xeon 4 · CPython 3.14t · 10K scalars, one call each
# ?
# ?   math.sin    699 µs    1.00x
# ?   np.sin    1'946 µs    2.78x  dispatch dwarfs the arithmetic
# ?
# ? NumPy, of course, becomes much faster when the input is a large array,
# ? as opposed to a single scalar value.
# ?
# ? When absolute bit-accuracy is not required, it's often possible to
# ? approximate mathematical functions using simpler, faster operations.
# ? For example, the Maclaurin series for sine:
# ?
# ?   sin(x) ≈ x − x³/3! + x⁵/5! − x⁷/7! + …
# ?
# ? is a simple polynomial approximation that converges quickly for small x.
# ? Both can be implemented in Python, NumPy, or Numba JIT, and benchmarked.


def f64_sine_math_maclaurin(x: float) -> float:
    return x - math.pow(x, 3) / 6.0 + math.pow(x, 5) / 120.0


def f64_sine_numpy_maclaurin(x: float) -> float:
    return x - np.pow(x, 3) / 6.0 + np.pow(x, 5) / 120.0


def f64_sine_maclaurin_powless(x: float) -> float:
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    return x - (x3 / 6.0) + (x5 / 120.0)


# ? Let's define a couple of helper functions to run benchmarks on these functions,
# ? and compare their performance on 10k random floats in [0, 2π]. We can also
# ? use Numba JIT to compile these functions to machine code, and compare the
# ? performance of the JIT-compiled functions with the standard Python functions.


def _f64_sine_run_benchmark_on_each(benchmark, sin_fn):
    """Applies `sin_fn` to 10k random floats in [0, 2π] individually."""
    inputs = np.random.rand(10_000)  # 10k random floats
    inputs = inputs.astype(np.float64) * 2 * np.pi  # [0, 2π]

    def call_sin_on_all():
        for x in inputs:
            sin_fn(x)

    result = benchmark(call_sin_on_all)
    return result


def _f64_sine_run_benchmark_on_all(benchmark, sin_fn):
    """Applies `sin_fn` to 10k random floats in [0, 2π] all at once."""
    inputs = np.random.rand(10_000)  # 10k random floats
    inputs = inputs.astype(np.float64) * 2 * np.pi  # [0, 2π]
    call_sin_on_all = lambda: sin_fn(inputs)  # noqa: E731
    result = benchmark(call_sin_on_all)
    return result


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sine_math(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_math)


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sine_numpy(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_numpy)


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sine_maclaurin_math(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_math_maclaurin)


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sine_maclaurin_numpy(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_numpy_maclaurin)


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sine_maclaurin_powless(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_maclaurin_powless)


# ? Numba compiles the kernel to machine code, which erases the interpreter
# ? overhead that the three variants above were competing over — so one
# ? JIT-ed benchmark is enough to make the point that the algorithmic
# ? difference between them stops mattering once the interpreter is gone.


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sine_maclaurin_powless_jit(benchmark):
    sin_fn = numba.njit(f64_sine_maclaurin_powless)
    sin_fn(0.0)  # trigger compilation
    _f64_sine_run_benchmark_on_each(benchmark, sin_fn)


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sines_numpy(benchmark):
    _f64_sine_run_benchmark_on_all(benchmark, f64_sine_numpy)


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sines_maclaurin_numpy(benchmark):
    _f64_sine_run_benchmark_on_all(benchmark, f64_sine_numpy_maclaurin)


@pytest.mark.benchmark(group="02-numerics-sin")
def test_f64_sines_maclaurin_powless(benchmark):
    _f64_sine_run_benchmark_on_all(benchmark, f64_sine_maclaurin_powless)


# ? The results are somewhat shocking!
# ?
# ? `f64_sine_maclaurin_powless` and `test_f64_sine_maclaurin_numpy` are
# ? both the fastest and one of the slowest implementations, depending on
# ? the input shape - scalar or array: 29us vs 2610us.
# ?
# ? This little benchmark is enough to understand, why Python is broadly
# ? considered a "glue" language for various native languages and pre-compiled
# ? libraries for batch/bulk processing.

# endregion: Standard Library vs NumPy

# region: Matrix Decompositions

# ? Python benefits greatly from a wide ecosystem of numerics libraries.
# ? If you know Linear Algebra, there is a wide range of tricks you can
# ? use to speed up your code.
# ?
# ? For example, the Singular Value Decomposition (SVD) is a fundamental
# ? operation proven by the Eckart–Young–Mirsky theorem to be the best
# ? rank-k approximation of a matrix w.r.t. the Frobenius norm ‖·‖_F:
# ? A ≈ U_k · Σ_k · V_kᵀ minimizes ‖A − A_k‖_F over all rank-k matrices.
# ?
# ? https://en.wikipedia.org/wiki/Singular_value_decomposition
# ? https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm


@pytest.mark.benchmark(group="02-numerics-decomposition")
@pytest.mark.parametrize("n_dim", [1000, 5000])
@pytest.mark.parametrize("k_percent", ["100%", "20%", "4%"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_svd(benchmark, n_dim: int, k_percent: str, dtype: np.dtype):
    k = int(n_dim * int(k_percent[:-1]) / 100)
    m = 10  # Number of columns in the multiplication circuit

    # ? Just to prove a point, let's make the second dimension slightly
    # ? larger to show that this works for non-square matrices too.
    A = np.random.rand(n_dim, n_dim + 1).astype(dtype)
    X = np.random.rand(n_dim + 1, m).astype(dtype)

    def baseline():
        return A @ X

    def frobenius_norm(A: np.ndarray) -> float:
        return np.linalg.norm(A, ord="fro")

    def low_rank_approximation(A, k: int) -> tuple:
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        return U[:, :k], S[:k], VT[:k, :]

    # Compute the low-rank approximation of A just once
    U_k, S_k, VT_k = low_rank_approximation(A, k)
    S_truncated = np.zeros((k, k))
    np.fill_diagonal(S_truncated, S_k)

    # Estimate the error of the low-rank approximation
    mean_recovery_error = frobenius_norm(A - U_k @ S_truncated @ VT_k) / A.size
    expected_result = baseline()
    max_circuit_error = 0.0  # ? We will update this later

    # Decomposed multiplication: compute A_approx @ X where A_approx = U_k * diag(S_k) * VT_k.
    # This is computed in two steps: first compute (VT_k @ X), then scale by S_k, and finally multiply by U_k.
    def decomposed():
        temp = VT_k @ X  # (k x m)
        # Scale each row by the corresponding singular value without explicitly constructing
        # the diagonal matrix.
        temp = S_k[:, np.newaxis] * temp
        return U_k @ temp  # (n_dim x m)

    def bench_svd():
        product = baseline if k_percent == "100%" else decomposed
        result = product()
        mean_error = frobenius_norm(result - expected_result) / result.size

        # Update the `max_circuit_error`
        nonlocal max_circuit_error
        max_circuit_error = max(max_circuit_error, mean_error)
        return mean_error

    # Estimate FLOPs:
    # Full multiplication: A (n_dim x (n_dim+1)) times X ((n_dim+1) x m)
    # roughly requires 2 * n_dim * (n_dim+1) * m floating-point operations.
    full_flops = 2 * n_dim * (n_dim + 1) * m

    # Decomposed multiplication consists of:
    # 1. VT_k @ X: (k x (n_dim+1)) * ((n_dim+1) x m) → 2 * k * (n_dim+1) * m flops.
    # 2. Scaling by S_k: k * m flops (one multiplication per element).
    # 3. U_k @ (result): (n_dim x k) * (k x m) → 2 * n_dim * k * m flops.
    decomposed_flops = 2 * k * (n_dim + 1) * m + k * m + 2 * n_dim * k * m

    benchmark(bench_svd)
    benchmark.extra_info["mean_recovery_error"] = mean_recovery_error
    benchmark.extra_info["flops_reduction"] = full_flops / decomposed_flops


# ? SVD is not the only decomposition method. Less theoretically sound but
# ? often faster is the QR decomposition, which is applicable to any full-rank.
# ? It produces an orthogonal matrix Q and an upper triangular matrix R such
# ? that A = Q·R, with QᵀQ = I.


@pytest.mark.benchmark(group="02-numerics-decomposition")
@pytest.mark.parametrize("n_dim", [1000, 5000])
@pytest.mark.parametrize("k_percent", ["100%", "20%", "4%"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qr(benchmark, n_dim: int, k_percent: float, dtype: np.dtype):
    k = int(n_dim * int(k_percent[:-1]) / 100)
    m = 10  # Number of columns in the multiplication circuit

    # ? Just to prove a point, let's make the second dimension slightly
    # ? larger to show that this works for non-square matrices too.
    A = np.random.rand(n_dim, n_dim + 1).astype(dtype)
    X = np.random.rand(n_dim + 1, m).astype(dtype)

    def baseline():
        return A @ X

    def frobenius_norm(mat: np.ndarray) -> float:
        return np.linalg.norm(mat, ord="fro")

    # Compute the reduced QR decomposition of A.
    Q, R = np.linalg.qr(A, mode="reduced")
    Q_k = Q[:, :k]
    R_k = R[:k, :]

    # Estimate the recovery error of the truncated QR approximation.
    A_approx = Q_k @ R_k
    mean_recovery_error = frobenius_norm(A - A_approx) / A.size
    expected_result = baseline()
    max_circuit_error = 0.0  # ? We will update this later

    # Decomposed multiplication using the truncated QR factors.
    def decomposed():
        return Q_k @ (R_k @ X)

    def bench_qr():
        product = baseline if k_percent == "100%" else decomposed
        result = product()
        mean_error = frobenius_norm(result - expected_result) / result.size

        # Update the `max_circuit_error`
        nonlocal max_circuit_error
        max_circuit_error = max(max_circuit_error, mean_error)
        return mean_error

    # Estimate FLOPs:
    # Full multiplication: A (n_dim x (n_dim+1)) times X ((n_dim+1) x m)
    full_flops = 2 * n_dim * (n_dim + 1) * m
    # Decomposed multiplication:
    # 1. R_k @ X: 2 * k * (n_dim+1) * m flops.
    # 2. Q_k @ (result): 2 * n_dim * k * m flops.
    decomposed_flops = 2 * k * (n_dim + 1) * m + 2 * n_dim * k * m

    benchmark(bench_qr)
    benchmark.extra_info["mean_recovery_error"] = mean_recovery_error
    benchmark.extra_info["flops_reduction"] = full_flops / decomposed_flops


# ? For narrower classes of matrices we can do even better. SciPy also provides
# ? LU and Cholesky decompositions, but each buys its speed by narrowing the
# ? inputs it accepts: LU needs a square matrix, and Cholesky needs a symmetric
# ? positive-definite one — A = Aᵀ, and xᵀ·A·x > 0 for every x ≠ 0.
# ? They sit further along the same curve SVD and QR already trace, so the rule
# ? generalizes: pick the decomposition your matrix structure allows.


# endregion: Matrix Decompositions
