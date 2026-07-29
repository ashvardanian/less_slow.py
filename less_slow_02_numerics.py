#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Numerics — when the standard library beats NumPy, and picking a factorization.

Two lessons that pull in opposite directions. NumPy is not a faster `math`:
on a single scalar its dispatch machinery makes `np.sin` 2.78x slower than
`math.sin`, and it only pays off past 8 to 16 elements. Both are straight
lines — NumPy costs ~340 ns before touching an element and ~5.9 ns each
after, a Python loop costs nothing up front and ~40 ns each — so every
argument about whether NumPy is faster is an argument about where you sit on
them.

Then linear algebra, where the win is algorithmic rather than mechanical. A
rank-k factorization turns an O(n²m) product into O(knm), worth 30x at n=5000
and k=4%. Computing the factorization is the expensive part, and its cost is
set by how little the method assumes: on the same symmetric matrix, Cholesky
is 42x cheaper than QR and 128x cheaper than SVD. The constraint is what the
speed is made of.
"""

import math

import numpy as np
import pytest

from less_slow_00_shared import numba_installed

if numba_installed:
    import numba

# region: Standard Library vs NumPy

# ? `np.sin` is not a faster `math.sin`. It is a different kind of function:
# ? one that inspects its argument, decides on a dtype, allocates an output,
# ? and dispatches to a loop. Handed a single float, all of that machinery runs
# ? to produce one number.
# ?
# ?   math.sin(x)    a C call on a float
# ?   np.sin(x)      coerce → find a ufunc loop → allocate → run it → unbox
# ?
# ? The arithmetic is the same either way. What differs is everything around it.


def f64_sine_math(x: float) -> float:
    return math.sin(x)


def f64_sine_numpy(x: float) -> float:
    return np.sin(x)


# ? Intel Xeon 4 • CPython 3.14t • 10K scalars, one call each
# ?
# ?   math.sin    847 µs    1.00x
# ?   np.sin    2'262 µs    2.67x  dispatch dwarfs the arithmetic
# ?
# ? A second avenue is to compute less. When bit-accuracy is not required, the
# ? Maclaurin series approximates sine with nothing but multiplication:
# ?
# ?   sin(x) ≈ x − x³/3! + x⁵/5! − x⁷/7! + …
# ?
# ? It converges quickly for small x, and it is worth writing three ways —
# ? through `math.pow`, through `np.pow`, and with the powers unrolled into
# ? plain multiplications — because in an interpreted language the count of
# ? operations is the count of dispatches.


def f64_sine_math_maclaurin(x: float) -> float:
    return x - math.pow(x, 3) / 6.0 + math.pow(x, 5) / 120.0


def f64_sine_numpy_maclaurin(x: float) -> float:
    return x - np.pow(x, 3) / 6.0 + np.pow(x, 5) / 120.0


def f64_sine_maclaurin_powless(x: float) -> float:
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    return x - (x3 / 6.0) + (x5 / 120.0)


# ? Two harnesses, differing only in whether the function is called once per
# ? value or once for all of them. Every benchmark below picks one, and which
# ? one it picks matters more than which sine it measures.


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


# ? Numba compiles the kernel to machine code, which should erase the
# ? interpreter overhead the three variants above were competing over. It
# ? erases the overhead inside the kernel:
# ?
# ?   for x in inputs:      ← still interpreted, 10'000 times
# ?       sin_fn(x)         ← machine code, plus a boundary crossing
# ?
# ? The loop and the crossing are not compiled, and at one arithmetic
# ? expression per call they are most of the work.


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


# ? Intel Xeon 4 • CPython 3.14t • 10K values in [0, 2π], one run
# ?
# ?                             per scalar    whole array
# ?   math.sin                      847 µs             —
# ?   maclaurin, unrolled + JIT   1'804 µs             —
# ?   np.sin                      2'262 µs         127 µs
# ?   maclaurin, math.pow         2'389 µs             —
# ?   maclaurin, unrolled         2'861 µs        37.7 µs
# ?   maclaurin, np.pow          20'730 µs        78.9 µs
# ?
# ? Read the two columns against each other. `maclaurin, unrolled` is the
# ? second-slowest function per scalar and the fastest of all on an array — a
# ? 76x swing with the source unchanged. Only the shape of the argument moved.
# ?
# ? That is the whole case for Python as glue: the arithmetic is identical, and
# ? what costs is crossing from the interpreter into the kernel. Ten thousand
# ? crossings or one.


# ? Three details in that table are worth naming.
# ?
# ? `np.pow` on a scalar costs 20'730 µs against `math.pow`'s 2'389 — 8.7x for
# ? the same exponentiation, because each of the two `np.pow` calls per value
# ? repeats the coerce-dispatch-allocate sequence that `np.sin` pays once.
# ? Reaching for NumPy inside a scalar loop compounds with the operation count.
# ?
# ? Unrolling the powers into plain multiplications is *slower* per scalar than
# ? calling `math.pow` — 2'861 against 2'389 — and 2.1x faster on the array. In
# ? the interpreter, five bytecodes beat one C call only when the per-element
# ? dispatch is gone.
# ?
# ? Numba is the row that undercuts its own reputation. JIT-compiling the
# ? unrolled version gives 1'804 µs, a 1.59x win over the interpreted form and
# ? still 2.13x *slower* than plain `math.sin`. The kernel is machine code; the
# ? loop calling it ten thousand times is not, and that loop is the cost.


# ? Which raises the question the scalar table does not answer — how big does
# ? an array have to be before the crossing pays for itself?

ARRAY_SIZES = [1, 8, 16, 64, 256, 1024]


@pytest.mark.benchmark(group="02-numerics-crossover")
@pytest.mark.parametrize("elements", ARRAY_SIZES)
def test_crossover_math_loop(benchmark, elements: int):
    """`math.sin` over a Python list — no fixed cost, a per-element one."""
    values = (np.random.rand(elements) * 2 * np.pi).tolist()
    result = benchmark(lambda: [math.sin(value) for value in values])
    assert len(result) == elements


@pytest.mark.benchmark(group="02-numerics-crossover")
@pytest.mark.parametrize("elements", ARRAY_SIZES)
def test_crossover_numpy_array(benchmark, elements: int):
    """`np.sin` over an array — a fixed cost, then a cheap per-element one."""
    values = np.random.rand(elements) * 2 * np.pi
    result = benchmark(lambda: np.sin(values))
    assert result.shape == (elements,)


# ? Intel Xeon 4 • CPython 3.14t • timeit, one call over n values
# ?
# ?         n    math.sin loop    np.sin array    winner
# ?         1           72 ns          349 ns      math   4.83x
# ?         8          303 ns          394 ns      math   1.30x
# ?        16          568 ns          446 ns     numpy   1.27x
# ?        64        2'261 ns          686 ns     numpy   3.30x
# ?       256        9'789 ns        1'632 ns     numpy   6.00x
# ?     1'024       40'161 ns        5'519 ns     numpy   7.28x
# ?
# ? The crossover sits between 8 and 16 elements. Below it NumPy is the wrong
# ? tool by up to 4.83x; above it, it wins by a margin that keeps growing to
# ? about 7x and then flattens.
# ?
# ? Both rows are straight lines with different intercepts and slopes: NumPy
# ? pays ~340 ns before touching an element and ~5.9 ns each afterwards, while
# ? the loop pays nothing up front and ~40 ns each. Every "is NumPy faster"
# ? argument is really an argument about where you are on those two lines.

# endregion: Standard Library vs NumPy

# region: Matrix Decompositions

# ? The sine section wins by removing interpreter overhead. This one wins by
# ? doing less arithmetic, which is a larger and more durable kind of win.
# ?
# ? Multiplying A (n×(n+1)) by X ((n+1)×m) costs O(n²m). If A is close to some
# ? rank-k matrix, factor it once and multiply through the factors instead:
# ?
# ?   A · X          n×(n+1) · (n+1)×m         2n(n+1)m flops
# ?   Uₖ(Sₖ(Vₖᵀ·X))  three thin products       ≈4nkm flops
# ?
# ? At k = 4% of n that is a 25x reduction in arithmetic, paid for once by the
# ? factorization and by whatever accuracy the truncation costs. The
# ? Eckart–Young–Mirsky theorem says SVD's truncation is the best possible for
# ? a given k under the Frobenius norm ‖·‖, so it sets the accuracy bar the
# ? cheaper factorizations are measured against.
# ?
# ? https://en.wikipedia.org/wiki/Singular_value_decomposition

MATRIX_SIDES = [1000, 5000]
CIRCUIT_WIDTH = 10  # columns of X, deliberately narrow — this is a thin product


def _frobenius_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def _operands(n_dim: int, seed: int = 0):
    """A and X for the product A @ X, non-square on purpose."""
    generator = np.random.default_rng(seed)
    left = generator.random((n_dim, n_dim + 1), dtype=np.float32)
    right = generator.random((n_dim + 1, CIRCUIT_WIDTH), dtype=np.float32)
    return left, right


def _svd_factors(matrix: np.ndarray, rank: int):
    """Uₖ, Sₖ, Vₖᵀ — the optimal rank-k truncation."""
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    return left[:, :rank], singular[:rank], right[:rank, :]


def _qr_factors(matrix: np.ndarray, rank: int):
    """Qₖ, Rₖ — cheaper to compute, and not an optimal truncation."""
    orthogonal, triangular = np.linalg.qr(matrix, mode="reduced")
    return orthogonal[:, :rank], triangular[:rank, :]


@pytest.mark.benchmark(group="02-numerics-decomposition")
@pytest.mark.parametrize("n_dim", MATRIX_SIDES)
def test_matmul_full(benchmark, n_dim: int):
    """The undecomposed product, which every rank-k row is measured against."""
    left, right = _operands(n_dim)

    benchmark(lambda: left @ right)
    benchmark.extra_info["flops"] = 2 * n_dim * (n_dim + 1) * CIRCUIT_WIDTH


# ? Two errors are worth separating, and only one of them is the interesting
# ? one:
# ?
# ?   recovery error   ‖A − Aₖ‖ / A.size        how badly the factors miss A
# ?   circuit error    ‖AₖX − AX‖ / result      how badly the product misses
# ?
# ? The first is a property of the factorization and the rank. The second is
# ? what a caller actually observes, and it can be far smaller — a direction A
# ? loses in truncation costs nothing if X has no weight along it.


@pytest.mark.benchmark(group="02-numerics-decomposition")
@pytest.mark.parametrize("n_dim", MATRIX_SIDES)
@pytest.mark.parametrize("k_percent", [20, 4])
@pytest.mark.parametrize("factorization", ["svd", "qr"])
def test_matmul_low_rank(benchmark, n_dim: int, k_percent: int, factorization: str):
    """The same product through rank-k factors, timed without the error check."""
    rank = int(n_dim * k_percent / 100)
    left, right = _operands(n_dim)
    reference = left @ right

    if factorization == "svd":
        basis, singular, coefficients = _svd_factors(left, rank)
        approximation = basis @ (singular[:, np.newaxis] * coefficients)

        def product():
            # ! Scale by the singular values as a row broadcast rather than
            # ! building diag(Sₖ), which would be k² of mostly zeros.
            return basis @ (singular[:, np.newaxis] * (coefficients @ right))

        decomposed_flops = (
            2 * rank * (n_dim + 1) * CIRCUIT_WIDTH
            + rank * CIRCUIT_WIDTH
            + 2 * n_dim * rank * CIRCUIT_WIDTH
        )
    else:
        orthogonal, triangular = _qr_factors(left, rank)
        approximation = orthogonal @ triangular

        def product():
            return orthogonal @ (triangular @ right)

        decomposed_flops = (
            2 * rank * (n_dim + 1) * CIRCUIT_WIDTH + 2 * n_dim * rank * CIRCUIT_WIDTH
        )

    # ! Both errors are computed once, here, outside the timed region. Folding
    # ! them into the kernel would add a constant n×m subtraction and norm to
    # ! every repetition — a cost that does not shrink with k, which is exactly
    # ! the axis being measured, and would compress every speedup toward 1.0.
    recovery_error = _frobenius_norm(left - approximation) / left.size
    circuit_error = _frobenius_norm(product() - reference) / reference.size

    benchmark(product)
    benchmark.extra_info["mean_recovery_error"] = recovery_error
    benchmark.extra_info["mean_circuit_error"] = circuit_error
    benchmark.extra_info["flops_reduction"] = (
        2 * n_dim * (n_dim + 1) * CIRCUIT_WIDTH
    ) / decomposed_flops


# ? Intel Xeon 4 • CPython 3.14t • timeit, A·X against its rank-k factors
# ?
# ?                    n=1'000              n=5'000
# ?   full             69.8 µs   1.00x    5'825 µs    1.00x
# ?   svd  k=20%       66.9 µs   1.04x    4'948 µs    1.18x
# ?   qr   k=20%       58.4 µs   1.19x    2'159 µs    2.70x
# ?   svd  k=4%        34.5 µs   2.02x      194 µs   30.10x
# ?   qr   k=4%        34.8 µs   2.00x      187 µs   31.20x
# ?
# ? At n=5000 and k=4% the arithmetic drops 25x by the flop count and the wall
# ? clock drops 30x, so the model holds. At n=1000 the same k buys only 2x —
# ? the products are small enough that BLAS never reaches its stride, and a
# ? quarter of the theoretical win is left on the table.


# ? Two factorizations, two bargains. QR truncation keeps the *first* k
# ? directions; without column pivoting there is no reason those are the k
# ? largest. SVD keeps the k largest by construction, which is exactly what
# ? Eckart–Young–Mirsky guarantees:
# ?
# ?   svd  k=4%, n=5'000    recovery error   5.352e-05
# ?   qr   k=4%, n=5'000    recovery error   5.556e-05
# ?
# ? Within 4% of each other — which is a property of this operand rather than
# ? a general result. A uniform random matrix is dominated by one singular
# ? direction, so almost any k columns capture it. Give the matrix structure
# ? spread across many directions and the gap opens, at which point QR's
# ? cheapness stops being free.


@pytest.mark.benchmark(group="02-numerics-factorize")
@pytest.mark.parametrize("factorization", ["svd", "qr", "cholesky"])
def test_factorization_cost(benchmark, factorization: str):
    """What each factorization costs to compute, on a matrix admitting all three."""
    n_dim = 1000
    generator = np.random.default_rng(0)
    basis = generator.random((n_dim, n_dim), dtype=np.float32)
    # ! Symmetric positive-definite, so Cholesky is defined: A = Aᵀ and
    # ! xᵀAx > 0 for every x ≠ 0. The nI term guarantees the second condition.
    matrix = basis @ basis.T + n_dim * np.eye(n_dim, dtype=np.float32)

    kernels = {
        "svd": lambda: np.linalg.svd(matrix, full_matrices=False),
        "qr": lambda: np.linalg.qr(matrix, mode="reduced"),
        "cholesky": lambda: np.linalg.cholesky(matrix),
    }
    result = benchmark(kernels[factorization])
    assert result is not None


# ? Intel Xeon 4 • CPython 3.14t • factorizing one 1000×1000 SPD matrix
# ?
# ?   cholesky      13.8 ms      1.00x   needs A = Aᵀ and xᵀAx > 0
# ?   qr           588.4 ms     42.62x   needs nothing
# ?   svd        1'761.7 ms    127.62x   needs nothing, sorts by importance
# ?
# ? A 128x spread on the same matrix. Each buys its speed by refusing inputs:
# ?
# ?   cholesky   A = L·Lᵀ       symmetric positive-definite only
# ?   qr         A = Q·R        any matrix, in the column order given
# ?   svd        A = U·S·Vᵀ     any matrix, directions sorted by weight
# ?
# ? So the rule is to use the most constrained factorization the matrix
# ? actually admits, because the constraint is what the speed is made of.
# ? Knowing your matrix is symmetric is worth 42x here, and that knowledge
# ? comes from the problem rather than from the profiler.
# ?
# ? LU sits between QR and Cholesky and needs a square matrix; NumPy does not
# ? expose it directly, and reaching for `scipy.linalg.lu` would be the point
# ? at which this repository gained a dependency for one row of one table.

# endregion: Matrix Decompositions
