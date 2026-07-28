#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parallelism — threads, subinterpreters, and processes, with and without the GIL.

Since 3.14 all three share the executor interface, so one kernel drives them
all. Under the GIL, threads are slower than running serially. Free-threaded,
they scale. Processes never cared either way, because they never shared an
interpreter to contend over.

The row worth remembering is the third: subinterpreters scale on *both*
builds, since each owns a GIL — PEP 734 delivers parallelism without waiting
for the free-threaded build to stabilize. Swap the payload for large arrays
and the ranking inverts completely, because pickling dominates and a process
pool becomes slower than one core.

Two columns require running this file twice, once with `PYTHON_GIL=1`. And
note the skip guard on the subinterpreter benchmark: creating even one
subinterpreter permanently breaks cuBLAS in the same process.
"""

import multiprocessing
import sys

import numpy as np
import pytest

from less_slow_00_shared import gpu_matmul_usable

# region: Parallelism

# ? Python has three ways to use more than one core, and since 3.14 they share
# ? one interface, so the same kernel can be pointed at all of them. What they
# ? do not share is what it costs to get data in and out.
# ?
# ? Threads were historically the wrong answer for CPU-bound work, because the
# ? GIL let only one run bytecode at a time. Free-threaded builds remove that.
# ? Subinterpreters, PEP 734, take a different route: each interpreter has its
# ? own GIL, so they scale on *either* build. Processes always scaled, and
# ? always paid for it at the boundary.
# ?
# ? Every other benchmark in this file defines its kernel as a closure inside
# ? the test. This chapter cannot — closures are not picklable and cannot cross
# ? into a subinterpreter, so the kernels below are module-level by necessity.

from concurrent.futures import (  # noqa: E402
    InterpreterPoolExecutor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)

PARALLEL_WORKERS = 8
PARALLEL_CHUNKS = 8
PARALLEL_LIMIT = 40_000


def count_primes(limit: int) -> int:
    """Trial division — allocation-free, so it measures compute, not the GC."""
    found = 0
    for candidate in range(2, limit):
        divisor = 2
        while divisor * divisor <= candidate:
            if candidate % divisor == 0:
                break
            divisor += 1
        else:
            found += 1
    return found


def sum_array(values) -> float:
    """Trivial work on a large payload, to expose the cost of moving it."""
    return float(np.asarray(values).sum())


def identity(value):
    """No work at all — whatever this costs is pure dispatch overhead."""
    return value


@pytest.fixture(scope="module")
def thread_pool():
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        list(pool.map(identity, range(PARALLEL_WORKERS)))  # warm the threads
        yield pool


@pytest.fixture(scope="module")
def process_pool():
    # ! 3.14 changed the Linux default from `fork` to `forkserver`; say which
    # ! one you meant, or the numbers are not comparable across versions.
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS, mp_context=context) as pool:
        list(pool.map(identity, range(PARALLEL_WORKERS)))  # pay the spawn cost
        yield pool


@pytest.fixture(scope="module")
def interpreter_pool():
    if gpu_matmul_usable:
        pytest.skip("subinterpreters break cuBLAS in the same process")
    with InterpreterPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        list(pool.map(identity, range(PARALLEL_WORKERS)))
        yield pool


@pytest.mark.benchmark(group="08-parallelism-cpu-bound")
def test_parallel_serial(benchmark):
    """One core, as the baseline every other row is measured against."""

    def kernel():
        return sum(count_primes(PARALLEL_LIMIT) for _ in range(PARALLEL_CHUNKS))

    result = benchmark(kernel)
    assert result == count_primes(PARALLEL_LIMIT) * PARALLEL_CHUNKS


@pytest.mark.benchmark(group="08-parallelism-cpu-bound")
def test_parallel_threads(benchmark, thread_pool):
    """Threads — real parallelism only when the GIL is disabled."""

    def kernel():
        return sum(thread_pool.map(count_primes, [PARALLEL_LIMIT] * PARALLEL_CHUNKS))

    result = benchmark(kernel)
    assert result == count_primes(PARALLEL_LIMIT) * PARALLEL_CHUNKS


# ! Creating even one subinterpreter permanently breaks cuBLAS in this
# ! process: the next `nvmath` matmul segfaults, whether the interpreter was
# ! destroyed first or left alive, and whether or not CUDA was initialized
# ! beforehand. Fifteen lines reproduce it without pytest. So on a machine
# ! with a working GPU this row is skipped rather than crashing the run —
# ! PEP 734 and the CUDA driver do not currently coexist.
@pytest.mark.skipif(
    gpu_matmul_usable, reason="subinterpreters break cuBLAS in the same process"
)
@pytest.mark.benchmark(group="08-parallelism-cpu-bound")
def test_parallel_interpreters(benchmark, interpreter_pool):
    """Subinterpreters — each carries its own GIL, so they scale either way."""

    def kernel():
        return sum(
            interpreter_pool.map(count_primes, [PARALLEL_LIMIT] * PARALLEL_CHUNKS)
        )

    result = benchmark(kernel)
    assert result == count_primes(PARALLEL_LIMIT) * PARALLEL_CHUNKS


@pytest.mark.benchmark(group="08-parallelism-cpu-bound")
def test_parallel_processes(benchmark, process_pool):
    """Processes — always parallel, and always paying at the boundary."""

    def kernel():
        return sum(process_pool.map(count_primes, [PARALLEL_LIMIT] * PARALLEL_CHUNKS))

    result = benchmark(kernel)
    assert result == count_primes(PARALLEL_LIMIT) * PARALLEL_CHUNKS


# ? The kernel above is deliberately tiny in bytes and heavy in cycles. Swap
# ? those around — a few megabytes per task, almost no arithmetic — and the
# ? ranking inverts, because now the boundary is the whole cost.


@pytest.mark.benchmark(group="08-parallelism-transfer")
def test_transfer_serial(benchmark):
    """Sum eight 16 MB arrays on one core, without moving anything."""
    arrays = [np.random.rand(2_000_000) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return sum(sum_array(values) for values in arrays)

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="08-parallelism-transfer")
def test_transfer_threads(benchmark, thread_pool):
    """Threads share the address space, so the arrays never move."""
    arrays = [np.random.rand(2_000_000) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return sum(thread_pool.map(sum_array, arrays))

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="08-parallelism-transfer")
def test_transfer_processes(benchmark, process_pool):
    """Processes must pickle 128 MB there and back, for one `sum` each."""
    arrays = [np.random.rand(2_000_000) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return sum(process_pool.map(sum_array, arrays))

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="08-parallelism-dispatch")
def test_dispatch_threads(benchmark, thread_pool):
    """Empty tasks, to find the floor below which parallelism cannot pay."""

    def kernel():
        return len(list(thread_pool.map(identity, range(PARALLEL_WORKERS))))

    result = benchmark(kernel)
    assert result == PARALLEL_WORKERS


@pytest.mark.benchmark(group="08-parallelism-dispatch")
def test_dispatch_processes(benchmark, process_pool):
    """The same empty tasks, across a process boundary."""

    def kernel():
        return len(list(process_pool.map(identity, range(PARALLEL_WORKERS))))

    result = benchmark(kernel)
    assert result == PARALLEL_WORKERS


# ! `fork` is deliberately absent. It is the fastest way to build a pool, and
# ! it cannot be measured here: by this point the suite holds a thread pool and
# ! a subinterpreter pool, and forking a process with threads breaks the child
# ! outright — `BrokenProcessPool`, reproducibly, not a warning. That is the
# ! reason 3.14 moved the Linux default to `forkserver`, and a benchmark that
# ! crashes is a firmer demonstration than a number would have been.
@pytest.mark.benchmark(group="08-parallelism-startup")
@pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
def test_startup_process_pool(benchmark, start_method: str):
    """Building a pool from scratch — paid once, but not always cheaply."""
    context = multiprocessing.get_context(start_method)

    def kernel():
        with ProcessPoolExecutor(
            max_workers=PARALLEL_WORKERS, mp_context=context
        ) as pool:
            return len(list(pool.map(identity, range(PARALLEL_WORKERS))))

    # ! Pool construction is the subject here, so it must sit inside the timed
    # ! callable — and it is far too slow to autocalibrate against.
    result = benchmark.pedantic(kernel, iterations=1, rounds=3)
    assert result == PARALLEL_WORKERS


# ? There is no `sys._set_gil_enabled`, so the two columns come from running
# ? the file twice on the *same* free-threaded binary, which holds allocator,
# ? compiler and object layout constant:
# ?
# ?     PYTHON_GIL=1 uv run pytest less_slow_08_parallelism.py
# ?                  uv run pytest less_slow_08_parallelism.py
# ?
# ? Intel Xeon 4 · CPython 3.14t · 8 workers on 16 vCPUs, 8 cores plus SMT
# ?
# ?                        GIL off            GIL on
# ?   serial            213.8 ms  1.00x    215.2 ms  1.00x
# ?   threads            52.0 ms  4.11x    226.1 ms  0.95x   below serial
# ?   subinterpreters    52.2 ms  4.10x     53.2 ms  4.05x   unaffected
# ?   processes          52.5 ms  4.07x     52.5 ms  4.10x   unaffected
# ?
# ? Threads going from useless to useful is the expected headline. The row
# ? worth remembering is the third: subinterpreters scale on *both* builds,
# ? because each owns a GIL. PEP 734 delivers parallelism without waiting for
# ? free-threading. Processes never moved — they had no interpreter to share.
# ?
# ? Serial cost is unchanged across builds, so the free-threading tax does not
# ? show up on this kernel at all. It is real on other shapes, in both
# ? directions, and not measured here — treat "5-10% slower" as a headline
# ? rather than a number you can rely on.
# ?
# ? Intel Xeon 4 · CPython 3.14t · 8 arrays of 16 MB, almost no arithmetic
# ?
# ?   threads              1.68 ms    1.00x  nothing moves
# ?   serial              11.81 ms    7.03x
# ?   processes          403.48 ms  240.17x  pickles 128 MB twice
# ?
# ? Pickling 128 MB there and back dwarfs the work. "Just use multiprocessing
# ? for CPU-bound work" quietly assumes a small payload; the zero-copy chapter
# ? above has the fix.
# ?
# ? And the floor: eight empty tasks cost 0.090 ms through a warm thread pool
# ? and 0.821 ms through a warm process pool, so anything finishing in under a
# ? few milliseconds measures the queue, not the parallelism.
# ?
# ? Intel Xeon 4 · CPython 3.14t · building an 8-worker pool from scratch
# ?
# ?   forkserver        1'262 ms    1.00x  workers re-import this module
# ?   spawn             1'369 ms    1.08x  a fresh interpreter each
# ?
# ? Startup is paid once, but 1.3 seconds is not nothing, and the cause is this
# ? file: every worker re-imports it, dragging in NumPy, Pandas, PyArrow, Numba
# ? and FastAPI. `fork` would skip all of that by copying the parent wholesale
# ? — and cannot be benchmarked here at all, because this process has threads
# ? by now and forking it breaks the child. See the note above the benchmark.

# endregion: Parallelism
