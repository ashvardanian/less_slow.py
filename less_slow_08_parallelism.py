#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parallelism — threads, subinterpreters, and processes, with and without the GIL.

Since 3.14 all three share the executor interface, so one kernel drives them
all. Under the GIL, threads are slower than running serially. Free-threaded,
they scale. Subinterpreters scale on *both* builds, because each owns a GIL —
PEP 734 delivers parallelism without waiting for free-threading to stabilize.
Processes never cared either way, and always paid at the boundary: swap the
payload for large arrays and a process pool becomes 46x slower than one core,
because pickling 128 MB dwarfs the arithmetic. Putting the same arrays in
shared memory recovers 146x of that.

Two results here are about correctness rather than speed. Eight threads
incrementing one shared cell lose 77.5% of their updates under free-threading
and none under the GIL, so this is code that passes its tests until the
interpreter stops serializing it. And the reflexive fix is the slowest thing
in the chapter: a `threading.Lock` costs 3.8x more than the broken version it
repairs, and 117x more than giving each thread its own accumulator.

There is no `sys._set_gil_enabled`, so the two-column tables come from running
the suite twice on the same free-threaded binary, which holds allocator,
compiler and object layout constant:

    PYTHON_GIL=1 uv run pytest less_slow_08_parallelism.py
                 uv run pytest less_slow_08_parallelism.py
"""

import hashlib
import multiprocessing
import threading
from multiprocessing import shared_memory

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
# ? A closure cannot cross either boundary. Pickle refers to functions by
# ? qualified name, and a subinterpreter has its own module table, so anything
# ? handed to a worker must be findable at module level in a fresh import:
# ?
# ?   thread pool         any callable          shares the address space
# ?   process pool        module-level only     pickled by name
# ?   interpreter pool    module-level only     re-imported per interpreter
# ?
# ? That constraint shapes every parallel Python program, and it is why the
# ? kernels here are module-level functions rather than closures.

from concurrent.futures import (  # noqa: E402
    InterpreterPoolExecutor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)

PARALLEL_WORKERS = 8
PARALLEL_CHUNKS = 8
PARALLEL_LIMIT = 40_000
CONTENTION_STEPS = 200_000
ARRAY_ELEMENTS = 2_000_000


# ? Four kernels, chosen so that each isolates one cost:
# ?
# ?   count_primes      heavy in cycles, tiny in bytes    pure bytecode
# ?   sum_array         trivial arithmetic, 16 MB in      the boundary
# ?   hash_buffer       heavy in cycles, in C             a released GIL
# ?   bump_*            no data at all, one shared cell   contention
# ?
# ? `count_primes` uses trial division rather than a sieve on purpose: a sieve
# ? allocates, and allocation would put the memory allocator into a
# ? measurement about scheduling.


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


def bump_shared(box) -> None:
    """Increment one cell many times, from every thread at once."""
    # ! `box[0] += 1` is a load, an add and a store, with no lock around them.
    # ! Under free-threading the interleavings lose updates, which is why the
    # ! benchmark reports the final count instead of asserting on it.
    for _ in range(CONTENTION_STEPS):
        box[0] += 1


def bump_locked(pair) -> None:
    """The same shared cell, serialized by the fix everyone reaches for first."""
    box, lock = pair
    for _ in range(CONTENTION_STEPS):
        with lock:
            box[0] += 1


def bump_private(_) -> int:
    """The same increments, into a local nobody else can see."""
    total = 0
    for _ in range(CONTENTION_STEPS):
        total += 1
    return total


def hash_buffer(payload: bytes) -> str:
    """A C extension that drops the GIL, with no array library involved."""
    return hashlib.sha256(payload).hexdigest()


def sum_shared_block(name: str) -> float:
    """Attach to an existing shared buffer and reduce it — nothing is pickled."""
    block = shared_memory.SharedMemory(name=name)
    try:
        return float(
            np.ndarray(ARRAY_ELEMENTS, dtype=np.float64, buffer=block.buf).sum()
        )
    finally:
        block.close()


def identity(value):
    """No work at all — whatever this costs is pure dispatch overhead."""
    return value


# ? All three pools are module-scoped and warmed before use, which is a
# ? measurement decision rather than a convenience:
# ?
# ?   cold pool    spawn workers, import modules, then do the work
# ?   warm pool    hand work to threads that already exist
# ?
# ? Building a process pool costs seconds — measured at the end — so a
# ? benchmark that constructs one per repetition reports startup and calls it
# ? parallelism. Warming first isolates the steady-state cost, and the startup
# ? cost gets its own benchmark rather than contaminating every other row.


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


# ? Intel Xeon 4 • CPython 3.14t • 8 workers on 16 vCPUs, 8 cores plus SMT
# ?
# ?                        GIL off            GIL on
# ?   serial            213.8 ms  1.00x    215.2 ms  1.00x
# ?   threads            52.0 ms  4.11x    226.1 ms  0.95x   below serial
# ?   subinterpreters    52.2 ms  4.10x     53.2 ms  4.05x   unaffected
# ?   processes          52.5 ms  4.07x     52.5 ms  4.10x   unaffected
# ?
# ? Threads going from useless to useful is the expected headline. The third
# ? row is the one worth remembering: subinterpreters scale on *both* builds,
# ? because each owns a GIL. PEP 734 delivers this without waiting for
# ? free-threading to stabilize. Processes never moved — they had no
# ? interpreter to share in the first place.
# ?
# ? Serial cost is identical across builds, so free-threading taxes nothing on
# ? this kernel. That tax is real on other shapes and in both directions, and
# ? measuring it would need a different kernel.


# ? The kernel above is deliberately tiny in bytes and heavy in cycles. Swap
# ? those around — a few megabytes per task, almost no arithmetic — and the
# ? ranking inverts, because now the boundary is the whole cost.


@pytest.mark.benchmark(group="08-parallelism-transfer")
def test_transfer_serial(benchmark):
    """Sum eight 16 MB arrays on one core, without moving anything."""
    arrays = [np.random.rand(ARRAY_ELEMENTS) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return sum(sum_array(values) for values in arrays)

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="08-parallelism-transfer")
def test_transfer_threads(benchmark, thread_pool):
    """Threads share the address space, so the arrays never move."""
    arrays = [np.random.rand(ARRAY_ELEMENTS) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return sum(thread_pool.map(sum_array, arrays))

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="08-parallelism-transfer")
def test_transfer_processes(benchmark, process_pool):
    """Processes must pickle 128 MB there and back, for one `sum` each."""
    arrays = [np.random.rand(ARRAY_ELEMENTS) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return sum(process_pool.map(sum_array, arrays))

    result = benchmark(kernel)
    assert result > 0


# ? Intel Xeon 4 • CPython 3.14t • 8 arrays of 16 MB, almost no arithmetic
# ?
# ?   threads              1.58 ms    0.17x  nothing moves
# ?   serial               9.13 ms    1.00x
# ?   processes          419.07 ms   45.89x  pickles 128 MB twice
# ?
# ? A process pool is 46x slower than one core. "Just use multiprocessing for
# ? CPU-bound work" quietly assumes a small payload, and this one is 128 MB
# ? serialized on the way out and again on the way back, to perform eight
# ? additions.
# ?
# ? The fix is not to abandon processes. It is to stop copying — put the bytes
# ? where both sides can already see them.


@pytest.mark.benchmark(group="08-parallelism-transfer")
def test_transfer_shared_memory(benchmark, process_pool):
    """The same reductions, with the arrays in shared memory instead of pickled."""
    blocks = []
    names = []
    try:
        for _ in range(PARALLEL_CHUNKS):
            source = np.random.rand(ARRAY_ELEMENTS)
            block = shared_memory.SharedMemory(create=True, size=source.nbytes)
            np.ndarray(ARRAY_ELEMENTS, dtype=np.float64, buffer=block.buf)[:] = source
            blocks.append(block)
            names.append(block.name)

        def kernel():
            # ! Only the eight names cross the boundary. The 128 MB stays put,
            # ! and each worker maps it rather than receiving a copy.
            return sum(process_pool.map(sum_shared_block, names))

        assert benchmark(kernel) > 0
    finally:
        for block in blocks:
            block.close()
            block.unlink()


# ? Intel Xeon 4 • CPython 3.14t • the same 8 reductions, three ways across
# ?
# ?   threads              1.58 ms    1.00x  one address space
# ?   shared memory        2.88 ms    1.82x  eight names cross, not 128 MB
# ?   pickled              419.07 ms  264.8x
# ?
# ? Shared memory recovers 146x of the 265x. What crosses the boundary is eight
# ? strings; the pages are mapped into each worker, not copied to it.
# ?
# ? The residual 1.82x over threads is the mapping itself plus the dispatch,
# ? and it is the honest price of process isolation. Which is the useful
# ? conclusion: processes are not slow, pickling is, and the two are separable.


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


# ? Intel Xeon 4 • CPython 3.14t • 8 empty tasks through a warm pool
# ?
# ?   thread pool         69.9 µs    1.00x
# ?   process pool       384.2 µs    5.49x  a queue, a pipe, and a pickle
# ?
# ? That is the floor. A task finishing in under ~10 µs of real work cannot be
# ? worth dispatching to a thread, and under ~50 µs cannot be worth a process —
# ? the queue costs more than the work, and adding workers makes it worse.
# ?
# ? Both numbers assume a warm pool. Cold, the process figure is four orders of
# ? magnitude larger, which the startup benchmark further down measures
# ? directly.


# ? The kernel above is pure Python, which is the rarest thing anyone actually
# ? parallelizes. Most real Python concurrency calls into an extension that
# ? already drops the GIL, so measure that too before concluding anything.


@pytest.mark.benchmark(group="08-parallelism-released-gil")
def test_released_serial(benchmark):
    """Eight SHA-256 digests over 8 MB each, one after another."""
    payloads = [np.random.bytes(8_000_000) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return [hash_buffer(payload) for payload in payloads]

    assert len(benchmark(kernel)) == PARALLEL_CHUNKS


@pytest.mark.benchmark(group="08-parallelism-released-gil")
def test_released_threads(benchmark, thread_pool):
    """The same digests across threads — `hashlib` drops the GIL to compute them."""
    payloads = [np.random.bytes(8_000_000) for _ in range(PARALLEL_CHUNKS)]

    def kernel():
        return list(thread_pool.map(hash_buffer, payloads))

    assert len(benchmark(kernel)) == PARALLEL_CHUNKS


# ? Intel Xeon 4 • CPython 3.14t • 8 SHA-256 digests over 8 MB each
# ?
# ?   serial            40.55 ms    1.00x
# ?   threads            7.96 ms    5.10x  `hashlib` drops the GIL
# ?
# ? Which is the corrective to everything above. `hashlib` releases the GIL
# ? around its digest loop, so this threaded before free-threading existed and
# ? would thread on 3.9. The same is true of `zlib`, of NumPy's kernels, and of
# ? most of what a Python program spends its time in.
# ?
# ? If the hot loop is already inside a C extension, the GIL was never what was
# ? holding you back — which is worth establishing before rewriting anything
# ? around a free-threaded build.
# ?
# ? But once threads genuinely run at the same time, sharing state costs what
# ? it always cost. That was invisible while the GIL serialized them.


@pytest.mark.benchmark(group="08-parallelism-contention")
def test_contention_shared(benchmark, thread_pool):
    """Every thread incrementing one shared list cell, unsynchronized."""
    box = [0]

    def kernel():
        box[0] = 0
        list(thread_pool.map(bump_shared, [box] * PARALLEL_WORKERS))
        return box[0]

    observed = benchmark(kernel)
    expected = CONTENTION_STEPS * PARALLEL_WORKERS
    benchmark.extra_info["expected_total"] = expected
    benchmark.extra_info["observed_total"] = observed
    benchmark.extra_info["lost_updates"] = expected - observed
    # ! Deliberately not an equality assert. Free-threaded, this loses updates;
    # ! under the GIL it does not. The gap is the point, and it is reported
    # ! rather than enforced so the row survives on both builds.
    assert observed > 0


@pytest.mark.benchmark(group="08-parallelism-contention")
def test_contention_locked(benchmark, thread_pool):
    """The same shared cell behind a `threading.Lock`."""
    box = [0]
    lock = threading.Lock()

    def kernel():
        box[0] = 0
        list(thread_pool.map(bump_locked, [(box, lock)] * PARALLEL_WORKERS))
        return box[0]

    assert benchmark(kernel) == CONTENTION_STEPS * PARALLEL_WORKERS


@pytest.mark.benchmark(group="08-parallelism-contention")
def test_contention_private(benchmark, thread_pool):
    """The same work into per-thread accumulators, summed at the end."""

    def kernel():
        return sum(thread_pool.map(bump_private, range(PARALLEL_WORKERS)))

    assert benchmark(kernel) == CONTENTION_STEPS * PARALLEL_WORKERS


# ? Intel Xeon 4 • CPython 3.14t • 8 threads, 200K increments each, GIL off
# ?
# ?   private accumulators      7.47 ms     1.00x   correct
# ?   one shared cell         232.64 ms    31.16x   WRONG — see below
# ?   shared cell + Lock      875.51 ms   117.28x   correct
# ?
# ? The middle row is not merely slow, it is wrong:
# ?
# ?   expected    1'600'000    8 threads × 200'000 increments
# ?   observed      359'858
# ?   lost        1'240'142    77.5%
# ?
# ? `box[0] += 1` is a load, an add and a store with nothing holding them
# ? together. Free-threaded, the interleavings drop writes. Under the GIL the
# ? identical code is correct — which is the trap, because it passes its tests
# ? until the day the interpreter stops serializing it for you.


# ? The reflexive fix is the slowest row on the board. A `Lock` restores
# ? correctness at 117x the private version, and 3.8x worse than the broken
# ? code it repairs, because all 1.6 million increments now acquire and release
# ? while eight threads queue for one cell.
# ?
# ? So the way out is not to synchronize the sharing but to remove it: give
# ? each thread its own accumulator and combine once at the end. That is the
# ? top row — correct, and 117x faster than the careful version.
# ?
# ? This generalizes past counters. Whenever threads write to one location, the
# ? question to ask is not which lock to use but why they are all writing to
# ? one location.
# ?
# ! `fork` is absent from this sweep on purpose, not by oversight: by the time
# ! these run the suite holds a thread pool, and forking a threaded process
# ! leaves the child holding locks no thread will ever release. That hazard is
# ! precisely why 3.14 moved the Linux default to `forkserver`.
@pytest.mark.benchmark(group="08-parallelism-startup")
@pytest.mark.parametrize("start_method", ["forkserver", "spawn"])
def test_pool_startup(benchmark, start_method: str):
    """Building a pool from scratch — paid once per program, and not cheap."""
    context = multiprocessing.get_context(start_method)

    def kernel():
        with ProcessPoolExecutor(
            max_workers=PARALLEL_WORKERS, mp_context=context
        ) as pool:
            return len(list(pool.map(identity, range(PARALLEL_WORKERS))))

    # ! `pedantic` with one round: pool creation is not something to repeat
    # ! hundreds of times, and warm repetitions would measure a warm OS cache
    # ! rather than the cost a program actually pays on its first call.
    result = benchmark.pedantic(kernel, rounds=3, iterations=1)
    assert result == PARALLEL_WORKERS


# ? Intel Xeon 4 • CPython 3.14t • building an 8-worker pool from scratch
# ?
# ?   forkserver         3.32 s    1.00x  workers re-import this module
# ?   spawn              3.63 s    1.09x  a fresh interpreter each
# ?
# ? Three seconds, and the module is the cause. Every worker re-imports the
# ? module that defines its kernel, dragging in NumPy on the way. A
# ? pool is worth building once and keeping; building one per batch would cost
# ? more than most batches save.
# ?
# ? `fork` would avoid all of it by copying the parent wholesale, and is
# ? missing from the table for a reason worth knowing rather than an oversight
# ? — see the note above.

# endregion: Parallelism
