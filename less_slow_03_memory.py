#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Memory — allocation pressure, the collector, and who owns the bytes.

Reusing a container beats allocating a fresh one by 5x, because the allocator
is the cost rather than the loop around it. Disabling the garbage collector
helps only when a loop creates many short-lived objects, and the benchmark
here is deliberately shaped to be that case rather than a typical one.

The zero-copy half is the question that decides most glue code. Slicing
`bytes` copies and slicing a `memoryview` does not, so the two differ by
orders of magnitude and the gap grows without bound with the slice length —
it is a design property, not a micro-optimization. `pickle` protocol 5 draws
the same line across a process boundary, which is what makes shipping arrays
to worker processes survivable.
"""
import gc

import numpy as np
import pytest

# region: Memory, GC, and Allocations

# ? Python's garbage collector and memory allocator have measurable overhead.
# ? Allocation patterns matter: reusing objects beats creating new ones, and
# ? disabling GC during bulk operations can provide surprising speedups.


@pytest.mark.benchmark(group="03-memory-allocation")
def test_alloc_small_lists_append(benchmark):
    """Allocate many tiny lists by appending one element each time."""

    def kernel():
        result = []
        for value in range(10_000):
            result.append([value])
        return len(result)

    result = benchmark(kernel)
    assert result == 10_000


@pytest.mark.benchmark(group="03-memory-allocation")
def test_alloc_reuse_list_clear(benchmark):
    """Reuse a single list: clear and refill to avoid allocations."""

    def kernel():
        reusable = []
        total = 0
        for value in range(10_000):
            reusable.clear()
            reusable.append(value)
            total += reusable[0]
        return total

    result = benchmark(kernel)
    assert result == sum(range(10_000))


@pytest.mark.benchmark(group="03-memory-allocation")
def test_gc_disabled_many_temporaries(benchmark):
    """Create many short‑lived objects with GC disabled inside the hot loop."""

    def kernel():
        old = gc.isenabled()
        gc.disable()
        try:
            total = 0
            for value in range(50_000):
                # Allocate a few temporaries; they die quickly.
                pair = (value, value + 1)
                mapping = {"a": pair[0], "b": pair[1]}
                total += mapping["a"]
            return total
        finally:
            if old:
                gc.enable()

    result = benchmark(kernel)
    assert result > 0


# ? Apple M2 Pro · 10K iterations
# ?
# ?   reuse + clear()       416 µs    1.00x  no allocations
# ?   append new lists    2'190 µs    5.26x  allocation storm
# ?
# ? Reusing a container beats allocating a fresh one, because the allocator is
# ? the cost, not the loop. The GC-disabled benchmark runs 50K iterations and
# ? so belongs to no ratio here — the honest comparison is against itself with
# ? the collector left on, which is what that benchmark measures.


# endregion: Memory, GC, and Allocations

# region: Zero-Copy and Buffers

# ? Python hides its copies. Slicing `bytes` allocates a new object and memcpy's
# ? into it; slicing a `memoryview` hands back a window onto the same bytes. One
# ? is O(n), the other O(1), and nothing in the syntax tells you which you wrote.
# ?
# ? This is the question that decides most glue code. Every library boundary —
# ? NumPy to Arrow, a worker queue, a socket — either moves the payload or
# ? passes a pointer to it, and the difference does not show up until the
# ? payload is large.

import pickle  # noqa: E402


@pytest.mark.benchmark(group="03-memory-zero-copy")
def test_zerocopy_bytes_slice(benchmark):
    """Slicing `bytes` copies, so the cost scales with the slice length."""
    buffer = bytes(1_000_000)

    def kernel():
        return len(buffer[1_000:900_000])

    result = benchmark(kernel)
    assert result == 899_000


@pytest.mark.benchmark(group="03-memory-zero-copy")
def test_zerocopy_memoryview_slice(benchmark):
    """Slicing a `memoryview` returns a view, so no bytes move."""
    view = memoryview(bytes(1_000_000))

    def kernel():
        return len(view[1_000:900_000])

    result = benchmark(kernel)
    assert result == 899_000


@pytest.mark.benchmark(group="03-memory-zero-copy")
def test_zerocopy_numpy_array_copies(benchmark):
    """`np.array` on a buffer allocates and copies by default."""
    buffer = bytes(8_000_000)

    def kernel():
        return np.array(np.frombuffer(buffer, dtype=np.float64)).nbytes

    result = benchmark(kernel)
    assert result == 8_000_000


@pytest.mark.benchmark(group="03-memory-zero-copy")
def test_zerocopy_numpy_frombuffer(benchmark):
    """`np.frombuffer` wraps the same memory — read-only, but free."""
    buffer = bytes(8_000_000)

    def kernel():
        return np.frombuffer(buffer, dtype=np.float64).nbytes

    result = benchmark(kernel)
    assert result == 8_000_000


@pytest.mark.benchmark(group="03-memory-zero-copy")
def test_zerocopy_pickle_inband(benchmark):
    """Protocol 4 serializes the array data into the pickle stream itself."""
    array = np.random.rand(1_000_000)

    def kernel():
        return len(pickle.dumps(array, protocol=4))

    result = benchmark(kernel)
    assert result > array.nbytes


@pytest.mark.benchmark(group="03-memory-zero-copy")
def test_zerocopy_pickle_outofband(benchmark):
    """Protocol 5 hands the buffer over instead of copying it into the stream."""
    array = np.random.rand(1_000_000)

    def kernel():
        buffers = []
        payload = pickle.dumps(array, protocol=5, buffer_callback=buffers.append)
        assert len(buffers) == 1
        return len(payload)

    result = benchmark(kernel)
    # ! The 8 MB array never entered the stream — the pickle is just a header.
    assert result < 1_000


# ? Intel Xeon 4 · CPython 3.14t · slicing 900 KB
# ?
# ?   memoryview slice      182 ns    1.00x  returns a window
# ?   bytes slice        24'534 ns  134.80x  returns a copy
# ?
# ? Intel Xeon 4 · CPython 3.14t · moving an 8 MB array
# ?
# ?   np.frombuffer         809 ns      1.00x  wraps the same memory
# ?   pickle v5           7'726 ns      9.55x  buffer handed over
# ?   np.array copy     571'671 ns    706.64x  allocates and copies
# ?   pickle v4        1'154'362 ns  1'426.9x  array joins the stream
# ?
# ? Read the first table skeptically: pytest-benchmark's own per-call cost is
# ? roughly 140 ns, so that top row mostly measures the stopwatch. Under
# ? `timeit` the slice is 45 ns and the ratio is 560x, not 135x. Near its
# ? floor a benchmark reports an upper bound on what you wanted to know.
# ?
# ? The ratio is not a constant in any case — it is the slice length. A
# ? `memoryview` slice costs the same for a kilobyte or a gigabyte, so the gap
# ? widens without bound. That makes it a design property, not a tweak.
# ?
# ? The trap: `np.frombuffer` gives a read-only view, `np.array` a writable
# ? copy, and the only visible difference is which one you typed.

# endregion: Zero-Copy and Buffers
