#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Memory — allocation pressure, the collector, and who owns the bytes.

Two halves with opposite conclusions. The first measures the advice everyone
has heard — reuse containers, switch off the collector — and finds both worth
nothing on CPython 3.14: reuse is 1.6x *slower* than the allocation it avoids,
because it trades one opcode for two method calls, and `gc.disable()` changes
nothing even for cyclic garbage.

The second half is where the orders of magnitude live, and none of it is a
micro-optimization. Slicing `bytes` copies and slicing a `memoryview` does
not, so the gap between them grows without bound with the slice length.
`pickle` protocol 5 draws the same line across a process boundary, which is
what makes shipping arrays to worker processes survivable. The lesson is that
allocation is not worth tuning and copying is worth eliminating.
"""

import gc

import numpy as np
import pytest

# region: Memory, GC, and Allocations

# ? Two pieces of advice have followed Python around for twenty years: reuse
# ? containers instead of allocating fresh ones, and switch the collector off
# ? around bulk work. Both were worth real money once.
# ?
# ? Three loops over the same ten thousand iterations, differing only in what
# ? they allocate and what they keep:
# ?
# ?   reuse       one list, cleared and refilled     0 allocations, 0 survivors
# ?   discard     a fresh list, dropped immediately  n allocations, 0 survivors
# ?   retain      a fresh list, kept in a container  n allocations, n survivors


@pytest.mark.benchmark(group="03-memory-allocation")
def test_alloc_reuse(benchmark):
    """One list, cleared and refilled — the allocator is never called."""

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
def test_alloc_discard(benchmark):
    """A fresh list per iteration, dropped at once — allocation without retention."""

    def kernel():
        total = 0
        for value in range(10_000):
            temporary = [value]
            total += temporary[0]
        return total

    result = benchmark(kernel)
    assert result == sum(range(10_000))


@pytest.mark.benchmark(group="03-memory-allocation")
def test_alloc_retain(benchmark):
    """The same allocations, kept alive — this is what fills the nursery."""

    def kernel():
        survivors = []
        total = 0
        for value in range(10_000):
            temporary = [value]
            survivors.append(temporary)
            total += temporary[0]
        return total

    result = benchmark(kernel)
    assert result == sum(range(10_000))


# ? Intel Xeon 4 • CPython 3.14t • timeit, 10K iterations
# ?
# ?   discard immediately     372 µs    1.00x  allocates, keeps nothing
# ?   reuse, clear + append   596 µs    1.60x  allocates nothing
# ?   retain in a list        611 µs    1.64x  allocates and keeps
# ?
# ? Reuse loses to the allocation it was supposed to avoid. `[value]` is one
# ? `BUILD_LIST` opcode; `buf.clear()` then `buf.append(value)` is two attribute
# ? lookups and two calls. The technique trades allocator work for interpreter
# ? work, and on CPython the allocator work was the cheaper of the two — small
# ? objects come off a free list, which is a pop from a stack.
# ?
# ? Retention is nearly as quiet: the 1.64x row is mostly that extra `append`,
# ? and controlling for it leaves 1.14x for keeping ten thousand objects alive.


# ? Matching the operation counts removes the effect altogether. Rebuilding a
# ? list a thousand times, `buf.clear(); buf.extend(src)` against `list(src)`:
# ?
# ? Intel Xeon 4 • CPython 3.14t • timeit, 1000 rebuilds
# ?
# ?   1 element        fresh wins   1.11x
# ?   10 elements      reuse wins   1.04x
# ?   100 elements     reuse wins   1.05x
# ?   1'000 elements   reuse wins   1.04x
# ?
# ? Noise at every size, in both directions. There is no allocation cost left
# ? to recover, which is the whole answer: the free lists already did it.


# ! The collector state is toggled outside the timed kernel — flipping it
# ! inside would measure `gc.disable()` alongside the loop it is meant to
# ! isolate.
@pytest.mark.parametrize("collector", ["enabled", "disabled"])
@pytest.mark.benchmark(group="03-memory-gc")
def test_gc_many_temporaries(benchmark, collector):
    """Short-lived tuples and dicts, with the collector on and off."""

    def kernel():
        total = 0
        for value in range(50_000):
            pair = (value, value + 1)
            mapping = {"a": pair[0], "b": pair[1]}
            total += mapping["a"]
        return total

    was_enabled = gc.isenabled()
    if collector == "disabled":
        gc.disable()
    try:
        result = benchmark(kernel)
    finally:
        if was_enabled:
            gc.enable()
    assert result > 0


# ? Intel Xeon 4 • CPython 3.14t • timeit, 50K short-lived tuples and dicts
# ?
# ?   collector enabled     2.72 ms    1.00x
# ?   collector disabled    2.86 ms    0.95x  no faster, and slightly noisier
# ?
# ? Intel Xeon 4 • CPython 3.14t • timeit, 20K reference cycles
# ?
# ?   collector enabled     4.51 ms    1.00x
# ?   collector disabled    4.49 ms    1.00x
# ?
# ? Nothing, even for cyclic garbage, which is the only kind the collector
# ? exists to handle. Reference counting frees the acyclic objects the instant
# ? they go out of scope — the collector never sees them — and the generational
# ? thresholds are not crossed often enough for the cyclic case to show up.
# ?
# ? Both techniques are dead on modern CPython, and the reason to know that is
# ? not the microseconds. It is that they are still recommended, still applied,
# ? and still make code worse to read in exchange for nothing. The wins in this
# ? chapter are all in the second half, and none of them is a micro-optimization.


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


# ? Intel Xeon 4 • CPython 3.14t • slicing 900 KB
# ?
# ?   memoryview slice      101 ns     1.00x  returns a window
# ?   bytes slice        20'095 ns   198.79x  returns a copy
# ?
# ? Read that top row skeptically — pytest-benchmark's own per-call cost is
# ? roughly 140 ns, so it mostly measures the stopwatch. Under `timeit` the
# ? slice is 45 ns and the ratio is 560x, not 199x. Near its floor a benchmark
# ? reports an upper bound on what you wanted to know.
# ?
# ? The ratio is not a constant either way — it is the slice length. A
# ? `memoryview` slice costs the same for a kilobyte or a gigabyte, so the gap
# ? widens without bound. That is a design property, not a tweak.


@pytest.mark.benchmark(group="03-memory-zero-copy")
def test_zerocopy_numpy_array_copies(benchmark):
    """`np.array` on a buffer allocates and copies by default."""
    buffer = bytes(8_000_000)
    # ! Wrapped outside the kernel so this times the copy alone. Calling
    # ! `frombuffer` inside would charge it to a benchmark it is not about.
    view = np.frombuffer(buffer, dtype=np.float64)

    def kernel():
        return np.array(view).nbytes

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


# ? Intel Xeon 4 • CPython 3.14t • moving an 8 MB array
# ?
# ?   np.frombuffer         504 ns       1.00x  wraps the same memory
# ?   pickle v5           4'105 ns       8.14x  buffer handed over
# ?   np.array copy     516'886 ns   1'025.6x   allocates and copies
# ?   pickle v4       1'044'907 ns   2'073.2x   array joins the stream
# ?
# ? `pickle` protocol 5 is the same trick across a process boundary. Protocol 4
# ? serializes the array data into the byte stream; protocol 5 hands the buffer
# ? to the transport and writes a header referring to it. That 254x between
# ? them is what makes shipping arrays to worker processes survivable at all.
# ?
# ? The trap in the top row: `np.frombuffer` gives a read-only view, `np.array`
# ? a writable copy, and the only visible difference is which one you typed.

# endregion: Zero-Copy and Buffers
