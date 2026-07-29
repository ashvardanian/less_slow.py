#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Basics — what an idiom costs inside a tight loop, and what it no longer does.

Four micro-lessons, two of which are no longer true. `if value:` really is
1.52x faster than `if len(value) > 0`, and `"".join(parts)` really is 5.4x
faster than `+=` in a loop. But hoisting `out.append` into a local — the
canonical CPython trick — is now 1.13x *slower* than not doing it, and local
against global against attribute lookup spans 7%, not the 2x that folklore
promises. The adaptive interpreter caches what the trick used to avoid.

The `+=` result is worth stating precisely, because the usual claim is that it
is quadratic. It is not: CPython resizes the buffer in place when the string
has one reference, so the loop stays linear and the 5.4x is a constant factor.
That is the shape of nearly everything here — in CPython the interpreter's
dispatch cost dwarfs the operation dispatched, so idiom choices move constants
rather than complexity.
"""

import pytest

# region: Truthiness

# ? `if value:` and `if len(value) > 0` ask the same question of a list, and
# ? the bytecode is where they differ:
# ?
# ?   if value:            POP_JUMP_IF_FALSE      → the type's length slot
# ?   if len(value) > 0:   LOAD_GLOBAL len        → a name lookup
# ?                        CALL                   → a call frame
# ?                        COMPARE_OP             → then the same slot
# ?
# ? Both end up reading the same number. The second one takes a detour through
# ? a global lookup and a call to get there, and `len` is a global precisely
# ? because any module may rebind it.


def _build_emptiness_sequence(length: int = 10_000):
    """Alternating empty and non-empty lists to test truthiness paths."""
    sequence = []
    for index in range(length):
        sequence.append([] if (index & 1) == 0 else [1])
    return sequence


@pytest.mark.benchmark(group="01-basics-truthiness")
def test_truthiness_implicit(benchmark):
    """The idiomatic spelling."""
    values = _build_emptiness_sequence()

    def kernel():
        counter = 0
        for value in values:
            if value:
                counter += 1
        return counter

    result = benchmark(kernel)
    assert result == len(values) // 2


@pytest.mark.benchmark(group="01-basics-truthiness")
def test_truthiness_explicit_len(benchmark):
    """The spelling that says what it means, and pays for saying it."""
    values = _build_emptiness_sequence()

    def kernel():
        counter = 0
        for value in values:
            if len(value) > 0:
                counter += 1
        return counter

    result = benchmark(kernel)
    assert result == len(values) // 2


# ? Intel Xeon 4 • CPython 3.14t • 10K iterations, alternating empty/non-empty
# ?
# ?   if value             151 µs    1.00x  reads the length slot
# ?   if len(value) > 0    230 µs    1.52x  a global lookup and a call first
# ?
# ? Note what the win is not. A list has no `__bool__`, so `if value:` does not
# ? find a faster truth test — `PyObject_IsTrue` falls through to the same
# ? length slot `len()` reads. The 1.52x is entirely the `LOAD_GLOBAL`, the
# ? call frame, and the comparison, none of which produce information.
# ?
# ? Which is the useful form of this lesson: the idiomatic spelling is faster
# ? because it asks the interpreter for less, not because Python special-cases
# ? it. Only the direction travels between machines; an Apple M2 Pro on 3.12
# ? put the gap at 1.6x.

# endregion: Truthiness

# region: Name Lookup

# ? Every name in a hot loop is resolved on every iteration, and where it lives
# ? decides how:
# ?
# ?   sin(x)        LOAD_FAST      a slot in the frame
# ?   sin(x)        LOAD_GLOBAL    a dict lookup in module then builtins
# ?   math.sin(x)   LOAD_ATTR      a global lookup, then a dict on the module
# ?
# ? The traditional advice follows straight from that list: bind what you use
# ? to a local before the loop. It was worth about 2x for years.


import math  # noqa: E402

LOOKUP_ROUNDS = 10_000


@pytest.mark.benchmark(group="01-basics-lookup")
def test_lookup_local(benchmark):
    """Bound to a local before the loop — what the folklore recommends."""

    def kernel():
        sine = math.sin
        total = 0.0
        for _ in range(LOOKUP_ROUNDS):
            total += sine(0.5)
        return total

    assert benchmark(kernel) > 0


@pytest.mark.benchmark(group="01-basics-lookup")
def test_lookup_global(benchmark):
    """Resolved from the module namespace on every call."""

    def kernel():
        total = 0.0
        for _ in range(LOOKUP_ROUNDS):
            total += _module_sine(0.5)
        return total

    assert benchmark(kernel) > 0


@pytest.mark.benchmark(group="01-basics-lookup")
def test_lookup_attribute(benchmark):
    """A global lookup and then an attribute lookup, every call."""

    def kernel():
        total = 0.0
        for _ in range(LOOKUP_ROUNDS):
            total += math.sin(0.5)
        return total

    assert benchmark(kernel) > 0


_module_sine = math.sin


@pytest.mark.benchmark(group="01-basics-lookup")
def test_lookup_append_direct(benchmark):
    """The method looked up on the object each time."""

    def kernel():
        out = []
        for index in range(LOOKUP_ROUNDS):
            out.append(index)
        return len(out)

    assert benchmark(kernel) == LOOKUP_ROUNDS


@pytest.mark.benchmark(group="01-basics-lookup")
def test_lookup_append_hoisted(benchmark):
    """The bound method hoisted into a local — the classic optimization."""

    def kernel():
        out = []
        push = out.append
        for index in range(LOOKUP_ROUNDS):
            push(index)
        return len(out)

    assert benchmark(kernel) == LOOKUP_ROUNDS


@pytest.mark.benchmark(group="01-basics-lookup")
def test_lookup_comprehension(benchmark):
    """No loop body to dispatch at all."""

    def kernel():
        out = [index for index in range(LOOKUP_ROUNDS)]
        return len(out)

    assert benchmark(kernel) == LOOKUP_ROUNDS


# ? Intel Xeon 4 • CPython 3.14t • timeit, 10K calls
# ?
# ?   local        372.7 µs    1.00x
# ?   global       387.5 µs    1.04x
# ?   attribute    398.2 µs    1.07x
# ?
# ? Seven percent across all three. The advice is dead: since 3.11 the adaptive
# ? interpreter rewrites `LOAD_GLOBAL` and `LOAD_ATTR` into inline-cached forms
# ? after a few executions, which is exactly the lookup the local binding was
# ? there to skip.


# ? Hoisting a method is worse than dead — it now costs:
# ?
# ? Intel Xeon 4 • CPython 3.14t • timeit, building a 10K list
# ?
# ?   comprehension        166.8 µs    1.00x  no per-item dispatch
# ?   out.append(index)    302.9 µs    1.82x  specialized method call
# ?   push = out.append    341.2 µs    2.05x  a bound-method object
# ?
# ? `out.append(i)` compiles to a specialized attribute-and-call pair that the
# ? interpreter fuses. Hoisting replaces it with a generic call through a bound
# ? method object, which defeats the specialization — so the optimization is
# ? 1.13x slower than not optimizing, and less readable.
# ?
# ? The comprehension wins because it removes the dispatch entirely rather than
# ? making it cheaper, which is the only one of these three that is still a
# ? real technique.

# endregion: Name Lookup

# region: Call Overhead

# ? Every number elsewhere in this suite is denominated in function calls, so
# ? it is worth knowing what one costs. A call builds a frame, binds arguments,
# ? executes, and tears the frame down — around an inlined expression that
# ? would otherwise be three bytecodes.


@pytest.mark.benchmark(group="01-basics-calls")
def test_calls_inlined(benchmark):
    """The arithmetic written where it is used."""

    def kernel():
        total = 0
        for index in range(LOOKUP_ROUNDS):
            total += index * 2 + 1
        return total

    assert benchmark(kernel) > 0


def _affine(value: int) -> int:
    return value * 2 + 1


@pytest.mark.benchmark(group="01-basics-calls")
def test_calls_via_function(benchmark):
    """The same arithmetic behind a name."""

    def kernel():
        total = 0
        for index in range(LOOKUP_ROUNDS):
            total += _affine(index)
        return total

    assert benchmark(kernel) > 0


# ? Intel Xeon 4 • CPython 3.14t • timeit, 10K iterations
# ?
# ?   inline expression    343.9 µs    1.00x
# ?   via a function       453.8 µs    1.32x   ≈11 ns per call
# ?
# ? Eleven nanoseconds is the unit that makes the rest of this suite legible.
# ? A generator resuming per item, an iterator's `__next__`, a property getter,
# ? a validator — each is at least one of these, and the counts are what
# ? separate the pipeline shapes elsewhere.
# ?
# ? It is also small enough to be the wrong thing to optimize. Inlining a
# ? function to save 11 ns is worth it only in a loop running millions of
# ? times, and every other lesson here moves more.

# endregion: Call Overhead

# region: String Building

# ? One case where the folklore predicts an asymptotic difference rather than a
# ? constant, and gets it wrong:
# ?
# ?   out += piece       allocate a new string, copy both  → O(n²), in theory
# ?   "".join(parts)     size the result once, copy once   → O(n)
# ?
# ? The theory assumes strings are immutable in practice as well as in the
# ? language. CPython cheats: when a string has exactly one reference, `+=`
# ? resizes the buffer in place instead of copying.


STRING_PIECES = 10_000


@pytest.mark.benchmark(group="01-basics-strings")
def test_strings_concatenate(benchmark):
    """Repeated `+=`, which the in-place resize keeps linear."""
    parts = ["abcdefgh"] * STRING_PIECES

    def kernel():
        out = ""
        for piece in parts:
            out += piece
        return len(out)

    assert benchmark(kernel) == STRING_PIECES * 8


@pytest.mark.benchmark(group="01-basics-strings")
def test_strings_join(benchmark):
    """One allocation of a known size, then one pass."""
    parts = ["abcdefgh"] * STRING_PIECES

    def kernel():
        return len("".join(parts))

    assert benchmark(kernel) == STRING_PIECES * 8


# ? Intel Xeon 4 • CPython 3.14t • timeit, building one string from n pieces
# ?
# ?              n=100     n=1'000    n=10'000    n=100'000
# ?   += loop     2.7 µs     26.0 µs    256.1 µs    2'575.9 µs
# ?   join        0.5 µs      4.7 µs     46.5 µs      479.1 µs
# ?   ratio       5.4x        5.6x        5.5x          5.4x
# ?
# ? Both rows are straight lines, and the ratio is flat at 5.4x across three
# ? orders of magnitude. `+=` is linear, not quadratic — the in-place resize
# ? holds — so this is a constant factor like the rest of these idioms.
# ?
# ? It stops holding the moment anything else references the string mid-loop:
# ? a debug list, a logging call, a closure capture. Then every append copies
# ? and the loop really is quadratic. So the reason to prefer `join` is not the
# ? 5.4x — it is that `join` cannot silently change complexity when someone
# ? adds a line.

# endregion: String Building
