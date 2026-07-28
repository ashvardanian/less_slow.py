#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Control flow — what an idiom costs inside a tight loop.

Python's truthiness check is a slot lookup; `len(value) > 0` is a function
call and a comparison. In a loop of ten thousand iterations that difference
is measurable, and it runs the same direction as readability: `if value:` is
both the idiomatic spelling and the fast one, by roughly 1.5x.

The wider lesson is that in CPython the interpreter's dispatch cost usually
dwarfs the operation being dispatched, so micro-choices show up as constant
factors rather than asymptotics. Only the top row of the table travels
between machines; the exact ratios do not.
"""
import pytest

# region: Control Flow and Micro-Patterns

# ? Small control-flow choices add up in tight loops. Let's compare
# ? common idioms for checking emptiness and incrementing counters.


def _build_emptiness_sequence(length: int = 10_000):
    """Alternating empty and non-empty lists to test truthiness paths."""
    sequence = []
    for index in range(length):
        sequence.append([] if (index & 1) == 0 else [1])
    return sequence


@pytest.mark.benchmark(group="01-basics-control-flow")
def test_cf_if_len_increment(benchmark):
    values = _build_emptiness_sequence()

    def kernel():
        counter = 0
        for value in values:
            if len(value) > 0:
                counter += 1
        return counter

    result = benchmark(kernel)
    assert result == len(values) // 2


@pytest.mark.benchmark(group="01-basics-control-flow")
def test_cf_add_bool_len_cmp(benchmark):
    values = _build_emptiness_sequence()

    def kernel():
        counter = 0
        for value in values:
            counter += len(value) > 0
        return counter

    result = benchmark(kernel)
    assert result == len(values) // 2


@pytest.mark.benchmark(group="01-basics-control-flow")
def test_cf_if_truthy_increment(benchmark):
    values = _build_emptiness_sequence()

    def kernel():
        counter = 0
        for value in values:
            if value:
                counter += 1
        return counter

    result = benchmark(kernel)
    assert result == len(values) // 2


@pytest.mark.benchmark(group="01-basics-control-flow")
def test_cf_add_bool_truthy(benchmark):
    values = _build_emptiness_sequence()

    def kernel():
        counter = 0
        for value in values:
            counter += bool(value)
        return counter

    result = benchmark(kernel)
    assert result == len(values) // 2


# ? Intel Xeon 4 · CPython 3.14t · 10K iterations, alternating empty/non-empty
# ?
# ?   if value                    151 µs    1.00x  __bool__ is fast
# ?   if len(value) > 0           230 µs    1.52x  an extra call
# ?   counter += len(value) > 0   298 µs    1.97x  len, compare, bool
# ?   counter += bool(value)      311 µs    2.06x  explicit conversion
# ?
# ? Python's truthiness check (`if value`) is highly optimized. Using explicit
# ? `len()` or `bool()` calls adds overhead. The idiomatic `if value:` pattern
# ? is not just cleaner—it's measurably faster than alternatives.
# ?
# ? Only the top row is portable. An Apple M2 Pro on CPython 3.12 put the gap
# ? at 1.6x rather than 1.52x and swapped the bottom two rows — see the tables
# ? section for a case where a version bump inverts a ranking outright.


# endregion: Control Flow and Micro-Patterns
