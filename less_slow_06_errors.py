#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Error handling — exceptions, wrapper objects, and plain status tuples.

Three ways to report that a four-step pipeline failed, measured across failure
rates from never to half the time. The ranking is not fixed: `raise` is the
fastest option available below a failure rate of about 15%, and the slowest
above 20%. Only its row moves, because it is the only style whose error path
costs more than its success path.

"Exceptions are slow" is therefore a claim about a workload rather than about
a language feature, and the workload term is the failure rate — the one number
most benchmarks of this question hold fixed and never report.

The wrapper objects are a separate lesson: an `Expected` class costs 3-4x a
bare status tuple carrying the same two values, and a `NamedTuple` costs
nearly as much. That multiple buys field names, not error handling.
"""

import random
from typing import NamedTuple, Tuple

import pytest

# region: Error Handling

# region: A Pipeline That Fails

# ? Control flow gets messy as soon as a program touches anything outside the
# ? CPU. A four-step pipeline — read a value, parse it, increment it, write it
# ? back — has four places to fail, and every I/O call is a coin flip whose
# ? bias nobody knows until production.
# ?
# ? Three ways to report that:
# ?
# ?   raise         cost on the failing path only, zero on the happy path
# ?   Expected      a wrapper object per step, whether or not it failed
# ?   (value, code) a tuple per step, unpacked by the caller
# ?
# ? Which wins depends on how often the coin comes up tails, so the failure
# ? rate is a parameter here rather than a constant baked into the fixtures.

STAGE_COUNT = 4
FAIL_RATES = [0.0, 0.01, 0.1, 0.5]
PIPELINE_RUNS = 1_000


def _failure_schedule(fail_rate: float, runs: int = PIPELINE_RUNS, seed: int = 0):
    """Which stage fails on each run, or -1 for a clean run.

    Shared by all three implementations so they see identical failures in an
    identical order. At most one stage fails per run, so `fail_rate` is the
    fraction of pipeline runs that fail rather than a per-stage probability.
    """
    generator = random.Random(seed)
    return [
        generator.randrange(STAGE_COUNT) if generator.random() < fail_rate else -1
        for _ in range(runs)
    ]


# ! Every implementation decides failure with the same `failing == N` integer
# ! compare, and raises with a constant message. Both matter: a variant that
# ! detected failure with `int()` inside a `try` while its rivals used
# ! `str.isnumeric()` would be measuring the predicate, and an f-string in the
# ! error message would charge formatting to the raising style alone.

# region: Raising


def read_or_raise(failing: int) -> str:
    if failing == 0:
        raise RuntimeError("read failed")
    return "1"


def parse_or_raise(value: str, failing: int) -> int:
    if failing == 1:
        raise ValueError("parse failed")
    return int(value)


def increment_or_raise(value: int, failing: int) -> str:
    if failing == 2:
        raise RuntimeError("increment failed")
    return str(value + 1)


def write_or_raise(value: str, failing: int) -> None:
    if failing == 3:
        raise RuntimeError("write failed")


def increment_file_or_raise(failing: int) -> None:
    read_value = read_or_raise(failing)
    parsed = parse_or_raise(read_value, failing)
    incremented = increment_or_raise(parsed, failing)
    write_or_raise(incremented, failing)


@pytest.mark.parametrize("fail_rate", FAIL_RATES)
@pytest.mark.benchmark(group="06-errors-styles")
def test_errors_raise(benchmark, fail_rate):
    """Exceptions: nothing on the happy path, a stack unwind on the unhappy one."""
    schedule = _failure_schedule(fail_rate)

    def runner():
        for failing in schedule:
            try:
                increment_file_or_raise(failing)
            except Exception:
                pass

    benchmark(runner)
    benchmark.extra_info["fail_rate"] = fail_rate


# ? The raising version pays nothing to report success — a function that
# ? returns normally has no error channel to populate. The other two pay on
# ? every call, which is the trade the rest of this section prices.

# endregion: Raising

# region: Wrapper Objects

# ? The alternative is to make the error a return value. Every step hands back
# ? an object that is either a result or a failure, and the caller checks
# ? before continuing:
# ?
# ?   raise       read → parse → increment → write        4 calls
# ?               └─ any failure unwinds to the handler
# ?
# ?   Expected    read → check → parse → check → …        4 calls, 4 checks,
# ?               each step allocates a wrapper           4 allocations
# ?
# ? This is `std::expected`, `Result<T, E>`, and Go's `(value, err)` — the
# ? shape languages without cheap exceptions converge on. The check is explicit
# ? and the control flow is local, which is the point. What it costs in Python
# ? is one object construction per step, on success as much as on failure.

from enum import Enum, auto  # noqa: E402


class Status(Enum):
    SUCCESS = auto()
    READ_FAILED = auto()
    PARSE_FAILED = auto()
    INCREMENT_FAILED = auto()
    WRITE_FAILED = auto()


class Expected:
    """A result-or-error wrapper, the shape `std::expected` popularized."""

    __slots__ = ("value", "error")

    def __init__(self, value=None, error: Status = None):
        self.value = value
        self.error = error

    def is_ok(self) -> bool:
        return self.error is None


def read_expected(failing: int) -> Expected:
    if failing == 0:
        return Expected(error=Status.READ_FAILED)
    return Expected(value="1")


def parse_expected(value: str, failing: int) -> Expected:
    if failing == 1:
        return Expected(error=Status.PARSE_FAILED)
    return Expected(value=int(value))


def increment_expected(value: int, failing: int) -> Expected:
    if failing == 2:
        return Expected(error=Status.INCREMENT_FAILED)
    return Expected(value=str(value + 1))


def write_expected(value: str, failing: int) -> Status:
    if failing == 3:
        return Status.WRITE_FAILED
    return Status.SUCCESS


def increment_file_expected(failing: int) -> Status:
    read_result = read_expected(failing)
    if not read_result.is_ok():
        return read_result.error
    parsed = parse_expected(read_result.value, failing)
    if not parsed.is_ok():
        return parsed.error
    incremented = increment_expected(parsed.value, failing)
    if not incremented.is_ok():
        return incremented.error
    return write_expected(incremented.value, failing)


@pytest.mark.parametrize("fail_rate", FAIL_RATES)
@pytest.mark.benchmark(group="06-errors-styles")
def test_errors_expected(benchmark, fail_rate):
    """A wrapper instance per step, allocated whether or not anything failed."""
    schedule = _failure_schedule(fail_rate)

    def runner():
        for failing in schedule:
            increment_file_expected(failing)

    benchmark(runner)
    benchmark.extra_info["fail_rate"] = fail_rate


# endregion: Wrapper Objects

# region: Status Tuples

# ? A wrapper class is not the only way to return two things. Python returns
# ? tuples natively, and `BUILD_TUPLE` is a single opcode against a Python-level
# ? `__init__`:
# ?
# ?   Expected(value, error)    call __init__ → two attribute stores
# ?   (value, code)             BUILD_TUPLE
# ?   Result(value, code)       call __new__ → BUILD_TUPLE → descriptors
# ?
# ? The third row exists to isolate one variable. `Result` is a `NamedTuple`
# ? carrying exactly the pair the bare tuple carries, so whatever separates
# ? them is the price of the field names alone.

StatusCode = int
STATUS_SUCCESS = 0
STATUS_READ_FAILED = 1
STATUS_PARSE_FAILED = 2
STATUS_INCREMENT_FAILED = 3
STATUS_WRITE_FAILED = 4


def read_status(failing: int) -> Tuple[str, StatusCode]:
    if failing == 0:
        return None, STATUS_READ_FAILED
    return "1", STATUS_SUCCESS


def parse_status(value: str, failing: int) -> Tuple[int, StatusCode]:
    if failing == 1:
        return None, STATUS_PARSE_FAILED
    return int(value), STATUS_SUCCESS


def increment_status(value: int, failing: int) -> Tuple[str, StatusCode]:
    if failing == 2:
        return None, STATUS_INCREMENT_FAILED
    return str(value + 1), STATUS_SUCCESS


def write_status(value: str, failing: int) -> StatusCode:
    if failing == 3:
        return STATUS_WRITE_FAILED
    return STATUS_SUCCESS


def increment_file_status(failing: int) -> StatusCode:
    read_value, code = read_status(failing)
    if code != STATUS_SUCCESS:
        return code
    parsed, code = parse_status(read_value, failing)
    if code != STATUS_SUCCESS:
        return code
    incremented, code = increment_status(parsed, failing)
    if code != STATUS_SUCCESS:
        return code
    return write_status(incremented, failing)


# ? The unpacking above is the tuple version's real ergonomic cost: `code` is
# ? rebound at every step, and a caller who forgets to check it gets `None`
# ? flowing silently into the next stage. Naming the fields fixes that, and the
# ? next four functions are the identical pipeline with names attached.


class Result(NamedTuple):
    """The same pair, named — to separate the wrapper idea from its cost."""

    value: object
    code: StatusCode


def read_named(failing: int) -> Result:
    if failing == 0:
        return Result(None, STATUS_READ_FAILED)
    return Result("1", STATUS_SUCCESS)


def parse_named(value: str, failing: int) -> Result:
    if failing == 1:
        return Result(None, STATUS_PARSE_FAILED)
    return Result(int(value), STATUS_SUCCESS)


def increment_named(value: int, failing: int) -> Result:
    if failing == 2:
        return Result(None, STATUS_INCREMENT_FAILED)
    return Result(str(value + 1), STATUS_SUCCESS)


def increment_file_named(failing: int) -> StatusCode:
    read_result = read_named(failing)
    if read_result.code != STATUS_SUCCESS:
        return read_result.code
    parsed = parse_named(read_result.value, failing)
    if parsed.code != STATUS_SUCCESS:
        return parsed.code
    incremented = increment_named(parsed.value, failing)
    if incremented.code != STATUS_SUCCESS:
        return incremented.code
    return write_status(incremented.value, failing)


@pytest.mark.parametrize("fail_rate", FAIL_RATES)
@pytest.mark.benchmark(group="06-errors-styles")
def test_errors_status(benchmark, fail_rate):
    """A bare tuple per step — `BUILD_TUPLE`, with no class involved."""
    schedule = _failure_schedule(fail_rate)

    def runner():
        for failing in schedule:
            increment_file_status(failing)

    benchmark(runner)
    benchmark.extra_info["fail_rate"] = fail_rate


@pytest.mark.parametrize("fail_rate", FAIL_RATES)
@pytest.mark.benchmark(group="06-errors-styles")
def test_errors_named(benchmark, fail_rate):
    """The same tuple with field names, to price the names alone."""
    schedule = _failure_schedule(fail_rate)

    def runner():
        for failing in schedule:
            increment_file_named(failing)

    benchmark(runner)
    benchmark.extra_info["fail_rate"] = fail_rate


# endregion: Status Tuples

# region: Which One, and When

# ? Intel Xeon 4 • CPython 3.14t • timeit, 1'000 pipeline runs of four stages
# ?
# ?                       never       1%       10%       50%
# ?   raise               175 µs    187 µs    204 µs    300 µs
# ?   status tuple        219 µs    224 µs    225 µs    187 µs
# ?   NamedTuple          681 µs    681 µs    653 µs    587 µs
# ?   Expected object     728 µs    730 µs    697 µs    627 µs
# ?
# ? Only one row moves with the parameter. `raise` costs 175 µs when nothing
# ? fails and 300 µs when half the runs do, because that is the only style
# ? whose error path costs more than its success path. The other three pay the
# ? same either way — a tuple is built per step regardless — which is why their
# ? rows are flat, and why they get slightly *faster* at 50%: a failing run
# ? short-circuits and never reaches the later stages.


# ? So the ranking is not a property of the styles. It has a crossover:
# ?
# ? Intel Xeon 4 • CPython 3.14t • timeit, raise against status tuple
# ?
# ?   10%    raise    210 µs    tuple    236 µs    raise wins   0.89x
# ?   15%    raise    220 µs    tuple    233 µs    raise wins   0.94x
# ?   20%    raise    236 µs    tuple    221 µs    tuple wins   1.07x
# ?   30%    raise    260 µs    tuple    216 µs    tuple wins   1.20x
# ?   40%    raise    286 µs    tuple    209 µs    tuple wins   1.37x
# ?
# ? Somewhere between 15% and 20%. Below it, exceptions are the fastest option
# ? available; above it, they are not. Almost every real pipeline sits far
# ? below that line, which is the practical answer: use exceptions, and stop
# ? treating "exceptions are slow" as though it were unconditional.


# ? The two wrapper rows are a separate lesson, and not about errors at all.
# ? `Expected` costs 3.35–4.16x the bare tuple, and a `NamedTuple` carrying the
# ? identical pair costs 3.14–3.88x — so nearly the whole gap survives when the
# ? only thing left is the field names.
# ?
# ? That multiple buys a Python-level `__new__` and a descriptor per attribute.
# ? The wrapper *pattern* is not what costs; constructing a Python object per
# ? call is. `std::expected` is cheap in C++ for precisely the reason it is
# ? expensive here — there it is a layout, not an allocation.

# endregion: Which One, and When

# endregion: Error Handling
