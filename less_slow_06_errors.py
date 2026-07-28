#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Error handling — exceptions, wrapper objects, and plain status tuples.

Three ways to report that a five-step file pipeline failed. The surprise is
the ordering: the `Expected` wrapper written specifically to avoid exceptions
is the slowest of the three, because it allocates an object on every call,
while `raise` allocates only on the unhappy path. Plain tuples win by doing
neither.

Which means the choice is not "exceptions are slow, avoid them" but "how
often does this actually fail, and what does the happy path allocate?" — a
question with a different answer per call site.
"""

from typing import Tuple

import pytest

# region: Error Handling

# region: Errors

# ?  In the real world, control-flow gets messy, as different methods will
# ?  break in different places. Let's imagine a system, that:
# ?
# ?  - Reads an integer from a text file.
# ?  - Increments it.
# ?  - Saves it back to the text file.
# ?
# ?  As soon as we start dealing with "external devices", as opposed to the CPU itself,
# ?  failures become regular. The file may not exist, the integer may not be a number,
# ?  the file may be read-only, the disk may be full, the file may be locked, etc.

fail_period_read_integer = 6
fail_period_convert_to_integer = 11
fail_period_next_string = 17
fail_period_write_back = 23


def read_integer_from_file_or_raise(file: str, iteration: int) -> str:
    # Simulate a file-read failure
    if iteration % fail_period_read_integer == 0:
        raise RuntimeError(f"File read failed at iteration {iteration}")
    # Simulate a bad string that cannot be converted
    if iteration % fail_period_convert_to_integer == 0:
        return "abc"
    # Otherwise, pretend the file contains "1"
    return "1"


def string_to_integer_or_raise(value: str, iteration: int) -> int:
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Conversion failed at iteration {iteration}")


def integer_to_next_string_or_raise(value: int, iteration: int) -> str:
    if iteration % fail_period_next_string == 0:
        raise RuntimeError(f"Increment failed at iteration {iteration}")
    return str(value + 1)


def write_to_file_or_raise(file: str, value: str, iteration: int) -> None:
    if iteration % fail_period_write_back == 0:
        raise RuntimeError(f"File write failed at iteration {iteration}")
    # Otherwise, success (do nothing).


def increment_file_or_raise(file: str, iteration: int) -> None:
    read_value = read_integer_from_file_or_raise(file, iteration)
    int_value = string_to_integer_or_raise(read_value, iteration)
    next_value = integer_to_next_string_or_raise(int_value, iteration)
    write_to_file_or_raise(file, next_value, iteration)


@pytest.mark.benchmark(group="06-errors-status-vs-raise")
def test_errors_raise(benchmark):
    def runner():
        file_path = "test.txt"
        iteration = 0
        for _ in range(1_000):
            iteration += 1
            try:
                increment_file_or_raise(file_path, iteration)
            except Exception:
                pass

    benchmark(runner)


# ? Now let’s define a simple status-based approach, akin to `std::expected`
# ? or a custom status enum in C++. It's not a common pattern in Python.

from enum import Enum, auto  # noqa: E402


class Status(Enum):
    SUCCESS = auto()
    READ_FAILED = auto()
    CONVERT_FAILED = auto()
    INCREMENT_FAILED = auto()
    WRITE_FAILED = auto()


class Expected:
    """
    A simple 'expected' type in Python.
    - If success, `error` is None and `value` holds the data.
    - If error, `error` is a Status, and `value` may be None or partial data.
    """

    __slots__ = ("value", "error")

    def __init__(self, value=None, error: Status = None):
        self.value = value
        self.error = error

    def is_ok(self) -> bool:
        return self.error is None


def read_integer_from_file_expected(file: str, iteration: int) -> Expected:
    if iteration % fail_period_read_integer == 0:
        return Expected(error=Status.READ_FAILED)
    if iteration % fail_period_convert_to_integer == 0:
        # Return "abc" with success => which triggers the "convert failed" later
        return Expected(value="abc", error=None)
    # Otherwise, pretend the file contains "1"
    return Expected(value="1", error=None)


def string_to_integer_expected(value: str, iteration: int) -> Expected:
    if not value.isnumeric():
        return Expected(error=Status.CONVERT_FAILED)
    return Expected(value=int(value), error=None)


def integer_to_next_string_expected(value: int, iteration: int) -> Expected:
    if iteration % fail_period_next_string == 0:
        return Expected(error=Status.INCREMENT_FAILED)
    return Expected(value=str(value + 1), error=None)


def write_to_file_expected(file: str, value: str, iteration: int) -> Status:
    if iteration % fail_period_write_back == 0:
        return Status.WRITE_FAILED
    return Status.SUCCESS


def increment_file_expected(file: str, iteration: int) -> Status:
    res_read = read_integer_from_file_expected(file, iteration)
    if not res_read.is_ok():
        return res_read.error
    res_int = string_to_integer_expected(res_read.value, iteration)
    if not res_int.is_ok():
        return res_int.error
    res_incr = integer_to_next_string_expected(res_int.value, iteration)
    if not res_incr.is_ok():
        return res_incr.error

    return write_to_file_expected(file, res_incr.value, iteration)


@pytest.mark.benchmark(group="06-errors-status-vs-raise")
def test_errors_expected(benchmark):
    def runner():
        file_path = "test.txt"
        iteration = 0
        for _ in range(1_000):
            iteration += 1
            increment_file_expected(file_path, iteration)

    benchmark(runner)


# ? As we know, classes and `__slots__` may add a noticeable overhead.
# ? So let's explore the less common Go-style approach of returning tuples
# ? and unpacking them on the fly.

StatusCode = int
STATUS_SUCCESS = 0
STATUS_READ_FAILED = 1
STATUS_CONVERT_FAILED = 2
STATUS_INCREMENT_FAILED = 3
STATUS_WRITE_FAILED = 4


def read_integer_from_file_status(file: str, iteration: int) -> Tuple[str, StatusCode]:
    if iteration % fail_period_read_integer == 0:
        return None, STATUS_READ_FAILED
    if iteration % fail_period_convert_to_integer == 0:
        # Return "abc" with success => which triggers the "convert failed" later
        return "abc", STATUS_SUCCESS
    # Otherwise, pretend the file contains "1"
    return "1", STATUS_SUCCESS


def string_to_integer_status(value: str, iteration: int) -> Tuple[int, StatusCode]:
    if not value.isnumeric():
        return None, STATUS_CONVERT_FAILED
    return int(value), STATUS_SUCCESS


def integer_to_next_string_status(value: int, iteration: int) -> Tuple[str, StatusCode]:
    if iteration % fail_period_next_string == 0:
        return None, STATUS_INCREMENT_FAILED
    return str(value + 1), STATUS_SUCCESS


def write_to_file_status(file: str, value: str, iteration: int) -> Status:
    if iteration % fail_period_write_back == 0:
        return STATUS_WRITE_FAILED
    return STATUS_SUCCESS


def increment_file_status(file: str, iteration: int) -> Status:
    read_value, read_status = read_integer_from_file_status(file, iteration)
    if read_status != STATUS_SUCCESS:
        return read_status
    int_value, int_status = string_to_integer_status(read_value, iteration)
    if int_status != STATUS_SUCCESS:
        return int_status
    next_value, next_status = integer_to_next_string_status(int_value, iteration)
    if next_status != STATUS_SUCCESS:
        return next_status
    write_status = write_to_file_status(file, next_value, iteration)
    return write_status


@pytest.mark.benchmark(group="06-errors-status-vs-raise")
def test_errors_status(benchmark):
    def runner():
        file_path = "test.txt"
        iteration = 0
        for _ in range(1_000):
            iteration += 1
            increment_file_status(file_path, iteration)

    benchmark(runner)


# ? Intel Xeon 4 · CPython 3.14t
# ?
# ?   status tuple     232 µs    1.00x
# ?   raise            348 µs    1.50x
# ?   Expected object  658 µs    2.83x  the explicit one is the slowest
# ?
# ? The ordering is the surprise: the wrapper written specifically to avoid
# ? exceptions is the slowest of the three, because it allocates an object per
# ? call while `raise` allocates only on the unhappy path. Plain tuples win by
# ? doing neither. The gap would widen with a `noexcept`-like annotation for
# ? functions that never raise: https://github.com/python/typing/issues/604

# endregion: Errors

# endregion: Error Handling
