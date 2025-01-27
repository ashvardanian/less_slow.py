#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
less_slow.py
============

Micro-benchmarks to build a performance-first mindset in Python.

This project is a spiritual brother to `less_slow.cpp` for C++20,
and `less_slow.rs` for Rust. Unlike low-level systems languages,
Python is a high-level with significant runtime overheads, and
no obvious way to predict the performance of code.

Moreover, the performance of different language and library components
can vary significantly between consecutive Python versions. That's
true for both small numeric operations, and high-level abstractions,
like iterators, generators, and async code.
"""
import pytest
import numpy as np

# region: Numerics

# region: Accuracy vs Efficiency of Standard Libraries

# ? Numerical computing is a core subject in high-performance computing (HPC)
# ? research and graduate studies, yet its foundational concepts are more
# ? accessible than they seem. Let's start with one of the most basic
# ? operations — computing the __sine__ of a number.

import math


def f64_sine_math(x: float) -> float:
    return math.sin(x)


def f64_sine_math_cached(x: float) -> float:
    # Cache the math.sin function lookup
    local_sin = math.sin
    return local_sin(x)


def f64_sine_numpy(x: float) -> float:
    return np.sin(x)


# ? NumPy is the de-facto standard for numerical computing in Python, and
# ? it's known for its speed and simplicity. However, it's not always the
# ? fastest option for simple operations like sine, cosine, or exponentials.
# ?
# ? NumPy lacks hot-path optimizations if the input is a single scalar value,
# ? and it's often slower than the standard math library for such cases:
# ?
# ?  - math.sin:  620us
# ?  - np.sin:    4200us
# ?
# ? NumPy, of course, becomes much faster when the input is a large array,
# ? as opposed to a single scalar value.
# ?
# ? When absolute bit-accuracy is not required, it's often possible to
# ? approximate mathematical functions using simpler, faster operations.
# ? For example, the Maclaurin series for sine:
# ?
# ?   sin(x) ≈ x - (x^3)/3! + (x^5)/5! - (x^7)/7! + ...
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


def f64_sine_maclaurin_multiply(x: float) -> float:
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    return x - (x3 * 0.1666666667) + (x5 * 0.008333333333)


# ? Let's define a couple of helper functions to run benchmarks on these functions,
# ? and compare their performance on 10k random floats in [0, 2π]. We can also
# ? use Numba JIT to compile these functions to machine code, and compare the
# ? performance of the JIT-compiled functions with the standard Python functions.

numba_installed = False
try:
    import numba

    numba_installed = True
except ImportError:
    pass  # skip if numba is not installed


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


@pytest.mark.benchmark(group="sin")
def test_f64_sine_math(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_math)


@pytest.mark.benchmark(group="sin")
def test_f64_sine_math_cached(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_math_cached)


@pytest.mark.benchmark(group="sin")
def test_f64_sine_numpy(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_numpy)


@pytest.mark.benchmark(group="sin")
def test_f64_sine_maclaurin_math(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_math_maclaurin)


@pytest.mark.benchmark(group="sin")
def test_f64_sine_maclaurin_numpy(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_numpy_maclaurin)


@pytest.mark.benchmark(group="sin")
def test_f64_sine_maclaurin_powless(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_maclaurin_powless)


@pytest.mark.benchmark(group="sin")
def test_f64_sine_maclaurin_multiply(benchmark):
    _f64_sine_run_benchmark_on_each(benchmark, f64_sine_maclaurin_multiply)


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
@pytest.mark.benchmark(group="sin")
def test_f64_sine_jit(benchmark):
    sin_fn = numba.njit(f64_sine_math)
    sin_fn(0.0)  # trigger compilation
    _f64_sine_run_benchmark_on_each(benchmark, sin_fn)


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
@pytest.mark.benchmark(group="sin")
def test_f64_sine_maclaurin_jit(benchmark):
    sin_fn = numba.njit(f64_sine_math_maclaurin)
    sin_fn(0.0)  # trigger compilation
    _f64_sine_run_benchmark_on_each(benchmark, sin_fn)


@pytest.mark.skipif(not numba_installed, reason="Numba not installed")
@pytest.mark.benchmark(group="sin")
def test_f64_sine_maclaurin_powless_jit(benchmark):
    sin_fn = numba.njit(f64_sine_maclaurin_powless)
    sin_fn(0.0)  # trigger compilation
    _f64_sine_run_benchmark_on_each(benchmark, sin_fn)


@pytest.mark.benchmark(group="sin")
def test_f64_sines_numpy(benchmark):
    _f64_sine_run_benchmark_on_all(benchmark, f64_sine_numpy)


@pytest.mark.benchmark(group="sin")
def test_f64_sines_maclaurin_numpy(benchmark):
    _f64_sine_run_benchmark_on_all(benchmark, f64_sine_numpy_maclaurin)


@pytest.mark.benchmark(group="sin")
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

# endregion: Accuracy vs Efficiency of Standard Libraries

# endregion: Numerics

# region: Pipelines and Abstractions

# ? Designing efficient micro-kernels is hard, but composing them into
# ? high-level pipelines without losing performance is just as difficult.
# ?
# ? Consider a hypothetical numeric processing pipeline:
# ?
# ?   1. Generate all integers in a given range (e.g., [1, 49]).
# ?   2. Filter out integers that are perfect squares.
# ?   3. Expand each remaining number into its prime factors.
# ?   4. Sum all prime factors from the filtered numbers.
# ?
# ? We'll explore four implementations of this pipeline:
# ?
# ?  - __Callback-based__ pipeline using lambdas,
# ?  - __Generators__, `yield`-ing values at each stage,
# ?  - __Range-based__ pipeline using a custom `PrimeFactors` iterator,
# ?  - __Polymorphic__ pipeline with a shared base class,
# ?  - __Async Generators__ with `async for` loops.

PIPE_START = 3
PIPE_END = 49


def is_power_of_two(x: int) -> bool:
    """Return True if x is a power of two, False otherwise."""
    return x > 0 and (x & (x - 1)) == 0


def is_power_of_three(x: int) -> bool:
    """Return True if x is a power of three, False otherwise."""
    MAX_POWER_OF_THREE = 12157665459056928801
    return x > 0 and (MAX_POWER_OF_THREE % x == 0)


# region: Callbacks

from typing import Callable, Tuple  # noqa: E402


def prime_factors_callback(number: int, callback: Callable[[int], None]) -> None:
    """Factorize `number` into primes, invoking `callback(factor)` for each factor."""
    factor = 2
    while number > 1:
        if number % factor == 0:
            callback(factor)
            number //= factor
        else:
            factor += 1 if factor == 2 else 2


def pipeline_callbacks() -> Tuple[int, int]:
    sum_factors = 0
    count = 0

    def record_factor(factor: int) -> None:
        nonlocal sum_factors, count
        sum_factors += factor
        count += 1

    for value in range(PIPE_START, PIPE_END + 1):
        if not is_power_of_two(value) and not is_power_of_three(value):
            prime_factors_callback(value, record_factor)

    return sum_factors, count


# endregion: Callbacks

# region: Generators
from typing import Generator  # noqa: E402
from itertools import chain  # noqa: E402
from functools import reduce  # noqa: E402


def prime_factors_generator(number: int) -> Generator[int, None, None]:
    """Yield prime factors of `number` one by one, lazily."""
    factor = 2
    while number > 1:
        if number % factor == 0:
            yield factor
            number //= factor
        else:
            factor += 1 if factor == 2 else 2


def pipeline_generators() -> Tuple[int, int]:

    values = range(PIPE_START, PIPE_END + 1)
    values = filter(lambda x: not is_power_of_two(x), values)
    values = filter(lambda x: not is_power_of_three(x), values)

    values_factors = map(prime_factors_generator, values)
    all_factors = chain.from_iterable(values_factors)

    # Use `reduce` to do a single-pass accumulation of (sum, count)
    sum_factors, count = reduce(
        lambda acc, factor: (acc[0] + factor, acc[1] + 1),
        all_factors,
        (0, 0),
    )
    return sum_factors, count


# endregion: Generators

# region: Iterators


class PrimeFactors:
    """An iterator to lazily compute the prime factors of a single number."""

    def __init__(self, number: int) -> None:
        self.number = number
        self.factor = 2

    def __iter__(self) -> "PrimeFactors":
        return self

    def __next__(self) -> int:
        while self.number > 1:
            if self.number % self.factor == 0:
                self.number //= self.factor
                return self.factor
            self.factor += 1 if self.factor == 2 else 2

        raise StopIteration


def pipeline_iterators() -> Tuple[int, int]:
    sum_factors = 0
    count = 0

    for value in range(PIPE_START, PIPE_END + 1):
        if not is_power_of_two(value) and not is_power_of_three(value):
            for factor in PrimeFactors(value):
                sum_factors += factor
                count += 1

    return sum_factors, count


# endregion: Iterators

# region: Polymorphic
from typing import List  # noqa: E402
from abc import ABC, abstractmethod  # noqa: E402


class PipelineStage(ABC):
    """Base pipeline stage, mimicking a C++-style virtual interface."""

    @abstractmethod
    def process(self, data: List[int]) -> None: ...


class ForRangeStage(PipelineStage):
    """Stage that appends [start..end] to `data`."""

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

    def process(self, data: List[int]) -> None:
        data.clear()
        data.extend(range(self.start, self.end + 1))


class FilterStage(PipelineStage):
    """Stage that filters out values in-place using a predicate."""

    def __init__(self, predicate: Callable[[int], bool]) -> None:
        self.predicate = predicate

    def process(self, data: List[int]) -> None:
        data[:] = [x for x in data if not self.predicate(x)]


class PrimeFactorsStage(PipelineStage):
    """Stage that expands each value into prime factors, storing them back into data."""

    def process(self, data: List[int]) -> None:
        result = []
        for val in data:
            # Use the generator-based prime factors
            result.extend(PrimeFactors(val))
        data[:] = result


def pipeline_dynamic_dispatch() -> Tuple[int, int]:
    pipeline_stages = [
        ForRangeStage(PIPE_START, PIPE_END),
        FilterStage(is_power_of_two),
        FilterStage(is_power_of_three),
        PrimeFactorsStage(),
    ]

    data: List[int] = []
    for stage in pipeline_stages:
        stage.process(data)

    return sum(data), len(data)


# endregion: Polymorphic

# region: Async Generators

import asyncio  # noqa: E402
from typing import AsyncGenerator  # noqa: E402


async def for_range_async(start: int, end: int) -> AsyncGenerator[int, None]:
    """Async generator that yields [start..end]."""
    for value in range(start, end + 1):
        yield value


async def filter_async(generator, predicate: Callable[[int], bool]):
    """Async generator that forwards `generator` outputs NOT satisfying `predicate`."""
    async for value in generator:
        if not predicate(value):
            yield value


async def prime_factors_async(generator):
    """Async generator that yields prime factors for outputs of `generator`."""
    async for val in generator:
        for factor in prime_factors_generator(val):
            yield factor


async def pipeline_async() -> Tuple[int, int]:
    values = for_range_async(PIPE_START, PIPE_END)
    values = filter_async(values, is_power_of_two)
    values = filter_async(values, is_power_of_three)
    values = prime_factors_async(values)

    sum_factors = 0
    count = 0
    async for factor in values:
        sum_factors += factor
        count += 1

    return sum_factors, count


# endregion: Async Generators

PIPE_EXPECTED_SUM = 645  # sum of prime factors from the final data
PIPE_EXPECTED_COUNT = 84  # total prime factors from the final data


@pytest.mark.benchmark(group="pipelines")
def test_pipeline_callbacks(benchmark):
    result = benchmark(pipeline_callbacks)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="pipelines")
def test_pipeline_generators(benchmark):
    result = benchmark(pipeline_generators)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="pipelines")
def test_pipeline_iterators(benchmark):
    result = benchmark(pipeline_iterators)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="pipelines")
def test_pipeline_dynamic_dispatch(benchmark):
    """Benchmark the dynamic-dispatch (trait-object) pipeline."""
    result = benchmark(pipeline_dynamic_dispatch)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="pipelines")
def test_pipeline_async(benchmark):
    """Benchmark the async-generators pipeline."""
    run_async = lambda: asyncio.run(pipeline_async())  # noqa: E731
    result = benchmark(run_async)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


# ? The results, as expected, are much slower than in similar pipelines
# ? written in C++ or Rust. However, the iterators, don't seem like a
# ? good design choice in Python!
# ?
# ?  - Callbacks: 16.8ms
# ?  - Generators: 23.3ms
# ?  - Iterators: 31.8ms
# ?  - Polymorphic: 33.2ms
# ?  - Async: 97.0ms
# ?
# ? For comparison, a fast C++/Rust implementation would take 200ns,
# ? or __84x__ faster than the fastest Python implementation here.

# endregion: Pipelines and Abstractions

# region: Structures, Tuples, ADTs, AOS, SOA

# region: Composite Structs

# ? Python has many ways of defining composite objects. The most common
# ? are tuples, dictionaries, named-tuples, dataclasses, and classes.
# ? Let's compare them by assembling a simple composite of numeric values.
# ?
# ? We prefer `float` and `bool` fields as the most predictable Python types,
# ? as the `int` integers in Python involve a lot of additional logic for
# ? arbitrary-precision arithmetic, which can affect the latencies.

from dataclasses import dataclass  # noqa: E402
from collections import namedtuple  # noqa: E402


@pytest.mark.benchmark(group="composite-structs")
def test_structs_dict(benchmark):
    def kernel():
        point = {"x": 1.0, "y": 2.0, "flag": True}
        return point["x"] + point["y"]

    result = benchmark(kernel)
    assert result == 3.0

@pytest.mark.benchmark(group="composite-structs")
def test_structs_dict_fun(benchmark):
    def kernel():
        point = dict(x=1.0, y=2.0, flag=True)
        return point["x"] + point["y"]

    result = benchmark(kernel)
    assert result == 3.0

class PointClass:
    def __init__(self, x: float, y: float, flag: bool) -> None:
        self.x = x
        self.y = y
        self.flag = flag


@pytest.mark.benchmark(group="composite-structs")
def test_structs_class(benchmark):
    def kernel():
        point = PointClass(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


@dataclass
class PointDataclass:
    x: float
    y: float
    flag: bool


@pytest.mark.benchmark(group="composite-structs")
def test_structs_dataclass(benchmark):
    def kernel():
        point = PointDataclass(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


@dataclass
class PointSlotsDataclass:
    __slots__ = ("x", "y", "flag")
    x: float
    y: float
    flag: bool


@pytest.mark.benchmark(group="composite-structs")
def test_structs_slots_dataclass(benchmark):
    def kernel():
        point = PointSlotsDataclass(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


PointNamedtuple = namedtuple("PointNamedtuple", ["x", "y", "flag"])


@pytest.mark.benchmark(group="composite-structs")
def test_structs_namedtuple(benchmark):
    def kernel():
        point = PointNamedtuple(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


@pytest.mark.benchmark(group="composite-structs")
def test_structs_tuple_indexing(benchmark):
    def kernel():
        point = (1.0, 2.0, True)
        return point[0] + point[1]

    result = benchmark(kernel)
    assert result == 3.0


@pytest.mark.benchmark(group="composite-structs")
def test_structs_tuple_unpacking(benchmark):
    def kernel():
        x, y, _ = (1.0, 2.0, True)
        return x + y

    result = benchmark(kernel)
    assert result == 3.0


# ? Interestingly, the `namedtuple`, that is often believed to be a
# ? performance-oriented choice, is 50% slower than both `dataclass` and
# ? the custom class... which are in turn slower than a simple `dict`
# ? with the same string fields!
# ?
# ? - Tuple: 47ns (indexing) vs 43ns (unpacking)
# ? - Dict:  101ns
# ? - Slots Dataclass: 112ns
# ? - Dataclass: 122ns
# ? - Class: 125ns
# ? - Namedtuple: 183ns
# ?
# ? None of those structures validates the types of the fields, so
# ? many Python developers resort to external libraries for that.
# ?
# ? - pydantic: over 6 Million downloads per day
# ? - attrs: over 5 Million downloads per day

pydantic_installed = False
try:
    from pydantic import BaseModel  # noqa: E402

    pydantic_installed = True
except ImportError:
    BaseModel = dict


from attrs import define, field, validators  # noqa: E402


class PointPydantic(BaseModel):
    x: float
    y: float
    flag: bool


@pytest.mark.skipif(not pydantic_installed, reason="Pydantic not installed")
@pytest.mark.benchmark(group="composite-structs")
def test_structs_pydantic(benchmark):
    def kernel():
        point = PointPydantic(x=1.0, y=2.0, flag=True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


@define
class PointAttrs:
    x: float = field(validator=validators.instance_of(float))
    y: float = field(validator=validators.instance_of(float))
    flag: bool = field(validator=validators.instance_of(bool))


@pytest.mark.benchmark(group="composite-structs")
def test_structs_attrs(benchmark):
    def kernel():
        point = PointAttrs(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


# endregion: Composite Structs

# region: Heterogenous Collections

# ? Python is a dynamically typed language, and it allows mixing different
# ? types in a single collection. However, the performance of such collections
# ? can vary significantly, depending on the types and their distribution.


# endregion: Heterogenous Collections

# region: Tables and Arrays

import pandas as pd  # noqa: E402

try:
    import pyarrow as pa  # noqa: E402
except ImportError:
    pass

# endregion: Tables and Arrays

# endregion: Structures, Tuples, ADTs, AOS, SOA

# region: Exceptions, Backups, Logging

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


@pytest.mark.benchmark(group="errors")
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


@pytest.mark.benchmark(group="errors")
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


@pytest.mark.benchmark(group="errors")
def test_errors_status(benchmark):
    def runner():
        file_path = "test.txt"
        iteration = 0
        for _ in range(1_000):
            iteration += 1
            increment_file_status(file_path, iteration)

    benchmark(runner)


# ? The results are quite interesting! Raising exceptions beats the more
# ? explicit `Expected` approach by 2x, but loses to tuple-based status
# ? codes by 50%.
# ?
# ? - Raise: 329us
# ? - Expected: 660us
# ? - Status: 236us
# ?
# ? That difference could grow further we get a `noexcept`-like mechanism
# ? to annotate functions that never raise exceptions, and need no stack
# ? tracing logic: https://github.com/python/typing/issues/604
# ?
# ? Stick to `tuple`-s with unpacking for the best performance!

# endregion: Errors

# regions: Logs

# endregion: Logs

# endregion: Exceptions, Backups, Logging

# region: Dynamic Code

# region: Reflection, Inspection

# endregion: Reflection, Inspection

# region: Evaluating Strings

# endregion: Evaluating Strings

# endregion: Dynamic Code

# region: Networking and Databases

# ? When implementing web-applications, Python developers often rush to
# ? use overloaded high-level frameworks, like Django, Flask, or FastAPI,
# ? without ever considering a lower-level route.
# ?
# ? Let's implement a simple "echo" client and server using Python's
# ? built-in `socket` module, and compare its performance with a similar
# ? implementation in the `asyncio` module and FastAPI.

import socket  # for TCP and UDP servers # noqa: E402
import inspect  # to get the source code of a function # noqa: E402
import subprocess  # to start a server in a subprocess # noqa: E402
import sys  # to get the Python executable # noqa: E402
import time  # sleep for a bit until the socket binds # noqa: E402
from abc import ABC, abstractmethod  # to define abstract classes # noqa: E402
from typing import Literal  # noqa: E402

# ? The User Datagram Protocol (UDP) is OSI Layer 4 "Transport protocol", and
# ? should be able to operate on top of any OSI Layer 3 "Network protocol".
# ?
# ? In most cases, it operates on top of the Internet Protocol (IP), which can
# ? have Maximum Transmission Unit (MTU) ranging 20 for IPv4 and 40 for IPv6
# ? to 65535 bytes. In our case, however, the OSI Layer 2 "Data Link Layer" is
# ? likely to be Ethernet, which has a MTU of 1500 bytes, but most routers are
# ? configured to fragment packets larger than 1460 bytes. Hence, our choice!
RPC_MTU = 1460
RPC_PORT = 12345
RPC_PACKET_TIMEOUT_SEC = 0.05
RPC_BATCH_TIMEOUT_SEC = 0.5


def fetch_public_ip() -> str:
    """
    Returns the 'default' (outbound) IP address of the current machine.
    Note that this may be a private IP if behind NAT (it won't be your
    real public-facing IP if you are behind a router/firewall).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # The IP/port here doesn't need to be reachable (we never send data);
        # we just need the OS to pick a default interface for this "outbound" connection.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


class EchoServer(ABC):
    """Abstract base class for echo servers."""

    def __init__(self, host: str = "0.0.0.0", port: int = RPC_PORT):
        """
        :param host: The host to bind the server to. Set to '0.0.0.0' to listen on all
            interfaces. Set to 'localhost' or '127.0.0.1' to listen on the loopback
            interface.
        :param port: The port to bind the server to.
        """
        self.host = host
        self.port = port

    @abstractmethod
    def run(self):
        """Run the echo server (blocking call)."""
        pass


class TCPEchoServer(EchoServer):
    """Simple TCP Echo Server."""

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            while True:
                conn, _ = server.accept()
                with conn:
                    while True:
                        data = conn.recv(RPC_MTU)
                        if not data:
                            break
                        conn.sendall(data)


class UDPEchoServer(EchoServer):
    """Simple UDP Echo Server."""

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            while True:
                data, addr = server.recvfrom(RPC_MTU)
                if not data:
                    break
                server.sendto(data, addr)


class EchoClient(ABC):
    """Abstract base class for echo clients."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = RPC_PORT,
        timeout: float = RPC_PACKET_TIMEOUT_SEC,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout

    @abstractmethod
    def connect(self):
        """Establish or prepare the client socket (TCP connect, or just open a UDP socket)."""
        pass

    @abstractmethod
    def send_and_receive(self, data: bytes) -> bytes:
        """Send data and receive its echo."""
        pass

    def send_and_receive_batch(self, messages: List[bytes]) -> List[bytes]:
        """Send a batch of messages and receive their echoes."""
        return [self.send_and_receive(m) for m in messages]

    @abstractmethod
    def close(self):
        """Close the underlying socket."""
        pass


class TCPEchoClient(EchoClient):
    """TCP Echo Client implementation."""

    def __init__(
        self,
        host="localhost",
        port=RPC_PORT,
        timeout=RPC_PACKET_TIMEOUT_SEC,
    ):
        super().__init__(host, port, timeout)
        self._sock = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def send_and_receive(self, data: bytes) -> bytes:
        self._sock.sendall(data)
        return self._sock.recv(RPC_MTU)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None


class UDPEchoClient(EchoClient):
    """UDP Echo Client implementation."""

    def __init__(
        self,
        host="localhost",
        port=RPC_PORT,
        timeout=RPC_PACKET_TIMEOUT_SEC,
    ):
        super().__init__(host, port, timeout)
        self._sock = None

    def connect(self):
        # For UDP, "connect" isn't needed
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout)

    def send_and_receive(self, data: bytes) -> bytes:
        # For UDP, we must specify the address on sendto unless we've "connected" the socket.
        self._sock.sendto(data, (self.host, self.port))
        resp, _ = self._sock.recvfrom(RPC_MTU)
        return resp

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None


class ServerProcess:
    """
    Wraps an EchoServer in a subprocess. On __enter__, spawns the server
    and returns `self`. On __exit__, kills the subprocess.
    """

    def __init__(self, server: EchoServer):
        self.server = server
        self._proc = None

    def __enter__(self):
        source_code = inspect.getsource(self.server.__class__)
        # We'll also need the base class if the server references it:
        base_code = inspect.getsource(EchoServer)
        # Recreate an identical server instance in another process and call run()
        script = f"""
import socket
from abc import ABC, abstractmethod

RPC_MTU = {RPC_MTU}
RPC_PORT = {RPC_PORT}

{base_code}
{source_code}

if __name__ == "__main__":
    server = {self.server.__class__.__name__}(host={self.server.host!r}, port={self.server.port})
    server.run()
"""

        self._proc = subprocess.Popen([sys.executable, "-c", script])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._proc:
            self._proc.kill()
            self._proc.wait()


def profile_echo_latency(
    benchmark,
    server_class,
    client_class,
    packet_length: int = 1024,
    rounds: int = 100_000,
    batch_size: int = 1,
    use_batching: bool = False,
    route: Literal["loopback", "public"] = "loopback",
):
    """
    A generic echo latency profiler that uses class-based server/client.
    """

    packet = b"ping" * (packet_length // 4)
    address_to_listen = "127.0.0.1" if route == "loopback" else "0.0.0.0"
    address_to_talk = "127.0.0.1" if route == "loopback" else fetch_public_ip()
    lost_packets = 0

    # Initialize server and run in a subprocess context
    server = server_class(host=address_to_listen)
    context = ServerProcess(server).__enter__()
    time.sleep(0.5)  # Short wait to ensure server is listening

    # Create client
    client = client_class(host=address_to_talk)
    client.connect()

    # We may want to allow executing the requests within the batch asynchronously
    send_many: callable = client_class.send_and_receive_batch
    emulate_sending_many: callable = EchoClient.send_and_receive_batch
    supports_batching: bool = emulate_sending_many is not send_many
    packets = [packet] * batch_size
    if use_batching:
        assert supports_batching, "Client does not support batching!"

    def runner():
        nonlocal lost_packets
        try:
            responses = send_many(client, packets)
            if any(r != packet for r in responses):
                raise ValueError("Mismatched echo response!")
        except socket.timeout:
            lost_packets += batch_size

    benchmark.pedantic(runner, iterations=1, rounds=rounds)
    benchmark.extra_info["lost_packets"] = lost_packets

    client.close()
    context.__exit__(None, None, None)  # kill the server


@pytest.mark.benchmark(group="echo")
def test_rpc_tcp_loopback(benchmark):
    profile_echo_latency(benchmark, TCPEchoServer, TCPEchoClient, route="loopback")


@pytest.mark.benchmark(group="echo")
def test_rpc_udp_loopback(benchmark):
    profile_echo_latency(benchmark, UDPEchoServer, UDPEchoClient, route="loopback")


@pytest.mark.benchmark(group="echo")
def test_rpc_tcp_public(benchmark):
    profile_echo_latency(benchmark, TCPEchoServer, TCPEchoClient, route="public")


@pytest.mark.benchmark(group="echo")
def test_rpc_udp_public(benchmark):
    profile_echo_latency(benchmark, UDPEchoServer, UDPEchoClient, route="public")


# ? There's a clear difference between sending packets via `127.0.0.1` (loopback)
# ? versus the machine's "public" IP. Loopback is effectively short-circuited in
# ? software, yielding minimal overhead and tighter latency distributions.
# ? By contrast, using the "public" IP can trigger NAT hairpin or firewall checks,
# ? resulting in higher average and more variable latency, especially for UDP.
# ?
# ? - TCP Loopback: from 11 us to 319 us worst-case, average 18 us
# ? - TCP Public: from 13 us to 2'773 us worst-case, average 19 us
# ? - UDP Loopback: from 15 us to 542 us worst-case, average 20 us
# ? - UDP Public: from 27 us to 4'790 us worst-case, average 34 us
# ?
# ? Sounds interesting? I suggest reading
# ?
# ? - "High Performance Browser Networking" by Ilya Grigorik:
# ?   https://hpbn.co/
# ? - "Moving past TCP in the data center, part 2" by Jake Edge:
# ?   https://lwn.net/Articles/914030/


class AsyncioTCPEchoServer(EchoServer):
    """Asyncio-based TCP Echo Server."""

    def run(self):
        import asyncio

        async def handle_echo(reader, writer):
            while True:
                data = await reader.read(RPC_MTU)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        async def main_loop():
            server = await asyncio.start_server(handle_echo, self.host, self.port)
            async with server:
                # Serve forever (blocking)
                await server.serve_forever()

        asyncio.run(main_loop())


class AsyncioTCPEchoClient(ABC):
    """
    Since your framework expects .connect(), .send_and_receive(), .close()
    in a synchronous style, we internally run an event loop and call asyncio
    functions with run_until_complete().
    """

    def __init__(self, host="localhost", port=RPC_PORT, timeout=RPC_PACKET_TIMEOUT_SEC):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._loop = None
        self._reader = None
        self._writer = None

    def connect(self):
        import asyncio

        self._loop = asyncio.new_event_loop()

        async def _connect():
            reader, writer = await asyncio.open_connection(self.host, self.port)
            # Optionally, we can set socket timeouts or other config here.
            return reader, writer

        self._reader, self._writer = self._loop.run_until_complete(_connect())

    def send_and_receive(self, data: bytes) -> bytes:
        async def _send_and_receive(d):
            self._writer.write(d)
            await self._writer.drain()
            resp = await self._reader.read(RPC_MTU)
            return resp

        return self._loop.run_until_complete(_send_and_receive(data))

    def send_and_receive_batch(self, messages: List[bytes]) -> List[bytes]:
        async def _send_and_receive_batch(msgs: List[bytes]):
            results = []
            for m in msgs:
                self._writer.write(m)
                await self._writer.drain()
                resp = await self._reader.read(RPC_MTU)
                results.append(resp)
            return results

        return self._loop.run_until_complete(_send_and_receive_batch(messages))

    def close(self):
        async def _close():
            if self._writer:
                self._writer.close()
                await self._writer.wait_closed()

        if self._loop:
            self._loop.run_until_complete(_close())
            self._loop.close()
            self._loop = None


@pytest.mark.benchmark(group="echo")
def test_batch16_rpc_asyncio_ordered(benchmark):
    profile_echo_latency(
        benchmark,
        AsyncioTCPEchoServer,
        AsyncioTCPEchoClient,
        route="loopback",
        batch_size=16,
        use_batching=True,
        rounds=1_000,
    )


@pytest.mark.benchmark(group="echo")
def test_batch16_rpc_asyncio_unordered(benchmark):
    profile_echo_latency(
        benchmark,
        AsyncioTCPEchoServer,
        AsyncioTCPEchoClient,
        route="loopback",
        batch_size=16,
        use_batching=True,
        rounds=1_000,
    )


# ? The results are unsettling. The promise of `asyncio` is to provide a
# ? high-performance, non-blocking I/O framework. However, the overhead
# ? of the event loop, the context switches, and the additional buffering
# ? can make it slower than the synchronous TCP client per call.
# ?
# ? For 16 calls in a batch, using the 'loopback' interface, the latency is:
# ? - Asyncio Ordered: from 579 us to 2'909 us worst-case, average 627 us
# ? - Asyncio Unordered: from 582 us to 2'598 us worst-case, average 631 us
# ?
# ? First, we don't see a significant improvement in latency when allowing
# ? out-of-order processing. Second, when normalizing throughput, the
# ? original blocking TCP client ends up being faster:
# ?
# ? - Asyncio Ordered: average 39 us
# ? - Asyncio Unordered: average 39 us
# ? - TCP Loopback: average 18 us
# ?
# ? This, however, may not be as bad as higher-level frameworks like FastAPI,
# ? and one of the most common underlying ASGI servers, Uvicorn.


class FastAPIEchoServer(EchoServer):
    """
    Minimal FastAPI-based HTTP echo server. It exposes a POST /echo endpoint
    that simply returns the raw request body as-is (using a binary media type).
    """

    def run(self):
        import uvicorn
        from fastapi import FastAPI, Request
        from fastapi.responses import Response

        app = FastAPI()

        @app.post("/echo")
        async def echo_endpoint(req: Request):
            data = await req.body()
            return Response(content=data, media_type="application/octet-stream")

        uvicorn.run(app, host=self.host, port=self.port, log_level="error")


class UvicornEchoServer(EchoServer):
    """
    Minimal raw ASGI echo server on /echo. No FastAPI or Starlette, just
    uvicorn + a single scope check for POST /echo. Returns the request
    body verbatim with content-type=application/octet-stream.
    """

    def run(self):
        import uvicorn

        async def app(scope, receive, send):
            if scope["type"] == "http":
                # Check path; if not /echo, return 404
                if scope.get("path", "") != "/echo":
                    await send(
                        {"type": "http.response.start", "status": 404, "headers": []}
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"Not Found",
                            "more_body": False,
                        }
                    )
                    return

                body = b""
                more_body = True
                while more_body:
                    event = await receive()
                    if event["type"] == "http.request":
                        body += event.get("body", b"")
                        more_body = event.get("more_body", False)

                # Echo the body
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"application/octet-stream"),
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                        "more_body": False,
                    }
                )

        uvicorn.run(app, host=self.host, port=self.port, log_level="error")


class RequestsClient(EchoClient):
    """
    A simple requests-based client, calling POST /echo with the raw data in the
    request body, and returning the response body as bytes.
    """

    def __init__(self, host="localhost", port=RPC_PORT, timeout=RPC_PACKET_TIMEOUT_SEC):
        super().__init__(host, port, timeout)
        self._session = None

    def connect(self):
        import requests

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/octet-stream"})

    def send_and_receive(self, data: bytes) -> bytes:
        url = f"http://{self.host}:{self.port}/echo"
        resp = self._session.post(url, data=data, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    def close(self):
        if self._session:
            self._session.close()
            self._session = None


class HTTPXAsyncEchoClient(EchoClient):
    """
    Uses the httpx library in async mode to talk to the /echo endpoint.
    Batching is done concurrently with asyncio.gather.
    """

    def __init__(self, host="localhost", port=RPC_PORT, timeout=RPC_PACKET_TIMEOUT_SEC):
        super().__init__(host, port, timeout)
        self._loop = None
        self._client = None

    def connect(self):
        import httpx
        import asyncio

        # We'll create a dedicated event loop for this client and
        # instantiate the AsyncClient inside it.
        self._loop = asyncio.new_event_loop()

        async def _setup():
            # Create an AsyncClient with the given timeout and
            # set headers for sending binary data.
            client = httpx.AsyncClient(timeout=self.timeout)
            client.headers.update({"Content-Type": "application/octet-stream"})
            return client

        self._client = self._loop.run_until_complete(_setup())

    def send_and_receive(self, data: bytes) -> bytes:
        """
        Sends a single request and awaits the response using AsyncClient.
        We wrap it in run_until_complete() for synchronous code compatibility.
        """

        async def _send_and_receive(d):
            url = f"http://{self.host}:{self.port}/echo"
            resp = await self._client.post(url, content=d)
            resp.raise_for_status()
            return resp.content

        return self._loop.run_until_complete(_send_and_receive(data))

    def send_and_receive_batch(self, messages: List[bytes]) -> List[bytes]:
        """
        Demonstrates concurrent batch logic using asyncio.gather.
        All requests are fired off in parallel, then we await all responses.
        """
        import asyncio

        async def _send_and_receive_batch(msgs: List[bytes]) -> List[bytes]:
            url = f"http://{self.host}:{self.port}/echo"

            # Build a coroutine for each message
            async def post(msg: bytes):
                resp = await self._client.post(url, content=msg)
                resp.raise_for_status()
                return resp.content

            # Fire them off concurrently
            tasks = [post(m) for m in msgs]
            results = await asyncio.gather(*tasks)
            return list(results)

        return self._loop.run_until_complete(_send_and_receive_batch(messages))

    def close(self):
        """
        Closes the AsyncClient and event loop.
        """
        import asyncio

        async def _close():
            if self._client:
                await self._client.aclose()

        if self._loop:
            self._loop.run_until_complete(_close())
            self._loop.close()
            self._loop = None


@pytest.mark.benchmark(group="echo")
def test_batch16_rpc_fastapi_requests(benchmark):
    profile_echo_latency(
        benchmark,
        FastAPIEchoServer,
        RequestsClient,
        route="loopback",
        batch_size=16,
        use_batching=False,  # ! Requests are typically synchronous
        rounds=1_000,
    )


@pytest.mark.benchmark(group="echo")
def test_batch16_rpc_fastapi_httpx(benchmark):
    profile_echo_latency(
        benchmark,
        FastAPIEchoServer,
        HTTPXAsyncEchoClient,
        route="loopback",
        batch_size=16,
        use_batching=True,
        rounds=1_000,
    )


@pytest.mark.benchmark(group="echo")
def test_batch16_rpc_uvicorn_requests(benchmark):
    profile_echo_latency(
        benchmark,
        FastAPIEchoServer,
        RequestsClient,
        route="loopback",
        batch_size=16,
        use_batching=False,  # ! Requests are typically synchronous
        rounds=1_000,
    )


@pytest.mark.benchmark(group="echo")
def test_batch16_rpc_uvicorn_httpx(benchmark):
    profile_echo_latency(
        benchmark,
        FastAPIEchoServer,
        HTTPXAsyncEchoClient,
        route="loopback",
        batch_size=16,
        use_batching=True,
        rounds=1_000,
    )


# ? The benchmark results are striking. For batch sizes of 16 messages:
# ?
# ? - Raw TCP with asyncio: 0.95 milliseconds per batch (59 us per message)
# ? - Requests+FastAPI/Uvicorn: 7.7 milliseconds per batch (0.5 ms per message)
# ? - Async HTTPX+FastAPI/Uvicorn: 12.5 milliseconds per batch (0.8 ms per message)
# ?
# ? This demonstrates why low-latency systems often avoid HTTP and high-level
# ? frameworks in favor of raw TCP/UDP, especially for internal services. The
# ? arguable convenience of FastAPI comes at a significant performance cost -
# ? about 10x slower than already slow IO stack of Python.

# endregion: Networking and Databases
