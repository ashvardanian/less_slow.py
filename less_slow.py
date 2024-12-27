#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
less_slow.py
============

Microbenchmarks to build a performance-first mindset in Python.
"""


# region: Numerics

# region: Accuracy vs Efficiency of Standard Libraries

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

from typing import Callable, Tuple


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
from typing import Generator
from itertools import chain
from functools import reduce


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

# endregion: Polymorphic
from typing import List


class PipelineStage:
    """Base pipeline stage, mimicking a C++-style virtual interface."""

    def process(self, data: List[int]) -> None:
        raise NotImplementedError


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

import asyncio


async def for_range_async(start: int, end: int) -> Generator[int, None, None]:
    """Async generator that yields [start..end]."""
    for value in range(start, end + 1):
        yield value


async def filter_async(generator: asyncio.coroutine, predicate: Callable[[int], bool]):
    """Async generator that yields values from `generator` that do NOT satisfy `predicate`."""
    async for value in generator:
        if not predicate(value):
            yield value


async def prime_factors_async(generator: asyncio.coroutine):
    """Async generator that yields all prime factors of the values coming from `generator`."""
    async for val in generator:
        for factor in prime_factors_iterator(val):
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
    run_async = lambda: asyncio.run(pipeline_async())
    result = benchmark(run_async)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


# endregion: Pipelines and Abstractions
