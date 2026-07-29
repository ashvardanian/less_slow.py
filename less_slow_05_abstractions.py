#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Abstractions — the price of each way to express the same pipeline.

One computation written six ways: a fused loop with no stages, callbacks,
generators, a hand-written iterator class, eager stages behind a base class,
and async generators. In C++ and Rust these collapse to nearly the same
machine code and the zero-cost claim earns its name. In Python the cheapest
way to have stages at all costs 1.35x over the fused loop, and the spread
across the rest is 2.4x.

Two of the numbers are traps. Half the cost of the async row is building an
event loop, not running the pipeline — `asyncio.run` inside a timed region
measures loop construction, which is where the widely-repeated "async is 5x
slower" comes from. And accumulating with `reduce` over a `(sum, count)` pair
allocates a tuple per item, which indicts the accumulator rather than the
generators it is attached to.

The honest ranking, once both are controlled: the iterator protocol is the
expensive abstraction at 1.66x over generators, and the scheduler is the
expensive one at 2.24x.
"""

import pytest

# region: Pipelines and Abstractions

# ? One computation, six ways to write it. Take the integers 3 through 49,
# ? drop the powers of two and the powers of three, expand what remains into
# ? prime factors, and sum them. Eighty-four factors come out the other end
# ? however it is expressed.
# ?
# ? What differs is where the values live between stages:
# ?
# ?   fused        one loop, no stages          nothing crosses a boundary
# ?   callbacks    push: stage calls the next   one call per item
# ?   generators   pull: caller drives          one resume per item
# ?   iterator     pull, via __next__           one method call per item
# ?   eager        each stage builds a list     one full list per stage
# ?   async        pull, through an event loop  one scheduler turn per item
# ?
# ? In C++ and Rust these collapse to nearly the same machine code, and the
# ? zero-cost claim earns its name. In Python every arrow in that column is a
# ? real interpreter operation, so the shape of the pipeline is a performance
# ? decision rather than a stylistic one.

PIPE_START = 3
PIPE_END = 49


def is_power_of_two(value: int) -> bool:
    """Return True if `value` is a power of two."""
    return value > 0 and (value & (value - 1)) == 0


# ! The largest power of three that fits in 64 bits. Every power of three
# ! divides it and nothing else in `[3, 49]` does, so one modulo replaces a
# ! division loop — the trick only holds because the range is bounded.
MAX_POWER_OF_THREE = 12157665459056928801


def is_power_of_three(value: int) -> bool:
    """Return True if `value` divides the largest 64-bit power of three."""
    return value > 0 and (MAX_POWER_OF_THREE % value == 0)


# region: Callbacks

# ? The push model. A producer computes a value and calls a function with it;
# ? nothing is stored, nothing is suspended, and control never returns to a
# ? driver in between. The accumulator is a closure over two locals.
# ?
# ? This is the cheapest composition Python offers that still has stages,
# ? because a call is the only mechanism involved. What it costs is control:
# ? the producer decides when to stop, the consumer cannot pause, and there is
# ? no value to hand to something else halfway through.

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

# ? The pull model. `yield` suspends the producer with its locals intact and
# ? returns to the caller, who resumes it when the next value is wanted. The
# ? stages compose as objects — `filter`, `map`, `chain` — so the pipeline can
# ? be assembled, passed around, and consumed by someone else.
# ?
# ? A resume is more work than a call, since the frame has to be reactivated
# ? rather than created, and the cost lands per item. Two accumulators are
# ? measured over the identical front end to keep that separate from how the
# ? results are summed.

from typing import Generator  # noqa: E402
from functools import reduce  # noqa: E402
from itertools import chain  # noqa: E402


def prime_factors_generator(number: int) -> Generator[int, None, None]:
    """Yield prime factors of `number` one by one, lazily."""
    factor = 2
    while number > 1:
        if number % factor == 0:
            yield factor
            number //= factor
        else:
            factor += 1 if factor == 2 else 2


def _lazy_factors():
    """The shared lazy front end: range → two filters → flattened factors."""
    values = range(PIPE_START, PIPE_END + 1)
    values = filter(lambda value: not is_power_of_two(value), values)
    values = filter(lambda value: not is_power_of_three(value), values)
    return chain.from_iterable(map(prime_factors_generator, values))


def pipeline_generators() -> Tuple[int, int]:
    sum_factors = 0
    count = 0
    for factor in _lazy_factors():
        sum_factors += factor
        count += 1
    return sum_factors, count


def pipeline_generators_reduce() -> Tuple[int, int]:
    """The same pipeline accumulated with `reduce` over a `(sum, count)` pair."""
    # ! This allocates a tuple per factor where the loop above allocates none,
    # ! so the pair is a comparison of accumulators, not of generators.
    return reduce(
        lambda carried, factor: (carried[0] + factor, carried[1] + 1),
        _lazy_factors(),
        (0, 0),
    )


# endregion: Generators

# region: Iterators

# ? The same pull semantics, written by hand. A class with `__iter__` and
# ? `__next__` does what a generator does, with the suspended state kept in
# ? attributes instead of a frame:
# ?
# ?   generator     state in the frame       resume, then yield
# ?   iterator      state in self.number     attribute load, store, return
# ?
# ? Every item now costs a Python-level method call plus attribute traffic,
# ? where the generator costs a frame resume. This is the one shape that C++
# ? and Rust programmers reach for by habit and that Python punishes hardest.


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

# region: Eager Stages

# ? Stages behind an abstract base class, each rewriting the whole list before
# ? the next one runs. This is the shape a C++ programmer writes with virtual
# ? `process` methods, and the name usually attached to it is "dynamic
# ? dispatch" — which is not what it measures. Four `stage.process(data)` calls
# ? cannot show dispatch cost against a pipeline that processes eighty-four
# ? items.
# ?
# ? What it does measure is materialization: `data[:] = [...]` builds a
# ? complete list at every stage, so peak memory is the largest intermediate
# ? rather than one item, and nothing can be consumed until everything is
# ? ready. That is the real trade against the lazy shapes above.

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
        for value in data:
            # ! The generator, not the `PrimeFactors` class — otherwise this
            # ! pipeline would carry the iterator protocol's cost too, and the
            # ! two rows could not be told apart.
            result.extend(prime_factors_generator(value))
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


# endregion: Eager Stages

# region: Async Generators

# ? The pull model again, with a scheduler in the middle. `async for` suspends
# ? into an event loop rather than straight back to the caller, so each item
# ? costs a resume plus a trip through the loop's ready queue.
# ?
# ? This buys nothing here — the pipeline never waits on anything — and that is
# ? deliberate. It prices the machinery on a workload with no I/O to overlap,
# ? which is what an `async` pipeline degenerates to when its stages turn out
# ? to be CPU-bound.
# ?
# ? Two variants, because where the loop comes from dominates the answer:
# ?
# ?   asyncio.run(...)         builds a loop, runs, tears it down
# ?   loop.run_until_complete  reuses a loop that already exists

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

# region: No Abstraction At All

# ? Everything above has stages. This has none: the two predicates are inlined
# ? as expressions, the factorization is the body of the same loop, and no
# ? value is ever handed from one named thing to another.
# ?
# ? It is unpleasant to read, impossible to reuse, and it is the only honest
# ? baseline. Without it, "callbacks are fastest" means fastest among the
# ? abstractions, which quietly assumes the answer to the question actually
# ? being asked — what does structuring this cost at all?


def pipeline_fused() -> Tuple[int, int]:
    """Every stage inlined into one loop — the floor these are measured against."""
    sum_factors = 0
    count = 0
    for value in range(PIPE_START, PIPE_END + 1):
        if value > 0 and (value & (value - 1)) == 0:
            continue
        if value > 0 and MAX_POWER_OF_THREE % value == 0:
            continue
        number = value
        factor = 2
        while number > 1:
            if number % factor == 0:
                sum_factors += factor
                count += 1
                number //= factor
            else:
                factor += 1 if factor == 2 else 2
    return sum_factors, count


# endregion: No Abstraction At All

# ? Every implementation asserts against the same pair, which is the only
# ? guard that keeps this a comparison. Six pipelines that produce different
# ? answers are six different programs, and the fastest of them would be
# ? whichever one does the least work rather than whichever composes best.

PIPE_EXPECTED_SUM = 645  # sum of prime factors from the final data
PIPE_EXPECTED_COUNT = 84  # total prime factors from the final data


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_fused(benchmark):
    """No stages, no boundaries — the floor every other row is measured against."""
    result = benchmark(pipeline_fused)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_callbacks(benchmark):
    result = benchmark(pipeline_callbacks)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_generators(benchmark):
    result = benchmark(pipeline_generators)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_iterators(benchmark):
    result = benchmark(pipeline_iterators)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_dynamic_dispatch(benchmark):
    """Benchmark the dynamic-dispatch (trait-object) pipeline."""
    result = benchmark(pipeline_dynamic_dispatch)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.fixture(scope="module")
def run_on_shared_loop():
    """One event loop for the whole module, created once and reused."""
    loop = asyncio.new_event_loop()
    try:
        yield loop.run_until_complete
    finally:
        loop.close()


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_async(benchmark, run_on_shared_loop):
    """Async generators on an already-running loop — the pipeline alone."""
    result = benchmark(lambda: run_on_shared_loop(pipeline_async()))
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_async_run(benchmark):
    """The same pipeline through `asyncio.run`, which builds a loop per call."""
    result = benchmark(lambda: asyncio.run(pipeline_async()))
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


@pytest.mark.benchmark(group="05-abstractions-pipelines")
def test_pipeline_generators_reduce(benchmark):
    """The generator pipeline with a `reduce` accumulator instead of a loop."""
    result = benchmark(pipeline_generators_reduce)
    assert result == (PIPE_EXPECTED_SUM, PIPE_EXPECTED_COUNT)


# ? Intel Xeon 4 • CPython 3.14t • timeit, integers 3..49 factored and summed
# ?
# ?   fused loop            15.8 µs    1.00x  no stages at all
# ?   callbacks             21.4 µs    1.35x
# ?   eager stages          23.0 µs    1.45x
# ?   generators            23.1 µs    1.46x
# ?   generators + reduce   26.1 µs    1.65x
# ?   iterator class        38.4 µs    2.43x
# ?   async, shared loop    51.7 µs    3.27x
# ?   async, asyncio.run   102.3 µs    6.47x
# ?
# ? Abstraction costs 1.35x at the cheapest. That is the number worth carrying
# ? away, because it is the price of having stages at all — the fused loop is
# ? not a style anyone should write, and everything above it is paying for the
# ? ability to name and recombine the pieces.


# ? Two of those rows are traps rather than findings, and both are the kind a
# ? benchmark introduces by accident:
# ?
# ? Intel Xeon 4 • CPython 3.14t • what each comparison actually isolates
# ?
# ?   async, shared loop vs generators     2.24x   the scheduler
# ?   asyncio.run vs shared loop           1.98x   loop construction
# ?   iterator class vs generators         1.66x   __next__ over frame resume
# ?   reduce vs for loop                   1.13x   the tuple per item
# ?
# ? Half the `asyncio.run` figure — 50.6 µs of 102.3 — is building and tearing
# ? down an event loop rather than running anything. Time `asyncio.run` inside
# ? a benchmark and that setup is what you publish. The widely repeated "async
# ? generators are 5x slower" is mostly this.
# ?
# ? The `reduce` row indicts its accumulator, not its generators: carrying
# ? `(sum, count)` allocates a tuple per factor where a plain `for` allocates
# ? none. Both traps have the same shape — something outside the thing being
# ? compared varies along with it.


# ? Eager stages land at 1.45x, indistinguishable from generators, so
# ? materializing a list per stage costs nothing at this size — and would cost
# ? everything at a size that does not fit in memory. The lazy shapes are not
# ? faster than the eager one here; they are bounded, which is a property that
# ? does not show up in a table of one input size.
# ?
# ? A C++ or Rust version runs in roughly 200 ns, 79x the fused Python loop.
# ? Their equivalents of every row above would collapse onto that one number.
# ? That is what "zero-cost abstraction" means, and why this ranking has no
# ? counterpart there — the choice is free, so nobody measures it.

# endregion: Pipelines and Abstractions
