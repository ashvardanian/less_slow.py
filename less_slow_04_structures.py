#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data structures — what a record costs to build, to hold.

Four questions about the same data. Constructing it: a bare tuple is the
floor and `namedtuple` — reached for precisely because it sounds like the
performance choice — is the slowest of the built-ins. Holding it: `sys.getsizeof`
reports 48 bytes for a dataclass instance that actually costs 185, because
since 3.11 the attributes live in an inline preheader it never walks, and
`__slots__` saves 1.28x rather than the 2-3x folklore.

Then text, where the sharp result is that `re` caches compiled patterns in a
512-entry dict: exceed it with patterns built from user input and every call
silently recompiles, at 25x. And tables, where a major version of Pandas
inverted a ranking published here two years earlier — a reminder that measured
numbers have a shelf life.
"""
import gc
import sys

import numpy as np
import pandas as pd
import pytest

from less_slow_00_shared import pandas_installed, pyarrow_installed

if pyarrow_installed:
    import pyarrow as pa
    import pyarrow.compute as pc

# region: Data Structures

# region: Composite Structs

# ? A record with three fields can be a tuple, a dict, a class, a dataclass, a
# ? namedtuple, or a validated model, and the choice is usually made on how the
# ? code will read. It costs a factor of thirty-six.
# ?
# ? Fields are `float` and `bool` deliberately. Python's `int` is
# ? arbitrary-precision, so integer fields would mix allocation behaviour into
# ? a measurement about record shape.

from dataclasses import dataclass  # noqa: E402
from collections import namedtuple  # noqa: E402


@pytest.mark.benchmark(group="04-structures-composites")
def test_structs_dict(benchmark):
    def kernel():
        point = {"x": 1.0, "y": 2.0, "flag": True}
        return point["x"] + point["y"]

    result = benchmark(kernel)
    assert result == 3.0


@pytest.mark.benchmark(group="04-structures-composites")
def test_structs_dict_fun(benchmark):
    def kernel():
        point = dict(x=1.0, y=2.0, flag=True)
        return point["x"] + point["y"]

    result = benchmark(kernel)
    assert result == 3.0


# ? `{}` is syntax and `dict()` is a name, and nothing else separates them.
# ? Syntax compiles to `BUILD_MAP`. A name has to be looked up first, because
# ? any module is free to rebind `dict` — so the interpreter is not allowed to
# ? assume it means the builtin:
# ?
# ?   {"x": 1.0}     BUILD_MAP
# ?   dict(x=1.0)    LOAD_GLOBAL dict → CALL_KW
# ?
# ? Intel Xeon 4 • CPython 3.14t • timeit, one dict built and read per call
# ?
# ?   {"x": 1.0, ...}      85.8 ns    1.00x  one BUILD_MAP
# ?   dict(x=1.0, ...)    130.6 ns    1.52x  global lookup, then call
# ?
# ? The same asymmetry holds for `[]` against `list()` and `()` against
# ? `tuple()`. Dynamism has a standing price, and it is charged even to code
# ? that never uses it.


class PointClass:
    def __init__(self, x: float, y: float, flag: bool) -> None:
        self.x = x
        self.y = y
        self.flag = flag


@pytest.mark.benchmark(group="04-structures-composites")
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


@pytest.mark.benchmark(group="04-structures-composites")
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


@pytest.mark.benchmark(group="04-structures-composites")
def test_structs_slots_dataclass(benchmark):
    def kernel():
        point = PointSlotsDataclass(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


# ? `namedtuple` is where intuition fails. A tuple underneath ought to mean
# ? tuple speed, but the name is bound to generated Python:
# ?
# ?   tuple         BUILD_TUPLE, then index          24.4 ns
# ?   namedtuple    __new__ → _tuple_new, then       199.5 ns
# ?                 a property descriptor per field
# ?
# ? So it lands at 8.17x the thing it wraps, and slower than the `class` it was
# ? supposed to be a lightweight alternative to. Field names are not free when
# ? they are implemented in the language rather than in the layout.


PointNamedtuple = namedtuple("PointNamedtuple", ["x", "y", "flag"])


@pytest.mark.benchmark(group="04-structures-composites")
def test_structs_namedtuple(benchmark):
    def kernel():
        point = PointNamedtuple(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


@pytest.mark.benchmark(group="04-structures-composites")
def test_structs_tuple_indexing(benchmark):
    def kernel():
        point = (1.0, 2.0, True)
        return point[0] + point[1]

    result = benchmark(kernel)
    assert result == 3.0


# ? Intel Xeon 4 • CPython 3.14t • timeit, one record built and read per call
# ?
# ?   tuple                 24.4 ns    1.00x  no names, no class
# ?   dict literal          85.8 ns    3.51x
# ?   __slots__ dataclass  115.6 ns    4.73x
# ?   class                124.9 ns    5.11x
# ?   dict() call          130.6 ns    5.35x
# ?   dataclass            135.0 ns    5.53x
# ?   namedtuple           199.5 ns    8.17x  the "fast" one, in folklore
# ?   attrs                511.9 ns   20.97x  validation
# ?   pydantic             873.3 ns   35.77x  validation and coercion
# ?
# ? None of the first seven rows checks that `x` is a float. The last two do,
# ? which is the entire gap — `pydantic` will also coerce `"1.0"` into `1.0`,
# ? and that flexibility is what you are buying at 36x. Whether it is worth it
# ? depends on whether the data is already trusted, which for a record built
# ? in-process from your own code it usually is.

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
@pytest.mark.benchmark(group="04-structures-composites")
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


@pytest.mark.benchmark(group="04-structures-composites")
def test_structs_attrs(benchmark):
    def kernel():
        point = PointAttrs(1.0, 2.0, True)
        return point.x + point.y

    result = benchmark(kernel)
    assert result == 3.0


# endregion: Composite Structs

# region: Object Footprint

# ? Asking an object how big it is does not work. `sys.getsizeof` is shallow —
# ? it never follows a reference — and since CPython 3.11 it does not even
# ? count an instance's own attributes, which live in an inline preheader
# ? rather than a separate `__dict__`. It under-reports precisely the objects
# ? worth measuring.
# ?
# ? Recursive sizers are worse: reading `obj.__dict__` *materializes* those
# ? inline values into a real dictionary, adding 64 B per instance. The tool
# ? changes the thing it measures.
# ?
# ? What works is allocating ten thousand records and watching the allocator.

import tracemalloc  # noqa: E402

POINT_DTYPE = np.dtype([("x", np.float64), ("y", np.float64), ("flag", np.bool_)])


def _bytes_per_record(build, count: int = 10_000) -> float:
    """Allocate `count` records through `build`, reporting the bytes each costs."""
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    records = build(count)
    after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(records) == count
    return (after - before) / count


def _build_numpy(count: int):
    points = np.zeros(count, dtype=POINT_DTYPE)
    points["x"] = np.arange(count, dtype=np.float64)
    points["y"] = np.arange(count, dtype=np.float64)
    return points


# ! Every record gets distinct field values on purpose. A factory returning
# ! `PointDataclass(1.0, 2.0, True)` would share three float objects across all
# ! ten thousand instances and understate every row by 80 bytes.
LAYOUTS = {
    "dict": lambda count: [
        {"x": float(index), "y": float(index), "flag": True} for index in range(count)
    ],
    "class": lambda count: [
        PointClass(float(index), float(index), True) for index in range(count)
    ],
    "dataclass": lambda count: [
        PointDataclass(float(index), float(index), True) for index in range(count)
    ],
    "slots": lambda count: [
        PointSlotsDataclass(float(index), float(index), True) for index in range(count)
    ],
    "namedtuple": lambda count: [
        PointNamedtuple(float(index), float(index), True) for index in range(count)
    ],
    "tuple": lambda count: [
        (float(index), float(index), True) for index in range(count)
    ],
    "numpy": _build_numpy,
}


# ! These are measurements, not benchmarks, so they take no `benchmark` fixture.
# ! Timing them would report how long `tracemalloc` takes to watch ten thousand
# ! allocations, which is not a number anybody wants.
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_footprint(layout, record_property):
    """Bytes per record, obtained by allocating rather than by asking."""
    measured = _bytes_per_record(LAYOUTS[layout])
    record_property("bytes_per_record", round(measured, 1))
    assert measured > 0


def test_footprint_getsizeof_under_reports():
    """`sys.getsizeof` is shallow, and since 3.11 blind to the preheader."""
    shallow = sys.getsizeof(PointDataclass(1.0, 2.0, True))
    measured = _bytes_per_record(LAYOUTS["dataclass"])
    record = (shallow, measured)
    # ! This assert is the lesson, and it should start failing the day CPython
    # ! learns to count its own inline values.
    assert measured > 2 * shallow, record


# ? Intel Xeon 4 • CPython 3.14t • 10K records, distinct field values
# ?
# ?   NumPy structured      17.0 B    1.00x  one buffer, no headers
# ?   __slots__ dataclass  144.6 B    8.49x  no instance dictionary
# ?   tuple                160.5 B    9.43x
# ?   namedtuple           168.6 B    9.90x  a tuple plus its class
# ?   class                184.9 B   10.85x
# ?   dataclass            184.9 B   10.85x  getsizeof claims 48
# ?   dict                 272.5 B   16.01x  the fattest
# ?
# ? Two pieces of folklore die here. `sys.getsizeof` reports 48 bytes for a
# ? dataclass instance that costs 185 — a 3.9x under-report. And `__slots__`
# ? buys 1.28x, not the 2–3x usually quoted, because key-sharing dictionaries
# ? and inline values had already captured most of that win before you asked.
# ?
# ? Free-threading is not free either: `sys.getsizeof(1.0)` is 40 bytes on
# ? 3.14t against 24 on the GIL build of the same version, the extra being
# ? header fields lock-free refcounting needs. Only small immutables grew.


# ? Identical bytes can still be laid out two ways. Array-of-Structs keeps each
# ? record's fields together, so walking one field steps over the others.
# ? Struct-of-Arrays gives each field its own buffer.
# ?
# ?   AoS   x y f │ x y f │ x y f │ x y f      one field → stride 17
# ?   SoA   x x x x │ y y y y │ f f f f        one field → stride 8
# ?
# ? A vectorized loop wants the second. The first is what you get by default
# ? from a structured array, a list of objects, or a row-oriented database
# ? driver — which is why columnar formats exist.


@pytest.mark.benchmark(group="04-structures-footprint-access")
def test_access_array_of_structs(benchmark):
    """Sum one field of a structured array — stride is the whole struct."""
    points = np.zeros(1_000_000, dtype=POINT_DTYPE)
    points["x"] = np.arange(1_000_000, dtype=np.float64)

    def kernel():
        return points["x"].sum()

    result = benchmark(kernel)
    benchmark.extra_info["stride"] = points["x"].strides[0]
    assert result > 0


@pytest.mark.benchmark(group="04-structures-footprint-access")
def test_access_struct_of_arrays(benchmark):
    """The same sum over a dedicated column — stride is the scalar."""
    xs = np.arange(1_000_000, dtype=np.float64)

    def kernel():
        return xs.sum()

    result = benchmark(kernel)
    benchmark.extra_info["stride"] = xs.strides[0]
    assert result > 0


# ? Intel Xeon 4 • CPython 3.14t • summing one field of 1M elements
# ?
# ?   struct-of-arrays      406 µs    1.00x  stride 8, contiguous
# ?   array-of-structs    1'096 µs    2.70x  stride 17, neither
# ?
# ? Both layouts hold the same bytes. The field view of a structured array
# ? inherits the struct's 17-byte stride, so it is neither contiguous nor
# ? aligned, and cannot be fed to a vectorized loop.

# endregion: Object Footprint

# region: Numeric Coercion

# ? Data arrives in whatever type its source happened to use: a form field is a
# ? string, a currency column is a `Decimal`, a JSON body is an `int`. Summing
# ? that pile means deciding, per element, what it is — and there are three
# ? ways to decide, which differ by four orders of magnitude.
# ?
# ? Coerce everything and never ask. Ask first with `isinstance`. Or do not ask,
# ? attempt the addition, and catch what fails. The last is the one Python
# ? culture recommends, under the name EAFP, and it is the slowest by far
# ? whenever failures are common rather than rare.


from decimal import Decimal  # noqa: E402
from fractions import Fraction  # noqa: E402


def _represent_with_different_types(value_int: int = 3):
    """Represent the same numeric value via different types (no invalid entries)."""
    return [
        value_int,  # int
        float(value_int),  # float
        np.int64(value_int),  # numpy integer
        np.float64(value_int),  # numpy float
        Decimal(value_int),  # decimal
        Fraction(value_int, 1),  # fraction
        f"{value_int}",  # numeric string
        f"{value_int}.0",  # numeric string with decimal point
    ]


@pytest.mark.benchmark(group="04-structures-coercion")
def test_heterogeneous_sum(benchmark):
    """Coerce all representations with float(); all succeed (no exceptions)."""
    seed = _represent_with_different_types()
    values = seed * 10_000

    def kernel():
        return sum(float(value) for value in values)

    result = benchmark(kernel)
    assert abs(result - 3.0 * len(values)) < 1e-9


@pytest.mark.benchmark(group="04-structures-coercion")
def test_type_matching_sum(benchmark):
    """Use isinstance() to dispatch: add directly or coerce with float()."""
    seed = _represent_with_different_types()
    values = seed * 10_000

    def kernel():
        sum_value = 0.0
        for value in values:
            if isinstance(value, (int, float, np.integer, np.floating)):
                sum_value += value
            else:
                sum_value += float(value)
        return sum_value

    result = benchmark(kernel)
    assert abs(result - 3.0 * len(values)) < 1e-9


@pytest.mark.benchmark(group="04-structures-coercion")
def test_try_except_sum(benchmark):
    """Use try/except to handle type coercion: EAFP style."""
    seed = _represent_with_different_types()
    values = seed * 10_000

    def kernel():
        sum_value = 0.0
        for value in values:
            try:
                sum_value += value
            except TypeError:
                sum_value += float(value)
        return sum_value

    result = benchmark(kernel)
    assert abs(result - 3.0 * len(values)) < 1e-9


# ? Intel Xeon 4 • CPython 3.14t • 80K values, eight representations
# ?
# ?   float() on everything   3'694 µs    1.00x  never asks
# ?   isinstance dispatch    16'068 µs    4.35x  asks first
# ?   try/except, EAFP       50'147 µs   13.58x  asks by failing
# ?
# ? EAFP costs 13.58x because three of the eight representations — `Decimal`
# ? and the two strings — raise `TypeError` when added to a float. That is
# ? 37.5% of the elements, and an exception is not a cheap branch. `Fraction`
# ? is not among them: its `__radd__` accepts a float and returns one. Whether
# ? EAFP is free or ruinous turns on details like that, which nobody can
# ? predict by reading the list of type names.
# ?
# ? Asking first is worse than not asking. `isinstance` costs 4.35x more than
# ? unconditional `float()` despite skipping the call for half the elements —
# ? the tuple-of-types check is dearer than the call it avoids. An optimization
# ? that inspects a value to decide whether to convert it must beat the
# ? conversion, and the conversion here is one C call.


# ? Three strategies for one problem, and all three accept the premise that the
# ? data is mixed. The two benchmarks below reject it — same element count, one
# ? type — to price what the premise costs:
# ?
# ?   sum(mixed)      per element: decide, then convert, then add
# ?   sum(floats)     per element: add
# ?   np.sum(array)   per element: add, in a C loop over a flat buffer
# ?
# ? A baseline that shares the workload but not the constraint is how you find
# ? out whether you are optimizing a solution or removing a problem.


@pytest.mark.benchmark(group="04-structures-coercion")
def test_homogeneous_sum(benchmark):
    """Baseline: homogeneous float list with the same value."""
    values = [3.0] * (8 * 10_000)  # same total length as the hetero lists

    def kernel():
        return sum(values)

    result = benchmark(kernel)
    assert abs(result - 3.0 * len(values)) < 1e-12


@pytest.mark.benchmark(group="04-structures-coercion")
def test_homogeneous_container_sum(benchmark):
    """Baseline: homogeneous float list with the same value."""
    values = [3.0] * (8 * 10_000)  # same total length as the hetero lists
    values = np.array(values, dtype=np.float64)

    def kernel():
        return np.sum(values)

    result = benchmark(kernel)
    assert abs(result - 3.0 * len(values)) < 1e-12


# ? Intel Xeon 4 • CPython 3.14t • 80K values, same count both ways
# ?
# ?   np.sum, one dtype        17.7 µs      1.00x
# ?   sum(), one dtype          273 µs     15.44x
# ?   float() on everything   3'694 µs    208.69x  ← best mixed-data strategy
# ?   isinstance dispatch    16'068 µs    907.67x
# ?   try/except, EAFP       50'147 µs     2'833x
# ?
# ? The 208x between the best mixed row and the homogeneous list is the real
# ? result, and it dwarfs the 13.58x separating the three strategies. Choosing
# ? well among them is worth an order of magnitude; not needing to choose is
# ? worth two.
# ?
# ? That is an argument about where the work happens, not how fast it is.
# ? Coercing at ingestion converts each value once; coercing at use converts it
# ? on every pass over the data.


# endregion: Numeric Coercion

# region: Text Representation

# ? A Python string has no fixed width per character. PEP 393 picks the
# ? narrowest representation that fits the widest character present, and the
# ? choice is made once for the whole string:
# ?
# ?   "hello"          ASCII, 1 byte/char    ← every character below U+0080
# ?   "héllo"          UCS-2, 2 bytes/char   ← one character forced the upgrade
# ?   "héllo 🔥"       UCS-4, 4 bytes/char   ← one character forced it again
# ?
# ? Nothing is per-character about that decision. One emoji at the end of ten
# ? thousand ASCII characters quadruples the whole buffer, because indexing has
# ? to stay O(1) and that requires a uniform stride.


def _make_ascii_string(length: int = 10_000) -> str:
    """Create a pure ASCII string."""
    return "a" * length


def _make_emoji_string(length: int = 10_000) -> str:
    """Create a string with emojis, forcing UCS-4 (4 bytes/char) representation."""
    return ("a" * 9 + "\U0001f525") * (length // 10)  # fire emoji


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_encode_ascii(benchmark):
    """Encode ASCII string to UTF-8 bytes."""
    text = _make_ascii_string(10_000)

    def kernel():
        return text.encode("utf-8")

    result = benchmark(kernel)
    assert len(result) == 10_000  # 1 byte per char


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_encode_emoji(benchmark):
    """Encode emoji string to UTF-8 bytes — emojis become 4 bytes each."""
    text = _make_emoji_string(10_000)

    def kernel():
        return text.encode("utf-8")

    result = benchmark(kernel)
    assert len(result) > 10_000  # emojis expand to 4 bytes


def test_string_width_quadruples():
    """One astral character sets the stride for every character in the string."""
    ascii_text = _make_ascii_string(10_000)
    emoji_text = _make_emoji_string(10_000)
    assert len(ascii_text) == len(emoji_text)
    narrow, wide = sys.getsizeof(ascii_text), sys.getsizeof(emoji_text)
    # ! Same character count, four times the buffer. The emoji is 10% of the
    # ! characters and 100% of the reason.
    assert 3.9 < wide / narrow < 4.1, (narrow, wide)


# ? Intel Xeon 4 • CPython 3.14t • 10K characters, one emoji every tenth
# ?
# ?   sys.getsizeof, ascii    10'057 B    1.00x    1.01 bytes per character
# ?   sys.getsizeof, emoji    40'076 B    3.98x    4.01 bytes per character
# ?
# ?   encode('utf-8'), ascii      211 ns    1.00x  memcpy, the widths agree
# ?   encode('utf-8'), emoji    5'251 ns   24.89x  transcode, they do not
# ?
# ? Memory is the boring half. The interesting half is that UTF-8 encoding an
# ? ASCII string is a `memcpy` — the internal representation and UTF-8 agree
# ? byte for byte — while encoding the emoji string transcodes every character
# ? from a 4-byte slot down to 1 to 4 UTF-8 bytes.
# ?
# ? So the cost is not "Unicode is slow". It is that one representation is
# ? already the wire format and the other has to be converted to it, and which
# ? one you have was decided by a single character you probably did not choose.
# ? Serialization, logging and every socket write pay this.

# endregion: Text Representation

# region: Parsing

# ? `re` caches compiled patterns in a dict keyed by pattern, flags and type,
# ? which is why `re.split` feels nearly as fast as a precompiled object and
# ? why precompiling is so often dismissed as premature.
# ?
# ? The cache holds 512 entries and, on overflow, is cleared entirely rather
# ? than evicted from. Patterns interpolated from user input or loop variables
# ? cross that line quietly:
# ?
# ?   ≤512 distinct patterns    lookup hits      ~1x
# ?    >512 distinct patterns   cache clears     every call recompiles
# ?
# ? There is no warning and no gradual decay. The same code is fast in testing
# ? with a handful of patterns and slow in production with thousands.

import re  # noqa: E402
import unicodedata  # noqa: E402
from datetime import datetime  # noqa: E402

_LOG_LINE = "2026-07-27T01:23:45.123456 INFO  worker-7 request=abc123 took=42ms"
_LOG_TEXT = "\n".join(_LOG_LINE for _ in range(10_000))
_CSV_LINE = ",".join(str(field) for field in range(8))
# ! 600 distinct patterns against a 512-entry cache, so every lookup misses.
_MANY_PATTERNS = [rf"\bworker-{index}\b" for index in range(600)]


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_str_split(benchmark):
    """`str.split` on a fixed delimiter — no regex engine involved."""

    def kernel():
        return len(_CSV_LINE.split(","))

    result = benchmark(kernel)
    assert result == 8


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_re_split_cached(benchmark):
    """`re.split` benefits from the module-level pattern cache."""

    def kernel():
        return len(re.split(",", _CSV_LINE))

    result = benchmark(kernel)
    assert result == 8


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_re_split_precompiled(benchmark):
    """A precompiled pattern skips the cache lookup entirely."""
    pattern = re.compile(",")

    def kernel():
        return len(pattern.split(_CSV_LINE))

    result = benchmark(kernel)
    assert result == 8


# ? Intel Xeon 4 • CPython 3.14t • splitting one 8-field line
# ?
# ?   str.split               400 ns    1.00x
# ?   re.split precompiled    680 ns    1.70x
# ?   re.split via cache    1'127 ns    2.82x  the lookup alone is 447 ns
# ?
# ? A dedicated delimiter split beats a regex that matches one literal
# ? character, and the cache lookup — hashing the pattern string, checking
# ? flags and type — costs more than the split it enables.


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_whitespace_str(benchmark):
    """Splitting 10K log lines on whitespace, the built-in way."""

    def kernel():
        return len(_LOG_TEXT.split())

    result = benchmark(kernel)
    assert result > 10_000


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_whitespace_regex(benchmark):
    """The same split, expressed as `\\s+`."""

    def kernel():
        return len(re.split(r"\s+", _LOG_TEXT))

    result = benchmark(kernel)
    assert result > 10_000


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_cache_thrash(benchmark):
    """600 distinct patterns against a 512-entry cache — every call recompiles."""

    def kernel():
        return sum(1 for pattern in _MANY_PATTERNS if re.search(pattern, _LOG_LINE))

    result = benchmark(kernel)
    assert result == 1


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_cache_thrash_precompiled(benchmark):
    """The same 600 patterns, compiled once outside the loop."""
    compiled = [re.compile(pattern) for pattern in _MANY_PATTERNS]

    def kernel():
        return sum(1 for pattern in compiled if pattern.search(_LOG_LINE))

    result = benchmark(kernel)
    assert result == 1


# ? Intel Xeon 4 • CPython 3.14t • 600 distinct patterns, 512-entry cache
# ?
# ?   precompiled            0.53 ms    1.00x
# ?   via the re cache      13.22 ms   24.94x  past 512, every call misses
# ?
# ? This is a production failure mode rather than a tuning knob. Nothing warns
# ? you, the degradation is not gradual, and the trigger — how many distinct
# ? patterns your inputs generate — is usually not visible in the code that
# ? pays for it.


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_translate(benchmark):
    """Stripping punctuation with a translation table."""
    table = str.maketrans("", "", ",!;.")

    def kernel():
        return len(_LOG_TEXT.translate(table))

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_re_sub(benchmark):
    """The same removal, expressed as a character class."""

    def kernel():
        return len(re.sub(r"[,!;.]", "", _LOG_TEXT))

    result = benchmark(kernel)
    assert result > 0


# ? Intel Xeon 4 • CPython 3.14t • rewriting 10K log lines
# ?
# ?   str.translate          1.51 ms    1.00x  one table lookup per character
# ?   str.split              2.60 ms    1.72x
# ?   re.sub(r"[,!;.]")      5.47 ms    3.62x
# ?   re.split(r"\s+")      17.07 ms   11.30x  backtracking on every run
# ?
# ? `str.translate` wins because a translation table turns the whole operation
# ? into an array index per character, with no matching at all. The gap between
# ? the two `str` rows and their regex equivalents is the engine's per-character
# ? bookkeeping, which is real work even when the pattern is trivial.


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_timestamp_isoformat(benchmark):
    """`datetime.fromisoformat` got a C fast path in 3.11."""
    stamp = "2026-07-27T01:23:45.123456"

    def kernel():
        return datetime.fromisoformat(stamp).year

    result = benchmark(kernel)
    assert result == 2026


@pytest.mark.benchmark(group="04-structures-parsing")
def test_parsing_timestamp_strptime(benchmark):
    """The general-purpose parser, which re-reads the format string."""
    stamp = "2026-07-27T01:23:45.123456"
    fmt = "%Y-%m-%dT%H:%M:%S.%f"

    def kernel():
        return datetime.strptime(stamp, fmt).year

    result = benchmark(kernel)
    assert result == 2026


# ? One correctness note, since everything above measures only speed. Lowercase
# ? is not the same operation as case-insensitive:
# ?
# ?   "Straße".lower()      → "straße"     never matches "strasse"
# ?   "Straße".casefold()   → "strasse"    matches
# ?
# ? `str.lower` maps each character to its lowercase form, and ß is already
# ? lowercase. `str.casefold` applies the Unicode folding that expands it to
# ? "ss". Measured on 10K characters they are within 4% of each other — 32.0 µs
# ? against 30.7 µs — so one of these is simply wrong for search and free to
# ? replace.


# ! A correctness demonstration, not a benchmark — the two spellings cost the
# ! same, so there is no timing worth publishing and none is taken.
def test_normalize_casefold_search():
    """Case-insensitive matching the way most code does it, and gets wrong."""
    haystack = _make_ascii_string(10_000) + "Straße"
    assert haystack.lower().find("strasse") == -1
    assert haystack.casefold().find("strasse") != -1


# ? Intel Xeon 4 • CPython 3.14t • parsing one ISO timestamp
# ?
# ?   datetime.fromisoformat  492 ns    1.00x  C fast path since 3.11
# ?   datetime.strptime     8'249 ns   16.77x  re-reads the format string
# ?
# ? `strptime` re-parses `"%Y-%m-%dT%H:%M:%S.%f"` on every call — the format
# ? string is data, interpreted each time — while `fromisoformat` hard-codes
# ? one grammar in C. Sixteen times, for knowing the answer in advance.


# ? The `re.compile` win is fixed per-call overhead rather
# ? than throughput — so it shrinks as the input grows:
# ?
# ? Intel Xeon 4 • CPython 3.14t • timeit, splitting one line on commas
# ?
# ?                        8 fields    10'000 fields
# ?   str.split             101.9 ns        325.7 µs
# ?   precompiled           347.6 ns        357.9 µs
# ?   via the re cache      820.9 ns        371.4 µs
# ?
# ?   precompiled leads        2.36x           1.04x
# ?   str.split leads          3.41x           1.10x
# ?
# ? Every ranking here survives, and every ratio collapses. Precompiling is
# ? worth 2.36x on short inputs and nothing at all on long ones, because the
# ? cost it removes is paid once per call rather than once per character.
# ? Measure at one input size and you publish a constant that is not one.

# endregion: Parsing

# region: Tables and Arrays

# ? Filter a column, then sum what survives. Three libraries, one workload,
# ? and the interesting part is not which wins but that the answer has an
# ? expiry date — the ranking below inverted across a Pandas major version.
# ?
# ? The three differ in where the filtered rows go. A NumPy boolean mask
# ? materializes a new array of the survivors before summing. PyArrow's
# ? compute kernels fuse the filter into the reduction and never build it.
# ? Pandas sits between, and which side it sits closer to is a version detail.


@pytest.mark.benchmark(group="04-structures-tables")
def test_tables_numpy_filter_sum(benchmark):
    """Baseline: filter and sum with NumPy boolean masks."""
    row_count = 100_000
    values = np.random.rand(row_count).astype(np.float64)
    labels = (np.random.rand(row_count) * 10).astype(np.int32)

    def kernel():
        mask = labels % 2 == 0
        return float(values[mask].sum())

    result = benchmark(kernel)
    assert 0.0 <= result <= float(values.sum())


@pytest.mark.skipif(not pandas_installed, reason="Pandas not installed")
@pytest.mark.benchmark(group="04-structures-tables")
def test_tables_pandas_filter_sum(benchmark):
    """Compare: Pandas DataFrame filter and sum on numeric column."""
    row_count = 100_000
    values = np.random.rand(row_count).astype(np.float64)
    labels = (np.random.rand(row_count) * 10).astype(np.int32)
    frame = pd.DataFrame({"value": values, "label": labels})

    def kernel():
        return float(frame.loc[frame["label"].mod(2).eq(0), "value"].sum())

    result = benchmark(kernel)
    assert 0.0 <= result <= float(values.sum())


@pytest.mark.skipif(not pyarrow_installed, reason="PyArrow not installed")
@pytest.mark.benchmark(group="04-structures-tables")
def test_tables_pyarrow_filter_sum(benchmark):
    """Compare: PyArrow Table filter + sum via compute kernels."""
    row_count = 100_000
    values = pa.array(np.random.rand(row_count).astype(np.float64))
    labels = pa.array((np.random.rand(row_count) * 10).astype(np.int32))
    table = pa.table({"value": values, "label": labels})

    def kernel():
        # PyArrow doesn't have a mod function, use bit_wise_and for mod 2
        mask = pc.equal(pc.bit_wise_and(table["label"], 1), 0)
        filtered = table.filter(mask)
        return float(pc.sum(filtered["value"]).as_py())

    result = benchmark(kernel)
    assert result >= 0.0


# ? Apple M2 Pro • Pandas 2.2 • 100K rows, filter even labels then sum
# ?
# ?   PyArrow compute    355 µs    1.00x  columnar kernels win
# ?   Pandas DataFrame   386 µs    1.09x  close behind, more features
# ?   NumPy mask         634 µs    1.79x  surprisingly slower
# ?
# ? Intel Xeon 4 • Pandas 3.0 • same workload
# ?
# ?   PyArrow compute    651 µs    1.00x  still wins
# ?   NumPy mask         862 µs    1.32x  now ahead of Pandas
# ?   Pandas DataFrame   911 µs    1.40x  lost its second place
# ?
# ? PyArrow's compute kernels are optimized for columnar data and win on both
# ? machines — that part of the lesson is stable. The Pandas-vs-NumPy ordering
# ? is not: Pandas 3.0 turned on Copy-on-Write and Arrow-backed strings, and on
# ? this workload it dropped behind NumPy instead of trailing PyArrow closely.
# ? A reminder that a major version of a dependency can invert a ranking you
# ? measured once and copied into a comment forever.


# endregion: Tables and Arrays

# endregion: Data Structures
