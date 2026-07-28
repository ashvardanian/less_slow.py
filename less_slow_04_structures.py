#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data structures — what a record costs to build, to hold, and to search.

Four questions about the same data. Constructing it: a bare tuple is the
floor and `namedtuple` — reached for precisely because it sounds like the
performance choice — is the slowest of the built-ins. Holding it: `sys.getsizeof`
reports 48 bytes for a dataclass instance that actually costs 185, because
since 3.11 the attributes live in an inline preheader it never walks, and
`__slots__` saves 1.28x rather than the 2-3x folklore.

Then text, where the sharp result is that `re` caches compiled patterns in a
512-entry dict: exceed it with patterns built from user input and every call
silently recompiles, at 25x. And tables, where a major version of Pandas
inverted a ranking this file had already published — a reminder that measured
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

# ? Python has many ways of defining composite objects. The most common
# ? are tuples, dictionaries, named-tuples, dataclasses, and classes.
# ? Let's compare them by assembling a simple composite of numeric values.
# ?
# ? We prefer `float` and `bool` fields as the most predictable Python types,
# ? as the `int` integers in Python involve a lot of additional logic for
# ? arbitrary-precision arithmetic, which can affect the latencies.

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


# ? Python's dynamic nature allows users to override builtin functions, this means that
# ? calls to `dict` (or `list`, `tuple`, etc) require a lookup before execution.
# ? However, that does not apply to `{}` (or `[]`, etc) since it is a language construct
# ? which users are unable to override, as such, the Python interpreter is free to
# ? *simply* create the dictionary instead of performing a lookup before.
# ? The bytecode makes it obvious: `{"a": 1}` compiles to a single `BUILD_MAP`,
# ? while `dict(a=1)` compiles to `LOAD_GLOBAL` + `CALL_KW` — and because that
# ? global is user-overridable, the interpreter cannot skip the lookup.
# ?
# ? Intel Xeon 4 · CPython 3.14t · one dict per call
# ?
# ?   {"x": 1.0, ...}      88.4 ns    1.00x  one BUILD_MAP
# ?   dict(x=1.0, ...)    139.9 ns    1.58x  global lookup, then call


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


# ? Intel Xeon 4 · CPython 3.14t · one instance built and read per call
# ?
# ?   tuple                62.6 ns    1.00x
# ?   dict                  112 ns    1.79x
# ?   __slots__ dataclass   192 ns    3.07x
# ?   class                 202 ns    3.23x
# ?   dataclass             203 ns    3.24x
# ?   namedtuple            298 ns    4.76x  the "fast" one, in folklore
# ?   attrs                 544 ns    8.69x
# ?   pydantic              869 ns   13.89x  validation is not free
# ?
# ? A bare tuple is the floor, and `namedtuple` — reached for precisely
# ? because it sounds like a performance choice — is the slowest of the six.
# ? None of them validates field types, which is why so much Python reaches
# ? for `pydantic` or `attrs` instead. Both are measured below, and both cost
# ? more than everything in this table.

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

# ? The section above times construction. Memory is the other half of the story,
# ? and the obvious tool for it is a trap. `sys.getsizeof` is shallow — it never
# ? follows references — and since CPython 3.11 it does not even count an
# ? instance's own attribute storage, which lives in an inline preheader rather
# ? than a separate `__dict__`. So it under-reports the very objects you most
# ? want to measure.
# ?
# ? `tracemalloc` sees what actually happened. We allocate many objects with
# ? distinct field values — reusing `1.0` would share one float and understate
# ? every row — and divide the delta by the count.

import tracemalloc  # noqa: E402


def _bytes_per_item(factory, count: int = 10_000) -> float:
    """Allocate `count` distinct objects, reporting the true bytes each costs."""
    gc.collect()
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    items = [factory(index) for index in range(count)]
    peak, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(items) == count
    return (peak - baseline) / count


@pytest.mark.benchmark(group="04-structures-footprint")
def test_footprint_dict(benchmark):
    """A `dict` per point — the most flexible layout, and the fattest."""

    def kernel():
        return _bytes_per_item(
            lambda index: {"x": float(index), "y": float(index), "flag": True}
        )

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="04-structures-footprint")
def test_footprint_dataclass(benchmark):
    """A dataclass — and the gap between reported and real size."""
    reported = sys.getsizeof(PointDataclass(1.0, 2.0, True))

    def kernel():
        return _bytes_per_item(
            lambda index: PointDataclass(float(index), float(index), True)
        )

    result = benchmark(kernel)
    benchmark.extra_info["getsizeof"] = reported
    # ! `getsizeof` under-reports by roughly 4x — this assert is the lesson,
    # ! and it will start failing the day CPython learns to count honestly.
    assert result > reported


@pytest.mark.benchmark(group="04-structures-footprint")
def test_footprint_slots_dataclass(benchmark):
    """`__slots__` drops the per-instance dictionary."""

    def kernel():
        return _bytes_per_item(
            lambda index: PointSlotsDataclass(float(index), float(index), True)
        )

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="04-structures-footprint")
def test_footprint_namedtuple(benchmark):
    """A `namedtuple` is a tuple, so it pays no dictionary at all."""

    def kernel():
        return _bytes_per_item(
            lambda index: PointNamedtuple(float(index), float(index), True)
        )

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="04-structures-footprint")
def test_footprint_tuple(benchmark):
    """A bare tuple — no field names, no class, no dictionary."""

    def kernel():
        return _bytes_per_item(lambda index: (float(index), float(index), True))

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="04-structures-footprint")
def test_footprint_class(benchmark):
    """A hand-written class, for comparison with the generated dataclass."""

    def kernel():
        return _bytes_per_item(
            lambda index: PointClass(float(index), float(index), True)
        )

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="04-structures-footprint")
def test_footprint_numpy_structured(benchmark):
    """One buffer for every point — no per-object header, no pointers."""
    dtype = np.dtype([("x", np.float64), ("y", np.float64), ("flag", np.bool_)])

    def kernel():
        count = 10_000
        gc.collect()
        tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()
        points = np.zeros(count, dtype=dtype)
        points["x"] = np.arange(count, dtype=np.float64)
        points["y"] = np.arange(count, dtype=np.float64)
        peak, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert points.shape == (count,)
        return (peak - baseline) / count

    result = benchmark(kernel)
    assert result > 0


# ? Footprint is only half of the layout question. Array-of-Structs and
# ? Struct-of-Arrays hold identical bytes; what differs is the stride you walk
# ? when you touch one field.


@pytest.mark.benchmark(group="04-structures-footprint-access")
def test_access_array_of_structs(benchmark):
    """Sum one field of a structured array — stride is the whole struct."""
    dtype = np.dtype([("x", np.float64), ("y", np.float64), ("flag", np.bool_)])
    points = np.zeros(1_000_000, dtype=dtype)
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


# ? Intel Xeon 4 · CPython 3.14t · 10K instances, distinct field values
# ?
# ?   NumPy structured      17.0 B    1.00x  one buffer, no headers
# ?   __slots__ dataclass  144.6 B    8.49x  no instance dictionary
# ?   tuple                160.5 B    9.43x
# ?   namedtuple           168.6 B    9.90x  a tuple plus its class
# ?   class                184.9 B   10.85x
# ?   dataclass            184.9 B   10.85x  getsizeof claims 48
# ?   dict                 272.5 B   16.00x  the fattest
# ?
# ? Two pieces of folklore die here. `sys.getsizeof` reports 48 bytes for a
# ? dataclass instance that costs 185, a 3.9x under-report, because since 3.11
# ? the attribute values sit in an inline preheader it never walks. And
# ? `__slots__` buys 1.28x, not the 2-3x usually quoted — key-sharing
# ? dictionaries plus those inline values already captured most of that win.
# ?
# ? Measuring it is harder than it looks. Any recursive sizer that reads
# ? `obj.__dict__` *materializes* the inline values into a real dictionary,
# ? adding 64 B per instance — the tool changes what it measures. And a factory
# ? returning `PointDataclass(1.0, 2.0, True)` shares three objects across all
# ? ten thousand instances, understating every row by 80 B.
# ?
# ? Free-threading is not free either: `sys.getsizeof(1.0)` is 40 bytes on
# ? 3.14t against 24 on the GIL build of the same version, the difference being
# ? header fields lock-free refcounting needs. Only small immutables grew.
# ?
# ? Intel Xeon 4 · CPython 3.14t · summing one field of 1M elements
# ?
# ?   struct-of-arrays      406 µs    1.00x  stride 8, contiguous
# ?   array-of-structs    1'096 µs    2.70x  stride 17, neither
# ?
# ? Both layouts hold the same bytes. The field view of a structured array
# ? inherits the struct's 17-byte stride, so it is neither contiguous nor
# ? aligned, and cannot be fed to a vectorized loop.

# endregion: Object Footprint

# region: Heterogeneous Data

# ? Python is a dynamically typed language, and it allows mixing different
# ? types in a single collection. However, the performance of such collections
# ? can vary significantly, depending on the types and their distribution.
# ?
# ? Consider a common scenario: you receive numeric data from various sources
# ? (user input as strings, database results as Decimals, API responses as ints)
# ? and need to aggregate them. What's the fastest way to sum 80,000 values
# ? when they come in 8 different representations of the same number?


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
        return sum(float(v) for v in values)

    result = benchmark(kernel)
    assert abs(result - 3.0 * len(values)) < 1e-9


@pytest.mark.benchmark(group="04-structures-coercion")
def test_type_matching_sum(benchmark):
    """Use isinstance() to dispatch: add directly or coerce with float()."""
    seed = _represent_with_different_types()
    values = seed * 10_000

    def kernel():
        sum_value = 0.0
        for v in values:
            if isinstance(v, (int, float, np.integer, np.floating)):
                sum_value += v
            else:
                sum_value += float(v)
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
        for v in values:
            try:
                sum_value += v
            except TypeError:
                sum_value += float(v)
        return sum_value

    result = benchmark(kernel)
    assert abs(result - 3.0 * len(values)) < 1e-9


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


# ? The results reveal a counter-intuitive hierarchy of approaches:
# ?
# ? Intel Xeon 4 · CPython 3.14t · 80K values, eight representations
# ?
# ?   np.sum, homogeneous     17.7 µs      1.00x
# ?   sum(), homogeneous       273 µs     15.44x
# ?   float() on everything  3'694 µs    208.69x
# ?   isinstance dispatch   16'068 µs    907.67x
# ?   try/except, EAFP      50'147 µs     2'833x
# ?
# ? The "Easier to Ask Forgiveness than Permission" (EAFP) pattern, often
# ? recommended in Python, is 19x slower than uniform coercion here! Why?
# ? Because ≈25% of our values — Decimal, Fraction, strings — raise TypeError
# ? when added to a float, and Python exceptions are expensive.
# ?
# ? Even more surprising: isinstance() checks are 3.3x slower than just
# ? calling float() on everything, despite avoiding the call for 50% of
# ? elements. The isinstance() overhead itself is more expensive than
# ? the redundant float() calls it's trying to avoid.
# ?
# ? The lesson: if you must work with heterogeneous numeric data, convert
# ? everything to a uniform type upfront. Better yet, ensure homogeneity
# ? at the data ingestion layer — a numpy array is 247x faster than
# ? iterating with float() coercion.


# ? Beyond numeric types, Python's string representation also has hidden costs.
# ? PEP 393 introduced "flexible string representation" where the internal
# ? encoding depends on the "widest" character in the string:
# ?
# ?   - ASCII-only (Latin-1): 1 byte per character
# ?   - BMP Unicode (UCS-2): 2 bytes per character (chars up to U+FFFF)
# ?   - Full Unicode (UCS-4): 4 bytes per character (emojis, rare scripts)
# ?
# ? A single emoji in an otherwise ASCII string causes the entire string to
# ? use 4 bytes per character. This affects memory, but also operations like
# ? substring search and hashing — critical for dict lookups.


def _make_ascii_string(length: int = 10_000) -> str:
    """Create a pure ASCII string."""
    return "a" * length


def _make_emoji_string(length: int = 10_000) -> str:
    """Create a string with emojis, forcing UCS-4 (4 bytes/char) representation."""
    return ("a" * 9 + "\U0001f525") * (length // 10)  # fire emoji


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_encode_ascii(benchmark):
    """Encode ASCII string to UTF-8 bytes."""
    s = _make_ascii_string(10_000)

    def kernel():
        return s.encode("utf-8")

    result = benchmark(kernel)
    assert len(result) == 10_000  # 1 byte per char


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_encode_emoji(benchmark):
    """Encode emoji string to UTF-8 bytes — emojis become 4 bytes each."""
    s = _make_emoji_string(10_000)

    def kernel():
        return s.encode("utf-8")

    result = benchmark(kernel)
    assert len(result) > 10_000  # emojis expand to 4 bytes


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_join_ascii(benchmark):
    """Join many ASCII strings."""
    parts = ["a" * 100] * 100  # 100 strings of 100 chars

    def kernel():
        return "".join(parts)

    result = benchmark(kernel)
    assert len(result) == 10_000


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_join_emoji(benchmark):
    """Join many emoji strings — must handle 4-byte chars."""
    parts = [("a" * 9 + "\U0001f525") * 10] * 100  # 100 strings of 100 chars

    def kernel():
        return "".join(parts)

    result = benchmark(kernel)
    assert len(result) == 10_000


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_hash_ascii(benchmark):
    """Hash ASCII string — used in every dict lookup."""
    s = _make_ascii_string(10_000)

    def kernel():
        return hash(s)

    result = benchmark(kernel)
    assert isinstance(result, int)


@pytest.mark.benchmark(group="04-structures-string-encodings")
def test_string_hash_emoji(benchmark):
    """Hash emoji string — same length, different internal representation."""
    s = _make_emoji_string(10_000)

    def kernel():
        return hash(s)

    result = benchmark(kernel)
    assert isinstance(result, int)


# ? The memory difference is dramatic — 4x for the same logical content:
# ?
# ? Intel Xeon 4 · CPython 3.14t · sys.getsizeof of the string object
# ?
# ?   ASCII, 10K chars    10'041 B    1.00x  1.00 bytes per character
# ?   emoji, 10K chars    40'060 B    3.99x  4.01 bytes per character
# ?
# ? Performance varies dramatically by operation:
# ?
# ? Intel Xeon 4 · CPython 3.14t · 10K-character strings
# ?
# ?   hash(ascii)     68.9 ns     1.00x
# ?   hash(emoji)      121 ns     1.76x
# ?   encode(ascii)    211 ns     3.06x
# ?   join(ascii)      444 ns     6.45x
# ?   join(emoji)    1'304 ns    18.94x
# ?   encode(emoji)  5'251 ns    76.25x
# ?
# ? The encode() operation shows the most dramatic difference — converting
# ? a UCS-4 string to UTF-8 requires examining each 4-byte character and
# ? encoding it as 1-4 bytes. For emojis specifically, each becomes 4 UTF-8
# ? bytes. This 40x slowdown matters for serialization, file I/O, and
# ? network protocols that use UTF-8.


# ? Encoding is covered above; parsing is the other half of every text
# ? pipeline. The folklore here is unusually wrong, so it is worth measuring
# ? rather than repeating.
# ?
# ? `re` caches compiled patterns in a dict keyed by pattern, flags and type,
# ? which is why `re.split` feels almost as fast as a precompiled object. The
# ? cache holds 512 entries. Exceed it — patterns built from user input, or
# ? interpolated in a loop — and every single call recompiles.

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


@pytest.mark.benchmark(group="04-structures-normalization")
def test_normalize_nfkc(benchmark):
    """NFKC folds compatibility characters — step one of corpus cleaning."""
    text = _make_emoji_string(10_000) + "ﬁancé ½ Ⅳ ｆｕｌｌｗｉｄｔｈ"

    def kernel():
        return len(unicodedata.normalize("NFKC", text))

    result = benchmark(kernel)
    assert result > 0


@pytest.mark.benchmark(group="04-structures-normalization")
def test_normalize_casefold_search(benchmark):
    """Case-insensitive matching the way most code does it, and gets wrong."""
    haystack = _make_ascii_string(10_000) + "Straße"

    def kernel():
        return haystack.lower().find("strasse")

    result = benchmark(kernel)
    # ! `str.lower` maps ß to ß, not ss, so this search fails. `casefold` is
    # ! the correct primitive — 'Straße'.casefold() == 'strasse'.
    assert result == -1
    assert "Straße".casefold() == "strasse"


# ? Intel Xeon 4 · CPython 3.14t · splitting one 8-field line
# ?
# ?   str.split               400 ns    1.00x
# ?   re.split precompiled    680 ns    1.70x
# ?   re.split via cache    1'127 ns    2.82x  the lookup alone is 447 ns
# ?
# ? Intel Xeon 4 · CPython 3.14t · parsing one ISO timestamp
# ?
# ?   datetime.fromisoformat  492 ns    1.00x  C fast path since 3.11
# ?   datetime.strptime     8'249 ns   16.77x  re-reads the format string
# ?
# ? Intel Xeon 4 · CPython 3.14t · rewriting 10K log lines
# ?
# ?   str.translate          1.51 ms    1.00x
# ?   str.split              2.60 ms    1.72x
# ?   re.sub(r"[,!;.]")      5.47 ms    3.62x
# ?   re.split(r"\s+")      17.07 ms   11.30x
# ?
# ? Intel Xeon 4 · CPython 3.14t · 600 distinct patterns, 512-entry cache
# ?
# ?   precompiled            0.53 ms    1.00x
# ?   via the re cache      13.22 ms   24.94x  past 512, every call misses
# ?
# ? That last pair is a production failure mode rather than a tuning knob:
# ? patterns interpolated in a loop or built from user input silently turn
# ? every call into a recompile, and nothing warns you.
# ?
# ? The `re.compile` win elsewhere is fixed per-call overhead, not throughput.
# ? On 8 fields precompiling leads by 2.14x; at 10'000 fields the two are
# ? identical and `str.split` leads a regex by only 1.21x. Benchmark a single
# ? input size and you will publish a constant that is not one.
# ?
# ? One correctness note, since this chapter measures only speed: `str.lower`
# ? leaves ß alone, so lowercase matching never finds "strasse" in "Straße".
# ? `str.casefold` is the primitive that does.


# ? The Numeric Types chapter showed which dtypes BLAS has kernels for. It
# ? also wants contiguous memory — C-order or Fortran-order — and a strided
# ? view like `arr[::2]` disqualifies an operand no matter how good its dtype.


@pytest.mark.benchmark(group="04-structures-layouts")
def test_layouts_sum_contiguous(benchmark):
    """Sum of C-contiguous array — optimal memory access pattern."""
    size = 1_000_000
    arr = np.arange(size, dtype=np.float64)

    def kernel():
        return np.sum(arr)

    result = benchmark(kernel)
    benchmark.extra_info["contiguous"] = arr.flags["C_CONTIGUOUS"]
    assert result > 0


@pytest.mark.benchmark(group="04-structures-layouts")
def test_layouts_matmul_strided(benchmark):
    """Matmul with strided input — forces internal copy or slow path."""
    size = 200
    rng = np.random.default_rng(42)
    # Create strided views by slicing every other row/column
    A_full = rng.random((size * 2, size * 2))
    B_full = rng.random((size * 2, size * 2))
    A = A_full[::2, ::2]  # strided view
    B = B_full[::2, ::2]

    def kernel():
        return A @ B

    result = benchmark(kernel)
    benchmark.extra_info["A_contiguous"] = A.flags["C_CONTIGUOUS"]
    assert result.shape == (size, size)


# ? The dtype impact on matrix multiplication (200×200) is substantial:
# ?
# ? A strided view cannot be handed to BLAS at all, so `matmul` falls back to
# ? the same generic loop that makes float16 slow:
# ?
# ? Intel Xeon 4 · CPython 3.14t · same 200×200 matmul, best of five
# ?
# ?   float32 contiguous    38.4 µs    1.00x  BLAS
# ?   float32 strided        174 µs    4.54x  no BLAS
# ?
# ? Two conditions get you the fast path: a dtype BLAS implements, and
# ? contiguous memory. Miss either and the fallback loop is waiting.
# ?
# ? Best of five runs, and it needs to be — BLAS timings swung 12x between
# ? consecutive runs on this shared machine, because multithreaded kernels
# ? compete for cores while the fallback loop does not.


# endregion: Heterogeneous Data

# region: Tables and Arrays

# ? Tabular data processing is central to data science and analytics.
# ? NumPy, Pandas, and PyArrow each offer different trade-offs for
# ? filtering and aggregating data. NumPy is fastest for simple numeric
# ? operations, but Pandas and PyArrow add convenience and handle
# ? mixed types better.


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


# ? Apple M2 Pro · Pandas 2.2 · 100K rows, filter even labels then sum
# ?
# ?   PyArrow compute    355 µs    1.00x  columnar kernels win
# ?   Pandas DataFrame   386 µs    1.09x  close behind, more features
# ?   NumPy mask         634 µs    1.79x  surprisingly slower
# ?
# ? Intel Xeon 4 · Pandas 3.0 · same workload
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
