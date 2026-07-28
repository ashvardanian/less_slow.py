#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reflection — attribute lookup, `eval`, and the cost of compiling at runtime.

Reflection itself is cheap: `getattr` costs a string lookup and little else,
so occasional use is free. Compilation is what costs. `eval` on a string pays
the parser on every single call, and precompiling the same source once with
`compile()` drops it by more than an order of magnitude to bytecode execution.

`ast.literal_eval` is the safe way to read a literal, and on this machine it
is also the faster of the two literal paths — which was not true on the
hardware this table previously described, so treat the ordering as local.
"""

import pytest

# region: Dynamic Code

# region: Reflection, Inspection

# ? Python's dynamic features — getattr, eval, introspection — are powerful
# ? but come with performance costs. Understanding these costs helps you
# ? decide when dynamic dispatch is worth it and when to use alternatives.

import ast  # noqa: E402


class _SmallObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def double(self) -> int:
        return self.value * 2


@pytest.mark.benchmark(group="07-reflection-lookup-vs-eval")
def test_reflection_direct_access(benchmark):
    """Baseline: direct attribute access."""
    holder = _SmallObject(7)

    def kernel():
        total = 0
        for _ in range(10_000):
            total += holder.value
        return total

    result = benchmark(kernel)
    assert result == 70_000


@pytest.mark.benchmark(group="07-reflection-lookup-vs-eval")
def test_reflection_getattr(benchmark):
    """Access attribute via getattr() — string lookup overhead."""
    holder = _SmallObject(7)

    def kernel():
        total = 0
        for _ in range(10_000):
            total += getattr(holder, "value")
        return total

    result = benchmark(kernel)
    assert result == 70_000


@pytest.mark.benchmark(group="07-reflection-lookup-vs-eval")
def test_reflection_eval_string(benchmark):
    """eval() with string source — must parse and compile each time."""
    source = "x + y"

    def kernel():
        total = 0
        for _ in range(1_000):
            total += eval(source, {}, {"x": 1, "y": 2})
        return total

    result = benchmark(kernel)
    assert result == 3_000


@pytest.mark.benchmark(group="07-reflection-lookup-vs-eval")
def test_reflection_eval_compiled(benchmark):
    """eval() with precompiled code object — skip parsing/compilation."""
    source = "x + y"
    code_object = compile(source, filename="<expr>", mode="eval")

    def kernel():
        total = 0
        for _ in range(1_000):
            total += eval(code_object, {}, {"x": 1, "y": 2})
        return total

    result = benchmark(kernel)
    assert result == 3_000


@pytest.mark.benchmark(group="07-reflection-lookup-vs-eval")
def test_reflection_literal_eval(benchmark):
    """ast.literal_eval() — safe parsing of literal expressions."""
    text = "[1, 2, 3, 4, 5]"

    def kernel():
        total = 0
        for _ in range(1_000):
            total += sum(ast.literal_eval(text))
        return total

    result = benchmark(kernel)
    assert result == 15_000


@pytest.mark.benchmark(group="07-reflection-lookup-vs-eval")
def test_reflection_eval_literal(benchmark):
    """eval() on literal — faster but unsafe for untrusted input."""
    text = "[1, 2, 3, 4, 5]"

    def kernel():
        total = 0
        for _ in range(1_000):
            total += sum(eval(text))
        return total

    result = benchmark(kernel)
    assert result == 15_000


# ? Intel Xeon 4 · CPython 3.14t · per operation, normalized by loop count
# ?
# ?   direct access    17.8 ns      1.00x
# ?   getattr()        31.6 ns      1.78x  a string lookup
# ?   eval(compiled)    156 ns      8.78x  bytecode only
# ?   eval(string)    4'049 ns    227.67x  parses every call
# ?   literal_eval()  8'162 ns    458.90x
# ?   eval(literal)   8'640 ns    485.82x
# ?
# ? Reflection is cheap; *compilation* is not. `getattr` costs a string lookup
# ? and nothing more, so occasional use is free. But `eval` on a string pays
# ? the parser every single call — precompile once with `compile()` and the
# ? same work drops to bytecode execution, a 26x difference. `ast.literal_eval`
# ? is the safe choice and here also the faster of the two literal paths, which
# ? was not true on the machine this table previously described.


# endregion: Reflection, Inspection

# endregion: Dynamic Code
