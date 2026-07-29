#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reflection — attribute lookup, `eval`, and the cost of compiling at runtime.

Reflection itself is cheap: `getattr` costs 1.78x a direct attribute read, so
occasional use is free. Compilation is what costs. `eval` on a string pays the
parser on every call, and precompiling the same source once with `compile()`
drops that 25.96x to bytecode execution.

The same shape appears wherever text becomes a plan — `re.compile`, prepared
statements, parsed templates — and most expensively in `inspect.signature`,
which costs 7.7 µs live against 19 ns for one built earlier. That is 405x, and
it is why every framework that reads signatures reads them at decoration time.
"""

import pytest

# region: Dynamic Code

# region: Reflection, Inspection

# ? "Reflection is slow" bundles two costs that differ by three orders of
# ? magnitude, and only one of them deserves the reputation:
# ?
# ?   look something up by name     a dict probe          nanoseconds
# ?   turn source text into code    tokenize, parse, compile   microseconds
# ?
# ? `getattr` is the first. `eval` on a string is the second, and it is the
# ? second every single time it is called, because nothing caches the result.

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


# ? Intel Xeon 4 • CPython 3.14t • per attribute read, 10K per round
# ?
# ?   holder.value              17.8 ns    1.00x  a specialized load
# ?   getattr(holder, "value")  31.6 ns    1.78x  a global lookup, then a call
# ?
# ? Fourteen nanoseconds, and none of it is the dictionary — `holder.value`
# ? reads the same dict. The difference is that `getattr` is a global name and
# ? a call frame, the same tax the truthiness idiom pays.
# ?
# ? So reflection by name is not expensive. Code that does it once per request
# ? cannot measure it, and the reason to avoid it is that a name assembled at
# ? runtime cannot be checked by anything before it runs.


# ? Compilation is the other cost, and it is a different order of magnitude.
# ? `eval` on a string does the whole front end on every call:
# ?
# ?   eval("x + y")     tokenize → parse → compile → execute
# ?   eval(code)        execute
# ?
# ? Nothing between those two calls is cached. The parse is repeated because
# ? the string might have changed, and Python has no way to know it did not.


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


# ? Intel Xeon 4 • CPython 3.14t • per `x + y` evaluation, 1K per round
# ?
# ?   eval(code_object)      156 ns     1.00x  bytecode only
# ?   eval("x + y")        4'049 ns    25.96x  parses on every call
# ?
# ? Twenty-six times, for work whose result was identical every time. One
# ? `compile()` outside the loop removes all of it.
# ?
# ? This generalizes past `eval`, and that is the reason to care: `re.compile`
# ? against a bare `re.match`, a prepared statement against a formatted query,
# ? a parsed template against `str.format` on a template string. The shape is
# ? always the same — something turns text into a plan, and the plan can be
# ? kept. `dataclasses` and `attrs` are the extreme case, generating source and
# ? `exec`-ing it once at class creation so every instance after that is free.


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


# ? Intel Xeon 4 • CPython 3.14t • per parse of "[1, 2, 3, 4, 5]" and its sum
# ?
# ?   eval(code_object)      156 ns     1.00x  no parsing at all
# ?   ast.literal_eval()   8'162 ns    52.32x  parses, every call
# ?
# ? Reading a five-element literal costs 52x running precompiled bytecode.
# ? That is the same result arriving from a different direction: the parser is
# ? the expense, and `literal_eval` is nothing but parser.


# ? `eval` would parse the same text within 6% — measured, and not published,
# ? because a 6% gap whose ordering does not survive a change of machine is not
# ? a result. The two are not competing on speed at all:
# ?
# ?   ast.literal_eval("[1, 2]")     builds a list
# ?   ast.literal_eval("f()")        raises ValueError
# ?   eval("[1, 2]")                 builds a list
# ?   eval("f()")                    calls f
# ?
# ? `eval` runs whatever the text describes — an import, a subprocess spawn, a
# ? file deletion. `literal_eval` walks the parse tree and refuses anything but
# ? literals. That is the entire decision, and since the costs are equal there
# ? is nothing to trade against it.

# endregion: Reflection, Inspection

# region: Introspection

# ? Reading a name is cheap. Asking what a *function* looks like is not, and
# ? the gap is why every framework that touches signatures caches them.


import inspect  # noqa: E402


def _annotated(alpha, beta=2, *args, gamma=3, **keywords):
    """A signature with enough shape to be worth inspecting."""
    return alpha


@pytest.mark.benchmark(group="07-reflection-introspection")
def test_introspection_signature_live(benchmark):
    """Building a `Signature` from scratch, the way a naive decorator would."""

    def kernel():
        return len(inspect.signature(_annotated).parameters)

    assert benchmark(kernel) == 5


@pytest.mark.benchmark(group="07-reflection-introspection")
def test_introspection_signature_cached(benchmark):
    """Reading one built once — what every real framework does."""
    signature = inspect.signature(_annotated)

    def kernel():
        return len(signature.parameters)

    assert benchmark(kernel) == 5


# ? Intel Xeon 4 • CPython 3.14t • one signature inspection
# ?
# ?   cached Signature       19.0 ns    1.00x
# ?   inspect.signature()   7'703 ns     405x  walks the code object
# ?
# ? Four hundred times, and 7.7 µs is enormous next to the 11 ns a call costs.
# ? A decorator calling `inspect.signature` per invocation would cost 700 times
# ? the function it wraps.
# ?
# ? Which is exactly the compile-once shape again, in a place people meet it
# ? without recognizing it. FastAPI, pytest and `functools.wraps` all inspect
# ? at decoration time and never again — the work happens once at import, and
# ? the per-call path reads a stored result.

# endregion: Introspection

# endregion: Dynamic Code
