# _Less Slow_ Python

> The spiritual little brother of [`less_slow.cpp`](https://github.com/ashvardanian/less_slow.cpp).
> Assuming Python is used in a different setting than C++, this repository focuses more on scripting, tool integration, and data processing.
> The benchmarks in this repository don't aim to cover every topic entirely, but they help form a mindset and intuition for performance-oriented software design.

Much modern code suffers from common pitfalls: bugs, security vulnerabilities, and performance bottlenecks.
University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.

![Less Slow Python](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.py.jpg?raw=true)

This repository offers practical examples of writing efficient Python code.
The topics range from basic micro-kernels executing in a few nanoseconds to more complex constructs involving parallel algorithms, coroutines, and polymorphism. Some of the highlights include:

- A single emoji in a 10K-char ASCII string quadruples its size and also makes `encode('utf-8')` - 25x slower.
- NumPy matrix multiplication with `int32` is over 100x slower than `float32` or `float64`, because BLAS has no kernel for integers at all.
- `if value:` is ~1.5x faster than `if len(value) > 0`, which pays for a global lookup and a call to read the same length.
- Binding `push = out.append` before a loop was the classic CPython speed-up; it is now 1.13x _slower_ than plain `out.append(x)`, because the shortcut defeats the interpreter's specialized method call.
- Async IO, batching, HTTPX, and FastAPI won't save you from slow IO: an HTTP round trip costs 58-112x a raw socket carrying the same bytes.
- Sending 16 HTTPX requests concurrently is 1.95x slower than sending them one at a time, because the uvicorn server behind them runs a single worker and handles them serially anyway.
- Using callbacks, lambdas, and `yield`-ing functions are much faster than iterator-based routines, unlike Rust and C++.
- Not all composite structures are equally fast: `namedtuple` is slower than { `dataclass`, `class` } is slower than `dict`.
- `sys.getsizeof` reports 48 bytes for a dataclass instance that really costs 185 — and `__slots__` saves 1.3x, not the 2–3x folklore.
- Slicing 900 KB of `bytes` copies it; slicing a `memoryview` doesn't — and the gap grows with the slice, without bound.
- Exceptions are the _fastest_ error-handling style below a ~15% failure rate and the slowest above 20%; only their cost moves with the rate.
- Clearing and refilling one list to avoid allocating a fresh one is 1.6x _slower_ than just allocating, because two method calls cost more than the allocation they replace.
- `gc.disable()` around a bulk loop changes nothing at all — 1.00x, even on garbage that forms reference cycles, which is the only kind the collector exists to handle.
- NumPy only overtakes the `math` module past 8 to 16 elements: it costs ~340 ns before touching a single one, against ~40 ns per element for a plain Python loop.
- JIT compilers like Numba can leave you 2.1x slower than `math.sin`, because the kernel is compiled but the loop calling it ten thousand times is not.

The benchmarks are split into numbered chapters, ordered from the silicon
outward — arithmetic, then memory, then data structures, then the interpreter,
then the OS, then the network:

```
less_slow_01_basics.py         less_slow_06_errors.py
less_slow_02_numerics.py       less_slow_07_reflection.py
less_slow_03_memory.py         less_slow_08_parallelism.py
less_slow_04_structures.py     less_slow_09_accelerators.py
less_slow_05_abstractions.py   less_slow_10_networking.py
```

Read them in order, or jump to whichever chapter you are most curious about.

## Reproducing the Benchmarks

If you are familiar with Python and want to review code and measurements as you read, you can clone the repository and execute the following commands to install the dependencies and run the benchmarks in your local environment.

```sh
git clone https://github.com/ashvardanian/less_slow.py.git # Clone the repository
cd less_slow.py                                            # Change the directory
pip install -r requirements.txt                            # Install the dependencies
pytest                                                     # Run all benchmarks
pytest less_slow_02_numerics.py                            # Run one chapter
pytest -x -k echo                                          # Filter and stop on failure
```

Alternatively, run the benchmarks in a controlled environment using [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync                          # Create .venv from the lockfile
uv run pytest -ra -q             # Run all benchmarks
```

`.python-version` pins `3.14t`, the free-threaded build — `requires-python` can only state a version range and has no way to name the no-GIL ABI, so without the pin `uv` picks the GIL build.
Pass `--python="3.14"` to compare the two; the suite passes on 3.10 through 3.14, free-threaded or not.

For `pytest`, the `-r` flag can be used to display a "short test summary info" at the end of the test session, making it easy to get a clear picture of all failures in large test suites.
The `-ra` variant limits the summary only to failed tests, avoiding "passed" and "passed with outputs" messages.

Dependencies live in `pyproject.toml` and are locked in `uv.lock`, which covers every platform and every Python from `requires-python` up.
The `requirements.txt` is generated from that lockfile so readers without `uv` keep a working `pip` path — regenerate it whenever dependencies change.

```sh
uv add <package>        # Or edit [project].dependencies, then `uv lock`
uv lock --upgrade       # Refresh every pin to the newest compatible release
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

## Citation

If this repository helps your research, teaching, or product, please cite it:

```bibtex
@software{Vardanian_less_slow_py,
  author = {Vardanian, Ash},
  title = {{less_slow.py: Less Slow Coding Practices in Python}},
  doi = {10.5281/zenodo.21626661},
  url = {https://github.com/ashvardanian/less_slow.py},
  license = {Apache-2.0}
}
```
