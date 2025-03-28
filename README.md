# _Less Slow_ Python

> The spiritual little brother of [`less_slow.cpp`](https://github.com/ashvardanian/less_slow.cpp).
> Assuming Python is used in a different setting than C++, this repository focuses more on scripting, tool integration, and data processing.
> The benchmarks in this repository don't aim to cover every topic entirely, but they help form a mindset and intuition for performance-oriented software design.

Much modern code suffers from common pitfalls: bugs, security vulnerabilities, and performance bottlenecks.
University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.

![Less Slow Python](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.py.jpg?raw=true)

This repository offers practical examples of writing efficient Python code.
The topics range from basic micro-kernels executing in a few nanoseconds to more complex constructs involving parallel algorithms, coroutines, and polymorphism. Some of the highlights include:

- Async IO, batching, HTTPX, and FastAPI won't save you from slow IO, potentially resulting in 30x slowdowns compared to the already slow Python-native TCP/IP stack.
- Using callbacks, lambdas, and `yield`-ing functions are much faster than iterator-based routines, unlike Rust and C++.
- Not all composite structures are equally fast: `namedtuple` is slower than { `dataclass`, `class` } is slower than `dict`.
- Depending on your design, error handling with status codes can be 50% faster or 2x slower than exceptions.
- NumPy-based logic can be much slower than `math` functions depending on the shape of the input.
- JIT compilers like Numba can make your code 2x slower, even if the kernels are precompiled if they are short.

To read, jump to the `less_slow.py` source file and read the code snippets and comments.

## Reproducing the Benchmarks

If you are familiar with Python and want to review code and measurements as you read, you can clone the repository and execute the following commands to install the dependencies and run the benchmarks in your local environment.

```sh
git clone https://github.com/ashvardanian/less_slow.py.git # Clone the repository
cd less_slow.py                                            # Change the directory
pip install -r requirements.txt                            # Install the dependencies
pytest less_slow.py                                        # Run all benchmarks
pytest less_slow.py -x -k echo                             # Filter and stop on failure
```

Alternatively, run the benchmarks in a controlled environment using [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv venv --python 3.12                       # Or your preferred Python version
source .venv/bin/activate                   # To activate the virtual environment
uv run --python="3.12" \
 --with-requirements requirements.in \
    pytest -ra -q less_slow.py
```

For `pytest`, the `-r` flag can be used to display a "short test summary info" at the end of the test session, making it easy to get a clear picture of all failures in large test suites.
The `-ra` variant limits the summary only to failed tests, avoiding "passed" and "passed with outputs" messages.

For `uv`, the `--no-sync` flag prevents `uv` from creating a `uv.lock` file or modifying an existing `.venv` folder.
To extend the current list of dependencies, update the `requirements.in` file and run `uv sync` to update the environment.

```sh
uv pip compile requirements.in --universal --output-file requirements.txt
uv pip sync requirements.txt
```
