# _Less Slow_ Python

> The spiritual little brother of [`less_slow.cpp`](https://github.com/ashvardanian/less_slow.cpp).
> Assuming Python is used in a different setting than C++, this repository focuses more on scripting, tool integration, and data processing.
> The benchmarks in this repository don't aim to cover every topic entirely, but they help form a mindset and intuition for performance-oriented software design.

Much of modern code suffers from common pitfalls: bugs, security vulnerabilities, and performance bottlenecks. University curricula often teach outdated concepts, while bootcamps oversimplify crucial software development principles.

![Less Slow Python](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/less_slow.py.jpg?raw=true)

This repository offers practical examples of writing efficient Python code.
The topics range from basic micro-kernels executing in a few nanoseconds to more complex constructs involving parallel algorithms, coroutines, and polymorphism. Some of the highlights include:

- Using callbacks, lambdas, and `yield`-ing functions are much faster than iterator-based routines, unlike Rust and C++.
- Not every all composite structures are equally fast: `namedtuple` is slower than { `dataclass`, `class` } is slower than `dict`.
- Error handling with status codes can be both 50% faster and 2x slower than exceptions, depending on your design.
- NumPy-based logic can be much slower than `math` functions depending on the shape of the input.
- JIT compilers like Numba can make your code 2x slower, even if the kernels are precompiled if they are short.

To read, jump to the `less_slow.py` source file and read the code snippets and comments.

## Reproducing the Benchmarks

If you are familiar with Python and want to review code and measurements as you read, you can clone the repository and execute the following commands to install the dependencies and run the benchmarks in your local environment.

```sh
git clone https://github.com/ashvardanian/less_slow.py.git # Clone the repository
cd less_slow.py                                            # Change the directory
pip install -r requirements.txt                            # Install the dependencies
pytest less_slow.py                                        # Run the benchmarks
```

Alternatively, consider using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to run the benchmarks in a controlled environment.

```sh
uv run --python="3.12" --no-sync --with-requirements requirements.in \
pytest -ra -q less_slow.py
```

The `--no-sync` flag is used to prevent `uv` from creating a `uv.lock` file or modifying an existing `.venv` folder.
To extend the current list of dependencies, update the `requirements.in` file and run `uv sync` to update the environment.

```sh
uv pip compile requirements.in --universal --output-file requirements.txt
uv pip sync requirements.txt
```
