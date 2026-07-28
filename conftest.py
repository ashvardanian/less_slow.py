#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session fixtures, auto-loaded by pytest.

Chapters do not import this file; pytest finds it. Its job is the banner
printed once per session, which reports the two things every measurement in
this suite depends on and neither of which is inferable from the numbers: the
CPU and, separately, whether this is a free-threaded build and whether the
GIL is actually disabled right now. `PYTHON_GIL=1` changes the second without
changing the first, and the parallelism chapter needs both to be legible.
"""

import multiprocessing
import platform
import sys
import sysconfig

import numpy as np
import pandas as pd
import pytest

from less_slow_00_shared import (
    gpu_matmul_usable,
    numba_installed,
    nvmath_installed,
    pandas_installed,
    pyarrow_installed,
)

if pyarrow_installed:
    import pyarrow as pa
if numba_installed:
    import numba
import numkong as nk

if nvmath_installed:
    import nvmath


@pytest.fixture(scope="session", autouse=True)
def print_environment_info():
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    cores = multiprocessing.cpu_count()
    py_impl = platform.python_implementation()
    py_ver = platform.python_version()
    runtime = sys.executable

    # ? Two different things, and the parallelism chapter needs both: whether
    # ? this is a free-threaded *build*, and whether the GIL is actually off
    # ? right now — `PYTHON_GIL=1` turns it back on without changing the build.
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    gil_state = "enabled" if gil_probe is None or gil_probe() else "disabled"
    build = "free-threaded" if free_threaded else "standard"

    lines = [
        f"Env: {system} {release} | {machine}",
        f"Cores: {cores} | start method: {multiprocessing.get_start_method()}",
        f"Python: {py_impl} {py_ver} {build} | GIL {gil_state}",
        f"Runtime: {runtime}",
        f"NumPy: {np.__version__}",
    ]

    if pandas_installed:
        lines.append(f"Pandas: {pd.__version__}")
    if pyarrow_installed:
        lines.append(f"PyArrow: {pa.__version__}")
    if numba_installed:
        lines.append(f"Numba: {numba.__version__}")
    lines.append(f"NumKong: {nk.__version__}")
    if nvmath_installed:
        state = "GPU matmul ready" if gpu_matmul_usable else "no usable GPU"
        lines.append(f"nvmath: {nvmath.__version__} | {state}")

    print("\n".join(lines))
