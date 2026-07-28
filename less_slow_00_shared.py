#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dependency probes shared by every chapter.

`pytest.mark.skipif` is evaluated at import time, not at call time, so these
flags cannot be fixtures — each chapter imports the ones it needs directly.

Only CUDA is genuinely optional: `nvmath` drags in roughly 2.9 GB of CUDA
libraries, while every other dependency here is smaller than pandas. Its
names are bound to `None` when absent, so a machine without a GPU gets skips
rather than a collection error. The `pyarrow` and `numba` probes are legacy
tolerance for partial installs; both are required in `pyproject.toml`.
"""

import ml_dtypes  # NumPy-native fp8 and bfloat16 dtypes
import numkong as nk
import numpy as np

# ? Bound unconditionally so chapters can import them and let `skipif` gate.
nvmath = cublas_matmul = MatmulQuantizationScales = None

pandas_installed = True
try:
    import pyarrow as pa  # noqa: E402
    import pyarrow.compute as pc  # noqa: E402

    pyarrow_installed = True
except ImportError:
    pyarrow_installed = False


numba_installed = False
try:
    import numba

    numba_installed = True
except ImportError:
    pass  # skip if numba is not installed

# ? Importing nvmath is not enough — it succeeds on machines with no GPU, and
# ? even on machines whose driver predates the bundled CUDA runtime. The guard
# ? has to attempt real work, so we defer that to the first benchmark.
nvmath_installed = False
try:
    import nvmath
    from nvmath.linalg.advanced import matmul as cublas_matmul
    from nvmath.linalg.advanced import MatmulQuantizationScales

    nvmath_installed = True
except ImportError:
    pass


def _cublas_usable() -> bool:
    """Probe an actual matmul: import success does not imply a working GPU."""
    if not nvmath_installed:
        return False
    try:
        probe = np.ones((16, 16), dtype=np.float32)
        cublas_matmul(probe, np.asfortranarray(probe))
        return True
    except Exception:
        return False


gpu_matmul_usable = _cublas_usable()
