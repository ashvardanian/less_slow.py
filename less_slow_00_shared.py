#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional-dependency probes, shared by every chapter.

`pytest.mark.skipif` is evaluated at import time, not at call time, so these
flags cannot be fixtures — each chapter imports the ones it needs directly.

Every optional name is bound unconditionally, to `None` when the import
fails. Without that a missing dependency turns into a collection error rather
than the skip it should be, which is exactly the failure this file exists to
prevent.
"""

import numpy as np

# ? Every optional name is bound unconditionally, so chapters can import it and
# ? let the `skipif` flags do the gating. Without this a missing dependency
# ? turns into a collection error instead of a skip.
pa = pc = numba = nk = ml_dtypes = nvmath = None
cublas_matmul = MatmulQuantizationScales = None

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

numkong_installed = False
try:
    import numkong as nk

    numkong_installed = True
except ImportError:
    pass  # skip if numkong is not installed

ml_dtypes_installed = False
try:
    import ml_dtypes  # provides NumPy-native fp8 and bfloat16 dtypes

    ml_dtypes_installed = True
except ImportError:
    pass

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
    if not (nvmath_installed and ml_dtypes_installed):
        return False
    try:
        probe = np.ones((16, 16), dtype=np.float32)
        cublas_matmul(probe, np.asfortranarray(probe))
        return True
    except Exception:
        return False


gpu_matmul_usable = _cublas_usable()
