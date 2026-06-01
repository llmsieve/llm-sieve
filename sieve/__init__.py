"""Sieve package init.

Sets ONNX Runtime / OpenMP environment variables before any submodule
imports ``fastembed`` (and therefore ``onnxruntime``). In sandboxed or
containerised environments onnxruntime's CPU-affinity probe logs noisy
``pthread_setaffinity_np failed`` warnings; disabling the probe and
pinning a small thread pool keeps the first-run output clean without
measurably affecting throughput.

Use ``setdefault`` so an operator who has already tuned these vars keeps
their setting.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("ONNXRUNTIME_DISABLE_CPU_AFFINITY", "1")


# Expose package version for introspection. Reads from installed package
# metadata so it always matches what pip / pipx / uv shipped, regardless
# of how the package was installed.
def _resolve_version() -> str:
    try:
        from importlib.metadata import version
        return version("llm-sieve")
    except Exception:
        return "0.0.0+unknown"


__version__ = _resolve_version()
__all__ = ["__version__"]
