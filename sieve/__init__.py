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
