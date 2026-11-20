"""
Runtime backend detection for MoonEP.

Default is GPU (CUDA). The NPU backend is only engaged when an Ascend runtime
is actually present (``torch_npu`` importable and either ``ASCEND_HOME_PATH``
set or an NPU device reported available). ``MOONEP_BACKEND`` may force a
backend explicitly (``gpu``/``cuda``/``npu``/``ascend``).

GPU hosts never ``import torch_npu`` here.
"""

from __future__ import annotations

import os


def _env_backend() -> str | None:
    val = os.environ.get("MOONEP_BACKEND", "").strip().lower()
    if not val:
        return None
    if val in ("gpu", "cuda"):
        return "gpu"
    if val in ("npu", "ascend"):
        return "npu"
    raise ValueError(
        f"MOONEP_BACKEND must be one of gpu/cuda/npu/ascend, got {val!r}"
    )


def npu_enabled() -> bool:
    """True only when the NPU (Ascend) backend should drive moonep imports."""
    forced = _env_backend()
    if forced == "npu":
        return True
    if forced == "gpu":
        return False
    try:
        import torch_npu
    except Exception:
        return False
    if not os.environ.get("ASCEND_HOME_PATH"):
        return False
    try:
        return bool(torch_npu.is_available())
    except Exception:
        return True


def backend_name() -> str:
    return "npu" if npu_enabled() else "gpu"
