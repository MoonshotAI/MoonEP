"""
npu_patch: Ascend (NPU) MoE expert-parallel communication runtime.

Two usage paths share this package:

1. **Call-chain ops** (C++ extension ``_C``): ``ShmemInstance`` + the
   ``Dispatch`` / ``Combine`` torch custom classes
   (ascend_deepep style). Business parameters (hidden/topk/expert shapes) are
   passed on every ``compute`` call.

2. **MoonEP-style Buffer API** (Python, shmem symmetric memory): ``Buffer``
   pre-allocates the symmetric communication chunks and instantiates the kernel
   op objects at init; ``dispatch`` / ``prefetch_weight`` / ``combine`` /
   ``reduce_grad`` are the migrated ``moonep/api.py`` entry points. Runtime
   lifecycle is handled by ``ShmemRuntime`` (shmem Python package UID
   bootstrap), distinct from the C++ ``ShmemInstance``.

When both paths are used in one process, the C++ extension reuses the SHMEM
runtime loaded by the Python package.  A process must never load two independent
``libshmem.so`` instances because their PE state and symmetric heaps diverge.
"""

import ctypes
import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def _load_package_shmem_libs():
    """Load the CANN SHMEM shared libraries shipped in ``lib/``.

    The C++ ``_C`` extension links against ``libshmem.so`` (rpath
    ``$ORIGIN/lib``); loading the copies here with RTLD_GLOBAL makes them
    resolvable before ``_C`` is imported.

    This fallback is only needed when the ``shmem`` Python package is not
    available.  If Python SHMEM was loaded first, ``_C`` must reuse that runtime
    instead of loading another absolute-path copy from ``lib/``.
    """
    lib_dir = _PKG_DIR / "lib"
    required_libs = [
        "libshmem_utils.so",
        "aclshmem_bootstrap_config_store.so",
        "libshmem.so",
    ]
    optional_libs = [
        "aclshmem_bootstrap_uid.so",
    ]
    mode = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in required_libs + optional_libs:
        lib_path = lib_dir / lib_name
        if not lib_path.exists():
            if lib_name in required_libs:
                raise ImportError(f"required SHMEM shared library not found: {lib_path}")
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=mode)
        except OSError as exc:
            raise ImportError(f"failed to load SHMEM shared library {lib_path}: {exc}") from exc


def _promote_python_shmem_libs():
    from shmem.shmem_config import _get_lib_dir

    lib_dir = Path(_get_lib_dir())
    mode = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in (
            "libshmem_utils.so",
            "aclshmem_bootstrap_config_store.so",
            "libshmem.so",
            "aclshmem_bootstrap_uid.so",
    ):
        lib_path = lib_dir / lib_name
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=mode)


def _has_cpp_extension() -> bool:
    """True when the C++ ``_C`` extension has been built into the package."""
    return any(_PKG_DIR.glob("_C*.so")) or any(_PKG_DIR.glob("_C*.pyd"))


# The Buffer API (npu_patch.api) is built on the `shmem` Python package.
# Force its native libraries to load before the C++ `lib/` copies so the
# package's duplicate-libshmem guard sees a single instance. Pure C++ op users
# without the shmem Python package are unaffected (failure swallowed; this is
# only an ordering optimization).
_PYTHON_SHMEM_LOADED = False
try:
    import shmem as _shmem

    _ = _shmem.aclshmem_get_unique_id  # force the package's lazy native load
    del _
    _promote_python_shmem_libs()
    _PYTHON_SHMEM_LOADED = True
except Exception:
    _shmem = None

if _has_cpp_extension():
    if not _PYTHON_SHMEM_LOADED:
        _load_package_shmem_libs()
    # Load torch/torch_npu first: _C.so's RUNPATH lacks the torch package lib
    # dir, so libc10/libtorch_cpu/libtorch_npu must be resolvable before _C is
    # linked (otherwise a bare ``import npu_patch`` fails with
    # "libtorch_npu.so: cannot open shared object").
    import torch
    import torch_npu

    from . import _C as _C

    ShmemInstance = torch.classes.AscendMoonEP.ShmemInstance
    GradReduce = torch.classes.AscendMoonEP.GradReduce
    PrefetchWeight = torch.classes.AscendMoonEP.PrefetchWeight
else:
    # The C++ extension is not built; the MoonEP-style Buffer API (Python,
    # shmem symmetric memory) still works. The call-chain classes are left as
    # None so callers can detect the missing extension.
    _C = None
    ShmemInstance = None
    GradReduce = None
    PrefetchWeight = None


def _load_buffer_api():
    """Import the MoonEP-style Buffer API modules on first access.

    The import is deferred so that C++-ops-only users (who may not have the
    ``shmem`` Python package / ``torch_npu`` installed) can import this package
    without pulling in the buffer API stack.
    """
    from .shmem_runtime import ShmemRuntime, runtime_library_info
    from .planning import MoonEPCommPlan
    from .api import Buffer

    return {
        "Buffer": Buffer,
        "MoonEPCommPlan": MoonEPCommPlan,
        "ShmemRuntime": ShmemRuntime,
        "runtime_library_info": runtime_library_info,
    }


_BUFFER_API_NAMES = frozenset({
    "Buffer", "MoonEPCommPlan", "ShmemRuntime", "runtime_library_info"
})
_BUF_API_CACHE = None


def __getattr__(name):
    if name in _BUFFER_API_NAMES:
        global _BUF_API_CACHE
        if _BUF_API_CACHE is None:
            _BUF_API_CACHE = _load_buffer_api()
        return _BUF_API_CACHE[name]
    raise AttributeError(f"module 'npu_patch' has no attribute {name!r}")


from .launchers import launch_grad_reduce, launch_prefetch

__all__ = [
    "ShmemInstance",
    "GradReduce",
    "launch_grad_reduce",
    "Buffer",
    "MoonEPCommPlan",
    "ShmemRuntime",
    "runtime_library_info",
    "PrefetchWeight",
    "launch_prefetch",
]
