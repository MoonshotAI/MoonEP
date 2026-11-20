from . import _backend as _backend

if _backend.npu_enabled():
    # NPU backend: expose the vendored Ascend implementation directly. The
    # CUDA/cutlass GPU kernel stack (moonep.api, moonep.planning,
    # dispatch/combine/prefetch/...) is deliberately NOT imported here: those
    # modules import `cuda.bindings.driver` / `cutlass`, which are not installed
    # on an NPU-only host and are never used by the NPU backend.
    # `moonep.buffer` re-exports its NPU-only helpers itself.
    from npu_patch.api import Buffer as Buffer
    from npu_patch.planning import MoonEPCommPlan as MoonEPCommPlan
    from npu_patch.shmem_runtime import ShmemRuntime as ShmemRuntime
else:
    from .api import Buffer, MoonEPCommPlan as MoonEPCommPlan


def init_shmem(group=None, **kwargs):
    """Initialize the shmem symmetric heap on NPU (no-op on GPU).

    Required on Ascend before the first dispatch/combine/prefetch/reduce_grad,
    after every ``Buffer`` has been constructed. On GPU this is a no-op so
    single-file test/benchmark code can call it unconditionally.
    """
    if not _backend.npu_enabled():
        return
    from npu_patch.shmem_runtime import ShmemRuntime
    ShmemRuntime.init_with_buffer(group=group, **kwargs)


__all__ = [
    "Buffer",
    "MoonEPCommPlan",
    "init_shmem",
]
