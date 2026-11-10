"""
KernelOp base: the ascend_deepep "op owns its symmetric workspace" pattern.

Mirrors ``ShmemWorkspaceOp`` in ``ascend_deepep/csrc/torch_classes.cpp``: the
op allocates its own workspace at construction (Buffer init time) and releases
it at destroy. Both are collective operations, so ops must be created and
destroyed in the same order on every rank.

Buffer-backed path only: the workspace is CANN VMM physical memory
(``create_vmm_physical``, registered with the buffer-backed shmem init); its
symmetric address is resolved lazily via ``map_to_sym_ptr`` in
``sym_workspace()``. Teardown reverses the VMM steps
(unmap -> release -> free_physical).
"""

import torch

from ._common import SymBuffer
from .buffer_c import create_vmm_physical, map_to_sym_ptr


class KernelOp:
    """Base for npu_patch kernel operators.

    The migrated AscendC kernels consume the local symmetric bases from ``ctx``
    (the communication chunks — ``hidden_buf`` / ``weights_buf`` /
    ``barrier_buf`` — addressed via their shmem addresses) plus this op's
    symmetric workspace; they must never assume a cross-rank contiguous VA.
    """

    def __init__(self, ws_bytes: int = 0, num_sms: int = 32):
        # No 2 MiB alignment requirement on Ascend: any positive size is OK
        # (VMM granularity alignment is handled by create_vmm_physical).
        self._num_sms = num_sms
        self._ws_bytes = ws_bytes
        self._ws_tensor = None
        self._ws = None
        if ws_bytes > 0:
            # VMM physical allocation; symmetric address resolved lazily in sym_workspace().
            self._ws_tensor, _, _ = create_vmm_physical(
                [ws_bytes], torch.uint8, torch.npu.current_device())
            self._ws = None
        # Collective: all ranks allocate the same workspaces in the same order.
        self._destroyed = False

    @property
    def num_sms(self) -> int:
        return self._num_sms

    def sym_workspace(self) -> SymBuffer | None:
        """The op-owned symmetric workspace (device-RMA scratch), if any.

        Returns the translated local shmem symmetric address (requires the
        buffer-backed shmem heap initialized and this op's workspace registered).
        """
        if self._ws_tensor is None:
            return None
        if self._ws is None:
            sym = map_to_sym_ptr(self._ws_tensor.data_ptr(), 'op_workspace')
            self._ws = SymBuffer(sym, self._ws_bytes)
        return self._ws

    def destroy(self) -> None:
        """Collective: all ranks must destroy ops in the same order."""
        if not self._destroyed:
            if self._ws_tensor is not None:
                allocation = getattr(self._ws_tensor, '_vmm_allocation', None)
                if allocation is not None:
                    # Reverse VMM steps (unmap -> release -> free_physical).
                    allocation.destroy()
                self._ws_tensor = None
                self._ws = None
            self._destroyed = True
