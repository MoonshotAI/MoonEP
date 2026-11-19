"""Storage parameters for the native Buffer API, without a scheduling protocol.

Allocate/register symmetric sources before SHMEM init; prepare these bindings
after init and before hot work. The caller still owns final-use/reuse ordering.
Bindings carry no Buffer identity; callers must select storage compatible with
the Buffer's layout and device.
"""

from dataclasses import dataclass

import torch

from .buffer_c import sym_tensor
from .shmem_runtime import ShmemRuntime


def tensor_signature(tensor):
    """Metadata only: detect rebinding without reading device data/versions."""
    storage = tensor.untyped_storage()
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device,
            storage._cdata, storage.nbytes())


def symmetric_source(source):
    if not ShmemRuntime.is_initialized():
        raise RuntimeError("initialize buffer-backed SHMEM before preparing bindings")
    if getattr(ShmemRuntime, '_poisoned', False):
        raise RuntimeError("binding preparation requires healthy SHMEM")
    if getattr(source, '_vmm_allocation', None) is None:
        raise ValueError("expected a registered VMM source tensor")
    return sym_tensor(source)


def overlaps(left, right):
    if left.device != right.device:
        return False
    a, b = left.data_ptr(), right.data_ptr()
    return (a < b + right.numel() * right.element_size() and
            b < a + left.numel() * left.element_size())


@dataclass(frozen=True, eq=False, repr=False)
class CommBuffers:
    """Prepared hidden/route views. Obtain with Buffer.prepare_comm_buffers.

    No allocation, lease, generation or completion state is owned by this value.
    Backing references keep the original source metadata/physical owners alive;
    callers must keep the tensors' storage and layout fixed while bound.
    """

    hidden: torch.Tensor
    route: torch.Tensor
    sources: tuple


@dataclass(frozen=True, eq=False, repr=False)
class ExpertPointerBinding:
    """Prepared expert addresses and one symmetric communication buffer.

    Weight prefetch reads expert_tensors; gradient reduction writes them.
    The caller keeps addresses fixed and orders final consumers before reuse.
    """

    expert_tensors: tuple
    expert_views: tuple
    comm_buffer: torch.Tensor
    comm_buffer_source: torch.Tensor
    expert_ptrs: torch.Tensor
    tensor_signatures: tuple

    def validate_expert_tensors(self, expert_tensors):
        """Reject replaced expert tensors or changed storage/layout."""
        validate_unchanged(tuple(expert_tensors), self.tensor_signatures)


def validate_unchanged(refs, signatures):
    if tuple(tensor_signature(ref) for ref in refs) != signatures:
        raise ValueError("prepared storage was rebound or its layout changed")


def comm_context(ctx, buffers):
    """Select the symmetric data banks for one call, sharing control and workspace.

    Each EP rank must select the corresponding banks for the collective.
    The caller orders submissions that share the workspace.
    """
    if not ShmemRuntime.is_initialized() or getattr(ShmemRuntime, '_poisoned', False):
        raise RuntimeError('explicit communication storage requires initialized healthy SHMEM')
    if not isinstance(buffers, CommBuffers):
        raise TypeError("buffers must be CommBuffers")
    local = dict(ctx)
    local['hidden_buf'] = local['hidden_buf_local'] = buffers.hidden
    local['weights_buf'] = local['weights_buf_local'] = buffers.route
    return local
