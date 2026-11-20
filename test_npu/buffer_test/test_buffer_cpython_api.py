"""Validate the C API wrappers of buffer_c.py.

Covers:
  1. init_shmem (single node defaults to ``ShmemRuntime.init_with_buffer``, buffer-backed UID
     bootstrap)
  2. create_vmm_physical + sym_tensor (drop-in for create_sym_dist_tensor: single chunk allocation)
  3. peer RMA cross-rank view (drop-in for the acl_dist_map contiguous view,
     no legacy 1GB-stride)
  4. cross-rank read/write consistency

Run:
    torchrun --nproc_per_node=8 --master_port=29507 test_npu/buffer_test/test_buffer_cpython_api.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import finalize_shmem_and_release, init_shmem
import torch
import torch.distributed as dist
import torch_npu

from npu_patch import ShmemRuntime
from npu_patch._common import sym_get_into, sym_put_from
from npu_patch._device import local_device_index
from npu_patch.buffer_c import (
    create_vmm_physical,
    sym_tensor,
)


def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = local_device_index(local_rank)
    torch.npu.set_device(device)
    print(f"[rank{rank}/{world}] ready")

    # ① allocate and register the chunk first → init_with_buffer (buffer-backed UID bootstrap)
    chunk_shape = [16, 32]
    local_src, _, _ = create_vmm_physical(chunk_shape, torch.float32, device)
    if not ShmemRuntime.is_initialized():
        init_shmem(group=None)
    local = sym_tensor(local_src)  # local symmetric chunk, shape=chunk_shape
    print(f"[rank{rank}] local shape={list(local.shape)} "
          f"ptr={local.data_ptr():#x}")
    assert list(local.shape) == chunk_shape

    # ③ Cross-rank write: each rank writes its rank value into its own chunk.
    local.copy_(torch.full(chunk_shape, float(rank), device=local.device))
    torch.npu.synchronize()
    dist.barrier()

    # ④ Cross-rank read: peer RMA (sym_get_into) pulls the peer chunk locally and validates it, equivalent
    #    to the full[peer] cross-rank read of the acl_dist_map contiguous view.
    stream = torch_npu.npu.current_stream().npu_stream
    recv = torch.empty(chunk_shape, dtype=torch.float32, device=local.device)
    ok = True
    bad_peer = None
    for peer in range(world):
        if peer == rank:
            continue
        sym_get_into(local, 0, recv, peer, stream)
        torch.npu.synchronize()
        if not bool(torch.all(recv == float(peer)).item()):
            ok, bad_peer = False, peer
            break
    print(f"[rank{rank}] cross-rank R/W {'ALL OK' if ok else 'FAIL'}")
    assert ok, f"cross-rank R/W mismatch on peer {bad_peer}"

    # Release the symmetric tensors before finalize.
    del local, recv
    torch.npu.synchronize()

    # Unified release order: finalize (tears down the symmetric mapping) → destroy VMM allocations → clear the registry.
    finalize_shmem_and_release(source_tensors=(local_src,))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
