"""Multiple symmetric buffer allocations in one heap + cross-rank read/write validation.

Simulates the real MoonEP case: hidden_buf (token data) and meta_buf (metadata) are allocated
back to back in one symmetric heap; validates that:
  1. one symmetric heap supports several create_vmm_physical calls (independent VMM chunk each)
  2. every block builds its own cross-rank view (its shape/stride is correct)
  3. cross-rank read/write is consistent for every block (peer RMA, not the acl_dist_map contiguous view)
  4. every chunk is registered in the buffer-backed init descriptor

Run (buffer-backed, UID broadcast over dist):
    torchrun --nproc_per_node=8 --master_port=29503 \
        test_npu/buffer_test/test_buffer_c_api.py
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

# Shapes of the hidden / meta chunks (the order must be identical on all ranks -- symmetric init is collective).
HIDDEN_SHAPE = [32, 64]
META_SHAPE = [128]


def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = local_device_index(local_rank)
    torch.npu.set_device(device)
    print(f"[rank{rank}/{world}] ready")

    # ------------------------------------------------------------------
    # buffer-backed only: chunks are allocated and registered first (create_vmm_physical),
    # init_with_buffer collects the global registry to build the symmetric heap, then sym_tensor yields symmetric tensors.
    # ------------------------------------------------------------------
    # ① hidden_buf: token data [NvS, H] bf16 (NvS = number of slots, H = hidden)
    hidden_src, _, _ = create_vmm_physical(HIDDEN_SHAPE, torch.bfloat16, device)
    # ② meta_buf: metadata [meta_chunk] int32 (one-dimensional)
    meta_src, _, _ = create_vmm_physical(META_SHAPE, torch.int32, device)
    if not ShmemRuntime.is_initialized():
        init_shmem(group=None)
    hidden_buf = sym_tensor(hidden_src)
    meta_buf = sym_tensor(meta_src)

    # The two chunks must have different symmetric base addresses (separately allocated chunks).
    assert hidden_buf.data_ptr() != meta_buf.data_ptr(), \
        "hidden and meta must be separate chunks"

    print(f"[rank{rank}] hidden: shape={list(hidden_buf.shape)} "
          f"stride={list(hidden_buf.stride())} ptr={hidden_buf.data_ptr():#x}")
    print(f"[rank{rank}] meta  : shape={list(meta_buf.shape)} "
          f"stride={list(meta_buf.stride())} ptr={meta_buf.data_ptr():#x}")
    assert tuple(hidden_buf.shape) == tuple(HIDDEN_SHAPE)
    assert tuple(meta_buf.shape) == tuple(META_SHAPE)

    # ------------------------------------------------------------------
    # Cross-rank write: each rank writes its rank value into its own chunk.
    # ------------------------------------------------------------------
    hidden_buf.fill_(rank)
    meta_buf.fill_(rank)
    torch.npu.synchronize()
    dist.barrier()

    # ------------------------------------------------------------------
    # Cross-rank read: peer RMA pulls the peer chunk locally and validates it (equivalent to the old
    # acl_dist_map full[peer] cross-rank read, but without the legacy 1GB-stride contiguous view).
    # ------------------------------------------------------------------
    stream = torch_npu.npu.current_stream().npu_stream

    def _check_peer(chunk, chunk_shape, dtype, label):
        recv = torch.empty(chunk_shape, dtype=dtype, device=f"npu:{device}")
        for peer in range(world):
            if peer == rank:
                continue
            sym_get_into(chunk, 0, recv, peer, stream)
            torch.npu.synchronize()
            if not bool(torch.all(recv == peer).item()):
                return False, peer
        return True, None

    ok_h, bad_h = _check_peer(hidden_buf, HIDDEN_SHAPE, torch.bfloat16, "hidden")
    ok_m, bad_m = _check_peer(meta_buf, META_SHAPE, torch.int32, "meta")
    print(f"[rank{rank}] hidden cross-rank R/W {'ALL OK' if ok_h else 'FAIL'}")
    print(f"[rank{rank}] meta   cross-rank R/W {'ALL OK' if ok_m else 'FAIL'}")
    assert ok_h, f"hidden cross-rank R/W mismatch on peer {bad_h}"
    assert ok_m, f"meta cross-rank R/W mismatch on peer {bad_m}"

    del hidden_buf, meta_buf
    torch.npu.synchronize()

    # Unified release order: finalize (tears down the symmetric mapping) → destroy VMM allocations → clear the registry.
    finalize_shmem_and_release(source_tensors=(hidden_src, meta_src))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
