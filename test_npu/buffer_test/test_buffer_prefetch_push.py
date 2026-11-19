# SPDX-License-Identifier: Apache-2.0
"""Canonical pointer PUSH correctness through the low-level launcher.

Same structure as ``test_buffer_prefetch.py`` (per-rank ``[epn + B, H, H']``
buffer-backed chunk, per-case SHMEM init, ``map_to_sym_ptr`` translation, same
``CASES`` / per-expert payload generator / per-slot reference check): the only
difference is the input — this variant hands the op a frozen INT64 ``[epn]``
table of ordinary BF16 source addresses plus the whole ``[R, B]`` plan, which is
exactly what ``launch_prefetch_from_ptrs`` takes. The GET variant instead passes
the ``[E, H, H']`` symmetric home view and this rank's ``[B]`` request row.

PUSH semantics (see ``npu_patch/native/kernels/prefetch_pointer_push_mte.cpp``):
the plan is row-major ``[destination PE, destination slot]``; the PE that owns
the listed expert pushes its own row into that slot. Replicating one request row
across all destinations therefore reproduces the GET expectations exactly — every
rank ends up with the same expert set in its own slots, pulled by the owner
instead of fetched by the consumer.

Run in fresh workers (EP2 or EP8; URMA requires Ascend950 + matching libraries):
    torchrun --nproc-per-node=2 -m pytest -s -x \
        test_npu/buffer_test/test_buffer_prefetch_push.py -k mte
Replace -k mte with -k urma to select URMA, or omit it to run both transports.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem

import pytest
import torch
import torch.distributed as dist
import torch_npu

from npu_patch._device import local_device_index
from npu_patch import ShmemRuntime
from npu_patch.buffer_c import (
    create_sym_tensor_from_ptr,
    map_to_sym_ptr,
    reset_symmetric_descriptors,
)
from npu_patch.launchers import launch_prefetch_from_ptrs
# Reuse the GET variant verbatim: same cases, same chunk layout, same per-expert
# payload generator and the same transport toggles.
from test_buffer_prefetch import (
    CASES,
    _configure_udma_env,
    _expert_value,
    make_symmetric_experts,
)


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_buffer_prefetch_push requires torchrun with at least 2 NPUs")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(
            backend="hccl", device_id=torch.device(f"npu:{local_device_index(local_rank)}"))
    rank = dist.get_rank()
    R = dist.get_world_size()
    if R < 2:
        if owns_process_group:
            dist.destroy_process_group()
        pytest.skip("test_buffer_prefetch_push requires at least 2 NPUs")

    yield rank, local_rank, R

    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()


def push_plan(expert_ids, R, device):
    """Replicated ``[destination PE, destination slot]`` table, independent of the
    token planner's load balance.

    The PUSH kernel walks the plan by ``destination_pe = entry / B`` and lets the
    owner of ``plan[destination_pe, slot]`` write into that destination's slot,
    so repeating one request row on every destination gives every rank the same
    per-slot expectations as the GET variant.
    """
    return torch.tensor([list(expert_ids)] * R, dtype=torch.int32, device=device)


def run_case(rank, local_rank, R, case, use_udma):
    E = case["E"]
    H = case["H"]
    Hp = case["Hp"]
    B = case["B"]
    device = f"npu:{local_device_index(local_rank)}"

    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['name']}: H/Hp must be multiples of 128"

    keepalive = make_symmetric_experts(R, E, H, Hp, B, local_rank)
    allocation = getattr(keepalive, "_vmm_allocation", None)
    epn = (E + R - 1) // R
    row_bytes = H * Hp * 2
    try:
        local_expert = keepalive[:epn]
        push_slots_src = keepalive[epn:]
        local_expert.zero_()
        for expert in range(rank * epn, min((rank + 1) * epn, E)):
            local_expert[expert - rank * epn].copy_(
                _expert_value(expert, H, Hp, device)
            )

        # Setup-frozen canonical sources: only this rank's own epn rows, as
        # ordinary addresses. The kernel adds the destination offset itself.
        source_ptrs = torch.tensor(
            [local_expert.data_ptr() + index * row_bytes for index in range(epn)],
            dtype=torch.int64, device=device,
        )

        expert_ids = case["experts"](E)
        assert len(expert_ids) == B
        sentinel = -123.0
        push_slots_src.fill_(sentinel)
        plan = push_plan(expert_ids, R, device)

        torch.npu.synchronize()
        dist.barrier()

        # buffer-backed only: chunk already registered -> init_shmem (auto-picks the UID/IP-port bootstrap)
        # -> map_to_sym_ptr translates it into a local symmetric tensor (data_ptr = symmetric base) before
        # the kernel writes into it with aclshmem RMA.
        _configure_udma_env(use_udma)
        if not ShmemRuntime.is_initialized():
            if use_udma:
                init_shmem(group=None, enable_udma=True, qp_count=2)
            else:
                init_shmem(group=None)

        symmetric_base = map_to_sym_ptr(
            keepalive.data_ptr(), "prefetch_push_chunk"
        )
        push_slots = create_sym_tensor_from_ptr(
            symmetric_base + epn * row_bytes,
            [B, H, Hp],
            torch.bfloat16,
            device,
        )

        launch_prefetch_from_ptrs(source_ptrs, push_slots, plan, epn)
        torch.npu.synchronize()

        ok = True
        for slot, expert in enumerate(expert_ids):
            expected = (
                torch.full_like(push_slots[slot], sentinel)
                if expert < 0
                else _expert_value(expert, H, Hp, device)
            )
            if not torch.equal(push_slots[slot], expected):
                ok = False
                max_diff = (
                    push_slots[slot].float() - expected.float()
                ).abs().max().item()
                print(
                    f"[rank {rank}] {case['name']} mismatch: buffer={slot}, "
                    f"expert={expert}, max_abs_diff={max_diff}"
                )

        ok_tensor = torch.tensor(
            [int(ok)], dtype=torch.int32, device=device
        )
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        assert ok_tensor.item() == 1, f"{case['name']} failed"
        if rank == 0:
            print(
                f"  [PASS] {case['name']} ({'urma' if use_udma else 'mte'}): "
                f"E={E}, B={B}, H={H}, Hp={Hp}, epn={epn}"
            )
        dist.barrier()
    finally:
        # Re-init per case: finalize first (tear down the symmetric mappings), then the VMM undo
        # ops, then clear the registry. Reversed, the next init's get_buffer_ptr would not see the new chunk.
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        if allocation is not None:
            allocation.destroy()
        reset_symmetric_descriptors()
        torch.npu.synchronize()


@pytest.mark.parametrize("use_udma", [True, False], ids=["urma", "mte"])
@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_buffer_prefetch_push(dist_env, case, use_udma):
    rank, local_rank, R = dist_env
    if R not in (2, 8):
        pytest.skip("pointer PUSH recipe requires EP2 or EP8")
    run_case(rank, local_rank, R, case, use_udma)
