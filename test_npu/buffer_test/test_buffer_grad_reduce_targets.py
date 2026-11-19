# SPDX-License-Identifier: Apache-2.0
"""Canonical pointer-target correctness through the low-level launcher.

Same structure as ``test_buffer_grad_reduce.py`` (buffer-backed VMM chunks,
per-case SHMEM init, direct ``npu_patch.launchers`` call, same ``CASES`` /
``PLAN_BUILDERS`` / data generators / reference verification): the only
difference is the input — this variant hands the op a frozen INT64 ``[epn]``
table of ordinary FP32 ``[H, H']`` target addresses instead of an
``[epn, H, H']`` home-grad chunk, which is exactly what
``launch_grad_reduce_targets`` takes.

Run in fresh workers (EP2 or EP8; URMA requires Ascend950 + matching libraries):
    torchrun --nproc-per-node=2 -m pytest -s -x \
        test_npu/buffer_test/test_buffer_grad_reduce_targets.py -k mte
Replace -k mte with -k urma to select URMA, or omit it to run both transports.
"""
import pytest
import torch
import torch.distributed as dist
import torch_npu

from kernel_test_utils import init_shmem
from npu_patch import ShmemRuntime
from npu_patch._device import local_device_index
from npu_patch.buffer_c import (
    create_vmm_physical,
    reset_symmetric_descriptors,
    sym_tensor,
)
from npu_patch.launchers import launch_grad_reduce_targets
# Reuse the dense variant verbatim: same cases, same plans, same data
# generators and the same reference verification.
from test_buffer_grad_reduce import (
    CASES,
    PLAN_BUILDERS,
    _configure_udma_env,
    _make_data_fns,
    _verify,
)


import os

# Documented canonical UDMA scratch contract: 512*1024*max(2*min(blocks,64),R*B)
# FP32 elements per rank. It is already scratch, so the plan-size stress case
# (R*B = 16384..65536) would need 32..128 GiB and cannot be honoured on a 64 GiB
# die next to the case's own tensors; that case stays MTE-covered.
_UDMA_SCRATCH_LIMIT_BYTES = 16 * 1024 ** 3


def _udma_scratch_elements(blocks, R, B):
    """Setup-owned local FP32 scratch the canonical UDMA entry validates."""
    return 512 * 1024 * max(2 * min(blocks, 64), R * B)


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_buffer_grad_reduce_targets requires torchrun with at least 2 NPUs")

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
        pytest.skip("test_buffer_grad_reduce_targets requires at least 2 NPUs")

    yield rank, local_rank, R

    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()


def _launch_grad_reduce_targets(
    target_ptrs, reduce_buffers, plan, epn, H, Hp, R, B, rank, blocks,
    udma_workspace,
):
    """Direct canonical-pointer launcher (no Buffer wrapper).

    The dense test calls ``launch_grad_reduce`` the same way; this variant only
    swaps the home-grad chunk for the pointer table.
    """
    launch_grad_reduce_targets(
        target_ptrs, reduce_buffers, plan,
        rank=rank, num_sms=blocks, udma_workspace=udma_workspace,
    )


def run_case(rank, local_rank, R, case, use_udma):
    dev = f"npu:{local_device_index(local_rank)}"
    epn = case["epn"]
    E = R * epn
    H = case["H"]
    Hp = case["Hp"]
    # Buffer uses exact-size per-PE symmetric chunks; the slot bank is an
    # independent input (``base_B``), exactly as in the dense variant.
    B = case["base_B"]
    blocks = case["num_sms"]
    seed = case.get("seed", 0)

    scratch_elements = _udma_scratch_elements(blocks, R, B)
    if use_udma and scratch_elements * 4 > _UDMA_SCRATCH_LIMIT_BYTES:
        pytest.skip(
            f"{case['name']}: canonical UDMA scratch would need "
            f"{scratch_elements * 4 >> 30} GiB per rank "
            f"(contract 512*1024*max(2*min(blocks,64),R*B))")

    plan = PLAN_BUILDERS[case.get("plan", "ring")](R, B, epn, dev, seed)
    grads_fn, slot_fn = _make_data_fns(
        case.get("data", "affine"), rank, epn, B, H, Hp, dev, seed)

    target_table = None
    reduce_buffers_src = None
    udma_workspace = None
    try:
        # Canonical targets are ordinary FP32 memory: one contiguous
        # ``[epn, H, H']`` table sliced per expert, so the pointer table is
        # simply its rows and the reference verification can compare the table.
        target_table = torch.empty(
            (epn, H, Hp), dtype=torch.float32, device=dev)
        target_ptrs = torch.tensor(
            [target_table.data_ptr() + i * H * Hp * 4 for i in range(epn)],
            dtype=torch.int64, device=dev)
        # Only the B-slot source bank is symmetric.
        reduce_buffers_src, _, _ = create_vmm_physical(
            (B, H, Hp), torch.float32, local_device_index(local_rank))

        _configure_udma_env(use_udma)
        if not ShmemRuntime.is_initialized():
            if use_udma:
                init_shmem(group=None, enable_udma=True, qp_count=2)
            else:
                init_shmem(group=None)
        reduce_buffers = sym_tensor(reduce_buffers_src)
        if use_udma:
            udma_workspace = torch.empty(
                scratch_elements, dtype=torch.float32, device=dev)

        reduce_buffers.zero_()
        target_table.copy_(grads_fn())
        reduce_buffers.copy_(slot_fn(rank))
        torch.npu.synchronize()
        dist.barrier()

        _launch_grad_reduce_targets(
            target_ptrs, reduce_buffers, plan, epn, H, Hp, R, B, rank, blocks,
            udma_workspace,
        )
        torch.npu.synchronize()
        dist.barrier()

        _verify(rank, R, epn, B, plan, grads_fn, slot_fn,
                target_table, reduce_buffers,
                f"{case['name']} grad_reduce_targets failed")

        if rank == 0:
            print(f"[PASS] {case['name']} epn={epn} B={B} H={H} Hp={Hp} "
                  f"blocks={blocks} udma={use_udma}", flush=True)
    finally:
        # Re-init per case: finalize (tear down symmetric mappings) first, then
        # reverse the VMM allocation, then clear the descriptor registry.
        dist.barrier()
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        alloc = getattr(reduce_buffers_src, "_vmm_allocation", None)
        if alloc is not None:
            alloc.destroy()
        reset_symmetric_descriptors()


@pytest.mark.parametrize("use_udma", [True, False], ids=["urma", "mte"])
@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_buffer_grad_reduce_targets(dist_env, case, use_udma):
    rank, local_rank, R = dist_env
    if R not in (2, 8):
        pytest.skip("canonical pointer GradReduce recipe requires EP2 or EP8")
    run_case(rank, local_rank, R, case, use_udma)
