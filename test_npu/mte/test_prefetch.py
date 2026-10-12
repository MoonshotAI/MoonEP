"""
Remote expert prefetch correctness test using ACLSHMEM symmetric memory.

Run with:
    ASCEND_RT_VISIBLE_DEVICES=8,9 \
    torchrun --nproc-per-node=8 -m pytest -s test_npu/mte/test_prefetch.py

The test cases and validation flow mirror MoonEP/tests/test_prefetch.py.
The Ascend implementation uses expert_id // (E / world_size) as the remote
owner, so the CUDA-only owner_offset axis is not needed here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import os

import pytest
import torch
import torch.distributed as dist
import torch_npu

from npu_patch._device import local_device_index
from npu_patch import ShmemRuntime, launch_prefetch
from npu_patch.buffer_c import (
    create_sym_tensor_from_ptr,
    create_vmm_physical,
    map_to_sym_ptr,
    reset_symmetric_descriptors,
)


def _random_experts(E, B, seed):
    """Random plan with holes and possible duplicates."""
    g = torch.Generator()
    g.manual_seed(seed)
    live = torch.rand(B, generator=g) < 0.6
    ids = torch.randint(0, E, (B,), generator=g, dtype=torch.int32)
    return torch.where(live, ids, torch.full_like(ids, -1)).tolist()


CASES = [
    {
        "name": "single_tile_one_buffer",
        "E": 4,
        "H": 128,
        "Hp": 128,
        "B": 1,
        "num_sms": 1,
        "experts": lambda E: [E - 1],
    },
    {
        "name": "multi_tile_with_unused_slot",
        "E": 8,
        "H": 256,
        "Hp": 256,
        "B": 4,
        "num_sms": 8,
        "experts": lambda E: [0, E - 1, E // 2, -1],
    },
    {
        "name": "rectangular_wide_tiles_duplicates",
        "E": 8,
        "H": 256,
        "Hp": 384,
        "B": 5,
        "num_sms": 16,
        "experts": lambda E: [E - 1, 0, E // 2, E // 2, -1],
    },
    {
        "name": "more_sms_than_tiles",
        "E": 16,
        "H": 512,
        "Hp": 128,
        "B": 3,
        "num_sms": 32,
        "experts": lambda E: [E - 1, 1, -1],
    },
    # All -1: no remote load should be issued and every sentinel stays intact.
    {
        "name": "empty_plan",
        "E": 8,
        "H": 128,
        "Hp": 128,
        "B": 4,
        "num_sms": 8,
        "experts": lambda E: [-1, -1, -1, -1],
    },
    # Many tiles per CTA exercise the producer/consumer pipeline wraparound.
    {
        "name": "pipeline_wraparound",
        "E": 16,
        "H": 512,
        "Hp": 384,
        "B": 8,
        "num_sms": 2,
        "experts": lambda E: [3, E - 1, 3, 0, E // 2, E - 1, 7, 1],
    },
    # Expert offsets cross 2^31 elements at expert ids >= 98.
    {
        "name": "i64_offset_7168x3072",
        "E": 104,
        "H": 7168,
        "Hp": 3072,
        "B": 3,
        "num_sms": 32,
        "experts": lambda E: [E - 1, 98, -1],
    },
    {
        "name": "random_plan_s1",
        "E": 32,
        "H": 256,
        "Hp": 128,
        "B": 16,
        "num_sms": 16,
        "experts": lambda E: _random_experts(E, 16, seed=1),
    },
    {
        "name": "random_plan_s2",
        "E": 32,
        "H": 256,
        "Hp": 128,
        "B": 16,
        "num_sms": 16,
        "experts": lambda E: _random_experts(E, 16, seed=2),
    },
    {
        "name": "random_plan_s3",
        "E": 32,
        "H": 256,
        "Hp": 128,
        "B": 16,
        "num_sms": 16,
        "experts": lambda E: _random_experts(E, 16, seed=3),
    },
    # Production 8-card scale: E=896, B=112, H=7168, Hp=3072. Symmetric chunk
    # [epn+B, H, Hp] bf16 = [224, 7168, 3072] ~9.9 GiB (under buffer-backed the VMM
    # descriptor decides the size, so LOCAL_MEM_SIZE is not needed).
    {
        "name": "prod_4096x16_896",
        "E": 896,
        "H": 7168,
        "Hp": 3072,
        "B": 112,
        "num_sms": 32,
        "experts": lambda E: [(i * 7) % E for i in range(112)],
    },
]


def _expert_value(expert, H, Hp, dev):
    gen = torch.Generator(device=dev).manual_seed(2026 + expert)
    return torch.randn(
        H,
        Hp,
        dtype=torch.bfloat16,
        device=dev,
        generator=gen,
    )


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_prefetch requires torchrun with at least 2 NPUs")

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    R = dist.get_world_size()
    # The device index must use LOCAL_RANK (multi-node RANK is in [0,16) while local device ids only go up to 7).
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    if R < 2:
        dist.destroy_process_group()
        pytest.skip("test_prefetch requires at least 2 NPUs")

    yield rank, local_rank, R

    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()
    # Do not destroy the process group: this lets several pytest files run sequentially in one torchrun run.


def make_symmetric_experts(rank, R, E, H, Hp, B, device_id):
    """Allocate the per-rank [ceil(E/R) + B, H, Hp] VMM chunk (registered for buffer-backed
    init). Return the source tensor (``data_ptr`` = source VA); the kernel arguments are
    translated into local symmetric tensors by ``map_to_sym_ptr`` inside ``run_case``. ``device_id`` uses the
    local NPU device id (LOCAL_RANK); on multi-node the global RANK must not be used directly."""
    epn = (E + R - 1) // R
    return create_vmm_physical([epn + B, H, Hp], torch.bfloat16, device_id)[0]


def run_case(rank, local_rank, R, case):
    E = case["E"]
    H = case["H"]
    Hp = case["Hp"]
    B = case["B"]
    num_sms = case["num_sms"]
    dev = f"npu:{local_device_index(local_rank)}"

    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['name']}: H/Hp must be multiples of 128"

    keepalive = make_symmetric_experts(rank, R, E, H, Hp, B, local_rank)
    epn = (E + R - 1) // R
    local_expert = keepalive[:epn]
    prefetch_src = keepalive[epn:]
    local_expert.zero_()
    for expert in range(rank * epn, min((rank + 1) * epn, E)):
        local_expert[expert - rank * epn].copy_(_expert_value(expert, H, Hp, dev))

    expert_ids = case["experts"](E)
    assert len(expert_ids) == B
    experts_to_copy = torch.tensor(
        expert_ids,
        dtype=torch.int32,
        device=dev,
    )
    sentinel = -123.0
    prefetch_src.fill_(sentinel)

    torch.npu.synchronize()
    dist.barrier()

    # Buffer-backed only: the chunk is registered -> init_with_buffer -> translated into a local symmetric
    # tensor (data_ptr = symmetric base address) before being passed to the kernel.
    if not ShmemRuntime.is_initialized():
        init_shmem(group=None)
    sym_base = map_to_sym_ptr(keepalive.data_ptr(), 'prefetch_chunk')
    row_bytes = H * Hp * 2
    remote_expert = create_sym_tensor_from_ptr(
        sym_base, [E, H, Hp], torch.bfloat16, dev)
    prefetch_buffers = create_sym_tensor_from_ptr(
        sym_base + epn * row_bytes, [B, H, Hp], torch.bfloat16, dev)

    launch_prefetch(
        remote_expert,
        prefetch_buffers,
        experts_to_copy,
        num_sms=num_sms,
    )
    torch.npu.synchronize()

    ok = True
    for b, expert in enumerate(expert_ids):
        expected = (
            torch.full_like(prefetch_buffers[b], sentinel)
            if expert < 0
            else _expert_value(expert, H, Hp, dev)
        )
        if not torch.equal(prefetch_buffers[b], expected):
            ok = False
            diff = (
                prefetch_buffers[b].float() - expected.float()
            ).abs().max().item()
            print(
                f"[rank {rank}] {case['name']} mismatch: buffer={b}, "
                f"expert={expert}, max_abs_diff={diff}"
            )

    ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device=dev)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    assert ok_tensor.item() == 1, f"{case['name']} failed"

    if rank == 0:
        print(
            f"  [PASS] {case['name']}: E={E}, B={B}, "
            f"H={H}, Hp={Hp}, num_sms={num_sms}"
        )
    dist.barrier()

    # cleanup (re-init per case): finalize first (tears down the symmetric mapping), then the VMM inverse
    # operations, and clear the registry last. The reverse order would make the next init's get_buffer_ptr fail to see the new chunk.
    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()
    alloc = getattr(keepalive, '_vmm_allocation', None)
    if alloc is not None:
        alloc.destroy()
    reset_symmetric_descriptors()
    torch.npu.synchronize()


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_prefetch_case(dist_env, case):
    rank, local_rank, R = dist_env
    if rank == 0:
        print(f"[test_prefetch] Running case={case['name']} with R={R}")
    run_case(rank, local_rank, R, case)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
