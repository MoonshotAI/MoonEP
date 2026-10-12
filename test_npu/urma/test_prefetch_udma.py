"""Prefetch UDMA correctness test on Ascend950.

Run with:
    torchrun --nproc-per-node=8 -m pytest -s test_npu/urma/test_prefetch_udma.py
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    live = torch.rand(B, generator=generator) < 0.6
    ids = torch.randint(0, E, (B,), generator=generator, dtype=torch.int32)
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
    {
        "name": "empty_plan",
        "E": 8,
        "H": 128,
        "Hp": 128,
        "B": 4,
        "num_sms": 8,
        "experts": lambda E: [-1, -1, -1, -1],
    },
    {
        "name": "pipeline_wraparound",
        "E": 16,
        "H": 512,
        "Hp": 384,
        "B": 8,
        "num_sms": 2,
        "experts": lambda E: [3, E - 1, 3, 0, E // 2, E - 1, 7, 1],
    },
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


def _expert_value(expert, H, Hp, device):
    generator = torch.Generator(device=device).manual_seed(2026 + expert)
    return torch.randn(
        H,
        Hp,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_prefetch_udma requires torchrun with at least 2 NPUs")

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    if world_size < 2:
        dist.destroy_process_group()
        pytest.skip("test_prefetch_udma requires at least 2 NPUs")

    yield rank, local_rank, world_size

    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()


def make_symmetric_experts(rank, R, E, H, Hp, B, device_id):
    del rank
    epn = (E + R - 1) // R
    return create_vmm_physical(
        [epn + B, H, Hp], torch.bfloat16, device_id
    )[0]


def run_case(rank, local_rank, R, case):
    E = case["E"]
    H = case["H"]
    Hp = case["Hp"]
    B = case["B"]
    num_sms = case["num_sms"]
    device = f"npu:{local_device_index(local_rank)}"

    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['name']}: H/Hp must be multiples of 128"

    keepalive = make_symmetric_experts(rank, R, E, H, Hp, B, local_rank)
    allocation = getattr(keepalive, "_vmm_allocation", None)
    epn = (E + R - 1) // R
    try:
        local_expert = keepalive[:epn]
        prefetch_source = keepalive[epn:]
        local_expert.zero_()
        for expert in range(rank * epn, min((rank + 1) * epn, E)):
            local_expert[expert - rank * epn].copy_(
                _expert_value(expert, H, Hp, device)
            )

        expert_ids = case["experts"](E)
        assert len(expert_ids) == B
        experts_to_copy = torch.tensor(
            expert_ids, dtype=torch.int32, device=device
        )
        sentinel = -123.0
        prefetch_source.fill_(sentinel)

        torch.npu.synchronize()
        dist.barrier()
        os.environ["ASCEND_MOONEP_USE_UDMA"] = "1"
        os.environ["ASCEND_MOONEP_PREFETCH_USE_UDMA"] = "1"
        init_shmem(
            group=None, enable_udma=True, qp_count=2
        )

        symmetric_base = map_to_sym_ptr(
            keepalive.data_ptr(), "prefetch_udma_chunk"
        )
        row_bytes = H * Hp * 2
        remote_expert = create_sym_tensor_from_ptr(
            symmetric_base, [E, H, Hp], torch.bfloat16, device
        )
        prefetch_buffers = create_sym_tensor_from_ptr(
            symmetric_base + epn * row_bytes,
            [B, H, Hp],
            torch.bfloat16,
            device,
        )

        launch_prefetch(
            remote_expert,
            prefetch_buffers,
            experts_to_copy,
            num_sms=num_sms,
        )
        torch.npu.synchronize()

        ok = True
        for slot, expert in enumerate(expert_ids):
            expected = (
                torch.full_like(prefetch_buffers[slot], sentinel)
                if expert < 0
                else _expert_value(expert, H, Hp, device)
            )
            if not torch.equal(prefetch_buffers[slot], expected):
                ok = False
                max_diff = (
                    prefetch_buffers[slot].float() - expected.float()
                ).abs().max().item()
                print(
                    f"[rank {rank}] {case['name']} mismatch: buffer={slot}, "
                    f"expert={expert}, max_abs_diff={max_diff}"
                )

        ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device=device)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        assert ok_tensor.item() == 1, f"{case['name']} failed"
        if rank == 0:
            print(
                f"  [PASS] {case['name']}: E={E}, B={B}, "
                f"H={H}, Hp={Hp}, num_sms={num_sms}"
            )
        dist.barrier()
    finally:
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        if allocation is not None:
            allocation.destroy()
        reset_symmetric_descriptors()
        torch.npu.synchronize()


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_prefetch_udma_case(dist_env, case):
    rank, local_rank, world_size = dist_env
    if rank == 0:
        print(f"[test_prefetch_udma] Running case={case['name']} with R={world_size}")
    run_case(rank, local_rank, world_size, case)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
