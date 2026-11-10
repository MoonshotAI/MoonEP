# Adapted from MindSpeed tests_extend/system_tests/moonep/test_grad_reduce.py
# SPDX-License-Identifier: Apache-2.0
"""Grad reduce correctness test for ascend-moonep.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s test_npu/mte/test_grad_reduce.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

import pytest
import torch
import torch.distributed as dist

import torch_npu

from npu_patch._device import local_device_index
from npu_patch import ShmemRuntime
from npu_patch.buffer_c import (
    create_vmm_physical,
    reset_symmetric_descriptors,
    sym_tensor,
    symmetric_descriptors,
)
from npu_patch.launchers import launch_grad_reduce

# --- Original implementation (option A: the shmem Python package) ---
# import importlib
# _SHMEM = importlib.import_module("shmem")
# _SHMEM.aclshmemx_init_status()


def _launch_grad_reduce(
    expert_grads, reduce_buffers, plan, epn, H, Hp, R, B, rank, blocks,
):
    del epn, H, Hp, R, B
    launch_grad_reduce(
        expert_grads,
        reduce_buffers,
        plan,
        rank=rank,
        num_sms=blocks,
        meta_buf=None,
        meta_stride=0,
        barrier_off=0,
        grid_sync_bar=None,
    )


# ---------------------------------------------------------------------------
# plan builders
# ---------------------------------------------------------------------------

def _plan_ring(R, B, epn, dev, seed=0):
    plan = torch.full((R, B), -1, dtype=torch.int32)
    for src_rank in range(R):
        dst_rank = (src_rank + 1) % R
        plan[src_rank, 0] = dst_rank * epn + (src_rank % epn)
        if B > 1:
            dst_rank = (src_rank + 2) % R
            plan[src_rank, 1] = dst_rank * epn + ((src_rank + 1) % epn)
        if B > 2:
            dst_rank = (src_rank + R - 1) % R
            plan[src_rank, 2] = dst_rank * epn + ((src_rank + 2) % epn)
        if B > 3:
            dst_rank = (src_rank + 1) % R
            plan[src_rank, 3] = (
                dst_rank * epn + (epn - 1 - (src_rank % epn)))
    return plan.to(dev)


def _plan_empty(R, B, epn, dev, seed=0):
    return torch.full((R, B), -1, dtype=torch.int32).to(dev)


def _plan_fan_in(R, B, epn, dev, seed=0):
    assert 2 * R <= B, f"fan_in plan needs B >= 2R, got B={B}, R={R}"
    plan = torch.full((R, B), -1, dtype=torch.int32)
    for dst in range(R):
        expert = dst * epn + (dst % epn)
        for src in range(R):
            plan[src, 2 * dst] = expert
            plan[src, 2 * dst + 1] = expert
    return plan.to(dev)


def _plan_balanced_fan_in(R, B, epn, dev, seed=0):
    """Give one expert on every destination one contribution per source."""
    assert B >= R, f"balanced_fan_in needs B >= R, got B={B}, R={R}"
    plan = torch.full((R, B), -1, dtype=torch.int32)
    for src in range(R):
        for dst in range(R):
            plan[src, dst] = dst * epn
    return plan.to(dev)


def _plan_asymmetric(R, B, epn, dev, seed=0):
    plan = torch.full((R, B), -1, dtype=torch.int32)
    for src in range(R):
        if src == 1:
            continue
        dst = src + 1 if src + 1 < R else 1
        plan[src, 0] = dst * epn + (src % epn)
        if B > 1:
            plan[src, 1] = dst * epn + ((src + 1) % epn)
    return plan.to(dev)


def _plan_random(R, B, epn, dev, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    live = torch.rand((R, B), generator=generator) < 0.6
    ids = torch.randint(
        0, R * epn, (R, B), generator=generator, dtype=torch.int32)
    return torch.where(live, ids, torch.full_like(ids, -1)).to(dev)


PLAN_BUILDERS = {
    "ring": _plan_ring,
    "empty": _plan_empty,
    "fan_in": _plan_fan_in,
    "balanced_fan_in": _plan_balanced_fan_in,
    "asymmetric": _plan_asymmetric,
    "random": _plan_random,
}


# ---------------------------------------------------------------------------
# data generators
# ---------------------------------------------------------------------------

def _grad_values(epn, H, Hp, dev):
    """Pristine home grad values for this rank's ``epn`` local experts."""
    expert = torch.arange(epn, dtype=torch.float32, device=dev).view(epn, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return expert * 17.0 + row * 0.125 + col * 0.0078125


def _slot_values(rank, B, H, Hp, dev):
    slot = torch.arange(B, dtype=torch.float32, device=dev).view(B, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=dev).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, 1, Hp)
    return (rank + 1) * 101.0 + slot * 11.0 + row * 0.03125 + col * 0.00390625


def _randn(shape, seed, dev):
    generator = torch.Generator(device=dev)
    generator.manual_seed(seed)
    return torch.randn(
        shape, generator=generator, device=dev, dtype=torch.float32)


def _make_data_fns(mode, rank, epn, B, H, Hp, dev, seed):
    """Return (grads_fn, slot_fn). Grads cover the local ``epn`` home experts."""
    if mode == "affine":
        return (
            lambda: _grad_values(epn, H, Hp, dev),
            lambda src: _slot_values(src, B, H, Hp, dev),
        )
    assert mode == "randn", mode
    return (
        lambda: _randn((epn, H, Hp), 200_000 + 1_000 * seed + rank, dev),
        lambda src: _randn(
            (B, H, Hp), 100_000 + 1_000 * seed + src, dev),
    )


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def _expected_for_rank(rank, R, epn, B, plan_cpu, grads_fn, slot_fn):
    """Expected grads for this rank's ``epn`` local home experts.

    Start from the pristine home grads and accumulate every contribution in the
    plan (all ranks' rows) that targets one of this rank's experts.
    """
    expected = grads_fn()  # [epn, H, Hp]
    for src in range(R):
        if not bool((plan_cpu[src] >= 0).any()):
            continue
        slot_vals = slot_fn(src)
        for b in range(B):
            global_expert = int(plan_cpu[src, b])
            if global_expert >= 0:
                local_expert = global_expert - rank * epn
                if 0 <= local_expert < epn:
                    expected[local_expert].add_(slot_vals[b])
    return expected.clone()


def _verify(rank, R, epn, B, plan, grads_fn, slot_fn,
            expert_grads, reduce_buffers, label):
    plan_cpu = plan.cpu()

    expected_local = _expected_for_rank(
        rank, R, epn, B, plan_cpu, grads_fn, slot_fn)
    ok_grads = torch.equal(expert_grads, expected_local)

    zero_ok = True
    unchanged_ok = True
    original = slot_fn(rank)
    for b in range(B):
        expert = int(plan_cpu[rank, b])
        if expert >= 0:
            if not torch.equal(reduce_buffers[b],
                               torch.zeros_like(reduce_buffers[b])):
                zero_ok = False
        elif not torch.equal(reduce_buffers[b], original[b]):
            unchanged_ok = False

    ok = bool(ok_grads and zero_ok and unchanged_ok)
    if not ok:
        print(
            f"[rank {rank}] {label}: grads={ok_grads} "
            f"zero={zero_ok} unchanged={unchanged_ok}"
        )

    ok_tensor = torch.tensor(
        [int(ok)], dtype=torch.int32, device=expert_grads.device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    assert int(ok_tensor.item()) == 1, label


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

CASES = [
    {"name": "prod_4096x16_896", "epn": 112, "H": 7168,
     "Hp": 3072, "base_B": 112, "num_sms": 32},
    {"name": "single_tile", "epn": 4, "H": 128, "Hp": 128,
     "base_B": 4, "num_sms": 8},
    {"name": "multi_tile", "epn": 4, "H": 256, "Hp": 128,
     "base_B": 5, "num_sms": 16},
    {"name": "i64_offset_7168x3072", "epn": 32, "H": 7168,
     "Hp": 3072, "base_B": 4, "num_sms": 32},
    {"name": "fan_in_gt_stages", "epn": 4, "H": 256, "Hp": 128,
     "base_B": 16, "num_sms": 8, "plan": "fan_in"},
    {"name": "balanced_fan_in", "epn": 4, "H": 256, "Hp": 128,
     "base_B": 8, "num_sms": 8, "plan": "balanced_fan_in"},
    {"name": "empty_plan", "epn": 4, "H": 128, "Hp": 128,
     "base_B": 4, "num_sms": 8, "plan": "empty"},
    {"name": "asymmetric", "epn": 4, "H": 256, "Hp": 128,
     "base_B": 4, "num_sms": 16, "plan": "asymmetric"},
    {"name": "epn1_num_sms1", "epn": 1, "H": 128, "Hp": 128,
     "base_B": 4, "num_sms": 1},
    # Large R*B validates direct GM plan scanning and cacheline-safe clear.
    {"name": "large_plan_direct_scan", "epn": 1, "H": 8, "Hp": 8,
     "base_B": 8192, "num_sms": 16},
    {"name": "ring_randn", "epn": 4, "H": 256, "Hp": 128,
     "base_B": 5, "num_sms": 16, "data": "randn", "seed": 7},
    {"name": "random_plan_s1", "epn": 8, "H": 256, "Hp": 128,
     "base_B": 16, "num_sms": 16, "plan": "random",
     "data": "randn", "seed": 1},
    {"name": "random_plan_s2", "epn": 8, "H": 256, "Hp": 128,
     "base_B": 16, "num_sms": 16, "plan": "random",
     "data": "randn", "seed": 2},
    {"name": "random_plan_s3", "epn": 8, "H": 256, "Hp": 128,
     "base_B": 16, "num_sms": 16, "plan": "random",
     "data": "randn", "seed": 3},
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_grad_reduce requires torchrun with at least 2 NPUs")

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
        pytest.skip("test_grad_reduce requires at least 2 NPUs")

    yield rank, local_rank, R

    dist.barrier()
    # Do not destroy the process group: this lets several pytest files run sequentially in one torchrun run.


def run_case(rank, local_rank, R, case):
    dev = f"npu:{local_device_index(local_rank)}"
    epn = case["epn"]
    E = R * epn
    H = case["H"]
    Hp = case["Hp"]
    # Buffer uses exact-size per-PE symmetric chunks.  Individual allocations
    # do not need 2 MiB padding; only the total SHMEM heap is aligned.
    B = case["base_B"]
    blocks = case["num_sms"]
    seed = case.get("seed", 0)

    plan = PLAN_BUILDERS[case.get("plan", "ring")](
        R, B, epn, dev, seed)

    grads_fn, slot_fn = _make_data_fns(
        case.get("data", "affine"), rank, epn, B, H, Hp, dev, seed)

    expert_grads_src = None
    reduce_buffers_src = None
    try:
        # Buffer-backed only: allocate/register the VMM chunk -> init_with_buffer -> sym_tensor translation.
        # Operator requirements (GradReduce kernel):
        #   - expert_grads is a shmem tensor of the **local** epn home grad rows, [epn, H, Hp]
        #     (not the whole E=R*epn); the kernel derives the global expert id from local_begin=rank*epn;
        #   - reduce_buffers is this rank's **local** B-slot chunk, [B, H, Hp] (same as
        #     the slot_chunks in test_buffer_grad_reduce.py, read across ranks via aclshmem RMA, so
        #     it must be a symmetric base address, not an unregistered plain local tensor).
        # Allocation/init are inside try: if init fails (e.g. the shmem 40 GiB heap limit, -2) the finally
        # block still finalizes/destroys/resets, so registered descriptors do not leak into the next case
        # (this previously made test_grad_reduce_repeated_launch fail in a cascade).
        expert_grads_src, _, _ = create_vmm_physical(
            (epn, H, Hp), torch.float32, local_device_index(local_rank))
        reduce_buffers_src, _, _ = create_vmm_physical(
            (B, H, Hp), torch.float32, local_device_index(local_rank))
        if not ShmemRuntime.is_initialized():
            # Single-node switched to UID bootstrap (rank0 generates a fresh UID and broadcasts it over HCCL), to avoid
            # the second-init reuse problem of ip_port/DEFAULT config-store on the fixed port
            # (this previously made the shmem re-init after the first case fail or hang).
            try:
                ShmemRuntime.init_with_buffer(
                    group=None,
                    descriptors=symmetric_descriptors(),
                )
            except Exception as e:
                if rank == 0:
                    print(f"  [WARN] shmem UID init failed: {e!r}")
                raise
        expert_grads = sym_tensor(expert_grads_src)
        reduce_buffers = sym_tensor(reduce_buffers_src)

        reduce_buffers.zero_()
        expert_grads.copy_(grads_fn())
        reduce_buffers.copy_(slot_fn(rank))
        torch.npu.synchronize()
        dist.barrier()

        _launch_grad_reduce(
            expert_grads, reduce_buffers, plan, epn, H, Hp, R, B, rank, blocks,
        )
        torch.npu.synchronize()
        dist.barrier()

        _verify(rank, R, epn, B, plan, grads_fn, slot_fn,
                expert_grads, reduce_buffers,
                f"{case['name']} grad_reduce failed")

        if rank == 0:
            print(
                f"  [PASS] {case['name']}: R={R}, E={E}, B={B}, "
                f"H={H}, Hp={Hp}, num_sms={blocks}"
            )

        dist.barrier()
    finally:
        # Collective barrier first: make sure every rank (including the _verify failure path) enters
        # teardown before the collective aclshmem_finalize runs, so a non-collective finalize cannot poison
        # the runtime and cascade into later cases (aligned with test_buffer_grad_reduce).
        try:
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"  [WARN] pre-finalize dist.barrier failed: {e!r}")
        # Re-init per case: finalize first, then the VMM inverse operations, and clear the registry last.
        if ShmemRuntime.is_initialized():
            try:
                ShmemRuntime.finalize()
            except Exception as e:
                # Log the rc instead of staying silent: a failed finalize poisons the runtime, so the next
                # case's init reports the error explicitly, which helps locate the first failure point.
                if rank == 0:
                    print(f"  [WARN] ShmemRuntime.finalize raised: {e!r}")
        for t in (expert_grads_src, reduce_buffers_src):
            if t is not None:
                alloc = getattr(t, '_vmm_allocation', None)
                if alloc is not None:
                    try:
                        alloc.destroy()
                    except Exception as e:
                        if rank == 0:
                            print(f"  [WARN] VMM alloc.destroy failed: {e!r}")
        try:
            reset_symmetric_descriptors()
        except Exception as e:
            if rank == 0:
                print(f"  [WARN] reset_symmetric_descriptors raised: {e!r}")


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_grad_reduce(dist_env, case):
    rank, local_rank, R = dist_env
    if rank == 0:
        print(f"[test_grad_reduce] case={case['name']} with R={R}")
    run_case(rank, local_rank, R, case)


def test_grad_reduce_repeated_launch(dist_env):
    rank, local_rank, R = dist_env
    dev = f"npu:{local_device_index(local_rank)}"
    epn, H, Hp, base_B, num_sms = 4, 256, 128, 5, 16
    B = base_B
    expert_grads_src = None
    reduce_buffers_src = None
    try:
        # Buffer-backed only: allocate/register the VMM chunk -> init_with_buffer -> sym_tensor translation.
        # The operator wants the local epn home grads ([epn,H,Hp]) plus the local B slots ([B,H,Hp]).
        expert_grads_src, _, _ = create_vmm_physical(
            (epn, H, Hp), torch.float32, local_device_index(local_rank))
        reduce_buffers_src, _, _ = create_vmm_physical(
            (B, H, Hp), torch.float32, local_device_index(local_rank))
        if not ShmemRuntime.is_initialized():
            # Single-node switched to UID bootstrap, same as run_case, to avoid cross-reusing
            # ip_port/DEFAULT and UID bootstrap within the same process.
            try:
                ShmemRuntime.init_with_buffer(
                    group=None,
                    descriptors=symmetric_descriptors(),
                )
            except Exception as e:
                if rank == 0:
                    print(f"  [WARN] shmem UID init failed: {e!r}")
                raise
        expert_grads = sym_tensor(expert_grads_src)
        reduce_buffers = sym_tensor(reduce_buffers_src)

        rounds = [("random", 11), ("empty", 0), ("random", 12)]
        for iteration, (plan_name, plan_seed) in enumerate(rounds):
            plan = PLAN_BUILDERS[plan_name](
                R, B, epn, dev, plan_seed)
            grads_fn, slot_fn = _make_data_fns(
                "randn", rank, epn, B, H, Hp, dev, seed=50 + iteration)

            expert_grads.copy_(grads_fn())
            reduce_buffers.zero_()
            reduce_buffers.copy_(slot_fn(rank))
            torch.npu.synchronize()
            dist.barrier()

            _launch_grad_reduce(
                expert_grads, reduce_buffers, plan,
                epn, H, Hp, R, B, rank, num_sms,
            )
            torch.npu.synchronize()
            dist.barrier()

            _verify(
                rank, R, epn, B, plan, grads_fn, slot_fn,
                expert_grads, reduce_buffers,
                f"repeated_launch round {iteration} ({plan_name}) failed",
            )

        if rank == 0:
            print(f"  [PASS] repeated_launch: R={R}, rounds={len(rounds)}")
        dist.barrier()
    finally:
        # Collective barrier first: avoids a non-collective finalize poisoning the runtime and cascading into
        # later cases (aligned with test_buffer_grad_reduce).
        try:
            dist.barrier()
        except Exception as e:
            if rank == 0:
                print(f"  [WARN] pre-finalize dist.barrier failed: {e!r}")
        # Finalize first, then the VMM inverse operations, and clear the registry last.
        if ShmemRuntime.is_initialized():
            try:
                ShmemRuntime.finalize()
            except Exception as e:
                if rank == 0:
                    print(f"  [WARN] ShmemRuntime.finalize raised: {e!r}")
        for t in (expert_grads_src, reduce_buffers_src):
            if t is not None:
                alloc = getattr(t, '_vmm_allocation', None)
                if alloc is not None:
                    try:
                        alloc.destroy()
                    except Exception as e:
                        if rank == 0:
                            print(f"  [WARN] VMM alloc.destroy failed: {e!r}")
        try:
            reset_symmetric_descriptors()
        except Exception as e:
            if rank == 0:
                print(f"  [WARN] reset_symmetric_descriptors raised: {e!r}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
