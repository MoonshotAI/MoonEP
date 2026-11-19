# SPDX-License-Identifier: Apache-2.0
"""MTE canonical-pointer GradReduce correctness and custom-class compatibility.

Same structure as this directory's dense ``test_grad_reduce.py`` (buffer-backed
VMM source bank, per-case SHMEM init, direct ``npu_patch.launchers`` call, same
``CASES`` / ``PLAN_BUILDERS`` / data generators / reference verification): the
only difference is the input — this variant hands the op a frozen INT64 ``[epn]``
table of ordinary FP32 ``[H, H']`` target addresses instead of an
``[epn, H, H']`` home-grad chunk, which is exactly what
``launch_grad_reduce_targets`` takes. Only the ``[B, H, H']`` source bank is
symmetric.

``test_grad_reduce_targets_custom_class`` keeps the narrower API contract checks
that the case suite does not cover: reverse-ordered, separately allocated and
therefore non-contiguous targets, the original custom-class signature (including
its default arguments and an ignored explicit scratch), the ``GradReduceOp``
ctx path, and a late producer behind the caller's readiness fence.

Run in fresh workers, with a whole-job external hard timeout:
    torchrun --nproc-per-node=8 -m pytest -s -x \
        test_npu/mte/test_grad_reduce_targets.py
"""
import os
import time

import pytest
import torch
import torch.distributed as dist
import torch_npu

from kernel_test_utils import init_shmem
from npu_patch import ShmemRuntime
from npu_patch._device import local_device_index
from npu_patch.buffer_c import create_vmm_physical, reset_symmetric_descriptors, sym_tensor
from npu_patch.launchers import launch_grad_reduce_targets
from npu_patch.ops import GradReduceOp
# Reuse the dense variant verbatim: same cases, same plans, same data
# generators and the same reference verification.
from test_grad_reduce import (
    CASES,
    PLAN_BUILDERS,
    _make_data_fns,
    _verify,
    dist_env,
)

_USE_UDMA_ENV = "ASCEND_MOONEP_USE_UDMA"
_GRAD_REDUCE_UDMA_ENV = "ASCEND_MOONEP_GRAD_REDUCE_USE_UDMA"

# Failed native work must be stopped collectively by the external launcher.
_KEEPALIVE = []


def _configure_udma_env(use_udma):
    """Set/unset the UDMA toggles consumed by the grad reduce launch."""
    if use_udma:
        os.environ[_USE_UDMA_ENV] = "1"
        os.environ[_GRAD_REDUCE_UDMA_ENV] = "1"
    else:
        os.environ.pop(_USE_UDMA_ENV, None)
        os.environ.pop(_GRAD_REDUCE_UDMA_ENV, None)


def run_case(rank, local_rank, R, case):
    dev = f"npu:{local_device_index(local_rank)}"
    epn = case["epn"]
    E = R * epn
    H = case["H"]
    Hp = case["Hp"]
    # The slot bank is an independent input (``base_B``), exactly as in the dense
    # variant: it needs one B-row symmetric chunk, not one row per local expert.
    B = case["base_B"]
    blocks = case["num_sms"]
    seed = case.get("seed", 0)

    plan = PLAN_BUILDERS[case.get("plan", "ring")](R, B, epn, dev, seed)
    grads_fn, slot_fn = _make_data_fns(
        case.get("data", "affine"), rank, epn, B, H, Hp, dev, seed)

    target_table = None
    reduce_buffers_src = None
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

        _configure_udma_env(False)
        if not ShmemRuntime.is_initialized():
            init_shmem(group=None)
        reduce_buffers = sym_tensor(reduce_buffers_src)

        reduce_buffers.zero_()
        target_table.copy_(grads_fn())
        reduce_buffers.copy_(slot_fn(rank))
        torch.npu.synchronize()
        dist.barrier()

        launch_grad_reduce_targets(
            target_ptrs, reduce_buffers, plan, rank=rank, num_sms=blocks,
        )
        torch.npu.synchronize()
        dist.barrier()

        _verify(rank, R, epn, B, plan, grads_fn, slot_fn,
                target_table, reduce_buffers,
                f"{case['name']} grad_reduce_targets failed")

        if rank == 0:
            print(f"[PASS] {case['name']} epn={epn} B={B} H={H} Hp={Hp} "
                  f"blocks={blocks} udma=False", flush=True)
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


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_grad_reduce_targets(dist_env, case):
    rank, local_rank, R = dist_env
    run_case(rank, local_rank, R, case)


@pytest.mark.parametrize("shape", [(129, 101), (6144, 7168), (7168, 3072)],
                         ids=["tail_tile", "fc1", "fc2"])
def test_grad_reduce_targets_custom_class(dist_env, shape, monkeypatch):
    rank, local_rank, ranks = dist_env
    if ShmemRuntime.is_initialized():
        raise RuntimeError("run this test without an active SHMEM heap")
    device = f"npu:{local_device_index(local_rank)}"
    local, b, blocks = 6, 6, 32
    h, hp = shape
    monkeypatch.setenv("ASCEND_MOONEP_GRAD_REDUCE_USE_UDMA", "0")
    source, _, _ = create_vmm_physical((b, h, hp), torch.float32,
                                      local_device_index(local_rank))
    _KEEPALIVE.append(source)
    init_shmem(enable_udma=False)
    slots = sym_tensor(source)
    slots.zero_()
    targets = tuple(torch.full((h, hp), float(100 + expert), dtype=torch.float32,
                               device=device) for expert in range(local))[::-1]
    table = torch.tensor([target.data_ptr() for target in targets], dtype=torch.int64, device=device)
    addresses = tuple(target.data_ptr() for target in targets)
    wanted = [float(100 + local - 1 - expert) for expert in range(local)]
    op = GradReduceOp(ws_bytes=0, num_sms=blocks)
    ctx = dict(E=ranks * local, R=ranks, B=b)
    # Supplied scratch must not override an explicit MTE flag.
    unused_scratch = torch.empty(1, dtype=torch.float32, device=device)
    _KEEPALIVE.extend((targets, table, op, unused_scratch))
    plan_cases = ("random", "empty", "asymmetric", "hotspot", "ring")
    for iteration, name in enumerate(plan_cases):
        final_round = iteration == len(plan_cases) - 1
        plan = (torch.zeros((ranks, b), dtype=torch.int32, device=device)
                if name == "hotspot" else
                PLAN_BUILDERS[name](ranks, b, local, device, 7 + iteration))
        _KEEPALIVE.append(plan)
        rows = plan.cpu().tolist()
        # Exercise a late producer; the caller's readiness fence must cover it.
        if rank == ranks - 1 and iteration == 2:
            time.sleep(.05)
        was_clean = []
        for slot, expert in enumerate(rows[rank]):
            if expert >= 0:
                was_clean.append(torch.all(slots[slot] == 0))
                slots[slot].add_(1 + rank * b + slot + iteration * 100)
            elif final_round:
                slots[slot].fill_(-1234)
        for peer, row in enumerate(rows):
            for slot, expert in enumerate(row):
                index = expert - rank * local
                if expert >= 0 and 0 <= index < local:
                    wanted[index] += 1 + peer * b + slot + iteration * 100
        # Native completion covers reduction/clear, not cross-rank publication.
        torch.npu.synchronize()
        dist.barrier()
        if iteration == 0:
            # The original custom-class signature defaults to MTE.
            torch.classes.AscendMoonEP.GradReduce().compute_targets(
                table, slots, plan, local, h, hp, ranks, b, rank, blocks)
        elif iteration == 1:
            torch.classes.AscendMoonEP.GradReduce().compute_targets(
                table, slots, plan, local, h, hp, ranks, b, rank, blocks,
                False, unused_scratch)
        else:
            op.compute_targets(ctx, table, slots, plan, rank, blocks)
        torch.npu.synchronize()
        checks = was_clean + [torch.all(target == value) for target, value in zip(targets, wanted)]
        checks += [torch.all(slots[slot] == (0 if expert >= 0 or not final_round else -1234))
                   for slot, expert in enumerate(rows[rank])]
        ok = torch.stack(checks).all().to(torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        assert bool(ok), f"MTE canonical {shape} round={iteration} failed"
        assert tuple(target.data_ptr() for target in targets) == addresses
    if rank == 0:
        print(f"[PASS] MTE canonical targets shape={shape} blocks={blocks} "
              f"changed_plans={len(plan_cases)} valid_slot_clear=True")
    # Release only after successful completion on every rank.
    torch.npu.synchronize()
    dist.barrier()
    ShmemRuntime.finalize()
    op.destroy()
    source._vmm_allocation.destroy()
    reset_symmetric_descriptors()
    _KEEPALIVE.clear()
