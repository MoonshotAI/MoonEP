# SPDX-License-Identifier: Apache-2.0
"""Prepared communication-bank correctness through Buffer.dispatch/combine.

Like test_buffer_combine.py, verify exact round trips through the Buffer API.
Keep three zero-copy results live in distinct caller-owned banks, then switch
back to the default storage and check that copying results survive later calls.
Dispatch and Combine use MTE; expert pointer operations are tested separately.

Run in fresh workers (EP2 or EP8):
    torchrun --nproc-per-node=2 -m pytest -s -x \
        test_npu/buffer_test/test_buffer_comm_buffers.py
"""
import torch
import torch.distributed as dist
import torch_npu
import pytest

from kernel_test_utils import init_shmem
from npu_patch import Buffer, ShmemRuntime
from npu_patch.buffer_c import create_vmm_physical, reset_symmetric_descriptors


# Retain backing on failure; the launcher must stop all workers together.
_KEEPALIVE = []


def test_buffer_comm_buffers(dist_env):
    rank, world = dist_env
    if world not in (2, 8):
        pytest.skip("communication-bank recipe requires EP2 or EP8")
    if ShmemRuntime.is_initialized():
        raise RuntimeError("run this test in fresh workers without an active SHMEM heap")
    s, h, k, local, b = 16, 128, 1, 2, 2
    device = torch.device("npu", torch.npu.current_device())
    buffer = Buffer(S=s, H=h, K=k, E=world * local, B=b, num_ep_ranks=world,
                    group=dist.group.WORLD, explicitly_destroy=True, use_udma=False)
    sources = []
    _KEEPALIVE.extend((buffer, sources))
    for _ in range(3):
        for shape, dtype in (((s * k, h), torch.bfloat16), ((s * k,), torch.int32)):
            sources.append(create_vmm_physical(shape, dtype, device.index)[0])
    # Register every bank in the same order on all ranks before SHMEM init.
    init_shmem(group=dist.group.WORLD, enable_udma=False)
    banks = [buffer.prepare_comm_buffers(*sources[i:i + 2]) for i in range(0, 6, 2)]
    current = torch_npu.npu.current_stream()
    routes = torch.zeros(s, k, dtype=torch.int32, device=device)
    counts = torch.bincount(routes.flatten().long(), minlength=world * local).int()
    probabilities = torch.ones(s, k, dtype=torch.float32, device=device)
    saved = []
    copied = []
    _KEEPALIVE.extend((saved, copied))
    for value, bank in zip((1, 2, 3), banks):
        hidden = torch.full((s, h), rank * 4 + value, dtype=torch.bfloat16, device=device)
        buffer.set_comm_buffers(bank)
        out, probs, _, plan, done = buffer.dispatch(
            hidden, probabilities, routes, counts, async_finish=True,
            zero_copy=True, router_weights_zero_copy=True)
        assert out.data_ptr() == bank.hidden.data_ptr()
        assert probs.data_ptr() == bank.route.data_ptr()
        saved.append((hidden, out, plan, done, bank))
    # Each earlier zero-copy result must survive the other banks' dispatches.
    for hidden, out, plan, done, bank in saved:
        current.wait_event(done)
        buffer.set_comm_buffers(bank)
        combined, _, event = buffer.combine(plan, out, async_finish=True, zero_copy=True)
        current.wait_event(event)
        torch.npu.synchronize()
        torch.testing.assert_close(combined, hidden, atol=0, rtol=0)

    # None explicitly restores the default communication storage.
    buffer.set_comm_buffers(None)
    for value in (5, 6):
        hidden = torch.full((s, h), value, dtype=torch.bfloat16, device=device)
        out, _, _, plan, event = buffer.dispatch(
            hidden, probabilities, routes, counts, async_finish=True)
        copied.append((hidden, out, plan, event))
    for hidden, out, plan, event in copied:
        current.wait_event(event)
        result, _, event = buffer.combine(plan, out, async_finish=True)
        current.wait_event(event)
        torch.npu.synchronize()
        torch.testing.assert_close(result, hidden, atol=0, rtol=0)

    # Only successful, globally drained execution may release symmetric backing.
    torch.npu.synchronize()
    dist.barrier()
    ShmemRuntime.finalize()
    buffer.destroy()
    for source in sources:
        source._vmm_allocation.destroy()
    reset_symmetric_descriptors()
    _KEEPALIVE.clear()
