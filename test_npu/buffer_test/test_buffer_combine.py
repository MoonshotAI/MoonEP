"""Buffer.combine correctness test (combine kernel driven by Buffer API).

Runs a full ``dispatch(plan=None)`` -> ``combine`` round-trip and verifies the
combine output: every token's K dispatched copies must sum back to
``K * hidden`` (K-sum), and the gathered route weights must round-trip exactly.
This complements ``test_buffer_plan.py`` / ``test_buffer_dispatch_with_reference_plan.py``
with an independent combine-precision check.

Run with (Ascend host, CANN + torch_npu + shmem + built npu_patch):

    torchrun --nproc_per_node=8 -m pytest -s test_npu/buffer_test/test_buffer_combine.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem

import pytest
import torch
import torch.distributed as dist

import torch_npu

from generate_topk_routing import generate_topk_routing

from npu_patch._device import local_device_index
from npu_patch import Buffer, ShmemRuntime
from npu_patch.buffer_c import reset_symmetric_descriptors
from kernel_test_utils import assert_close_all_ranks


def make_hidden(src_rank: int, token_id: int, S: int, H: int) -> torch.Tensor:
    base = float(src_rank * S + token_id)
    col = (torch.arange(H, dtype=torch.float32) % 16) / 64.0
    return (base + col).to(torch.bfloat16)


def hidden_sh_for(rank: int, S: int, H: int) -> torch.Tensor:
    return torch.stack([make_hidden(rank, t, S, H) for t in range(S)]).contiguous()


def make_weight(src_rank: int, token_id: int, topk_id: int, S: int, K: int) -> float:
    return float(src_rank * S * K + token_id * K + topk_id)


def weights_sk_for(rank: int, S: int, K: int) -> torch.Tensor:
    base = float(rank * S * K)
    return (torch.arange(S * K, dtype=torch.float32) + base).reshape(S, K).contiguous()


CASES = [
    {"name": "balanced", "S": 64, "K": 4, "H": 64, "E": 32, "bias": 0.0},
    {"name": "biased", "S": 64, "K": 4, "H": 64, "E": 32, "bias": 1.0},
    {"name": "k16", "S": 32, "K": 16, "H": 64, "E": 32, "bias": 0.0},
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_buffer_combine requires torchrun with at least 2 NPUs")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_device_index(local_rank))
    owns = not dist.is_initialized()
    if owns:
        dist.init_process_group(backend="hccl", device_id=torch.device(f"npu:{local_device_index(local_rank)}"))
    R = dist.get_world_size()
    if R < 2:
        if owns:
            dist.destroy_process_group()
        pytest.skip("test_buffer_combine requires at least 2 NPUs")
    yield rank, local_rank, R
    dist.barrier()
    if owns:
        dist.destroy_process_group()


def run_case(rank, local_rank, R, case, use_udma: bool = True):
    dev = f"npu:{local_device_index(local_rank)}"
    S, H, K, E = case["S"], case["H"], case["K"], case["E"]
    bias = case["bias"]
    seed = case.get("seed", 1234)
    assert E % R == 0 and K <= E

    buffer = None
    try:
        buffer = Buffer(S, H, K, E, R, explicitly_destroy=True, use_udma=use_udma)
        if not ShmemRuntime.is_initialized():
            init_shmem(group=None, enable_udma=use_udma)
        ctx = buffer._require_ctx()

        topk_all, tpe_all = generate_topk_routing(S, K, E, R, bias, "cpu", seed, all_ranks=True)
        topk_this = topk_all[rank].to(dev)
        tpe_this = tpe_all[rank].to(dev)

        hidden_sh = hidden_sh_for(rank, S, H).to(dev)
        route_weights_sk = weights_sk_for(rank, S, K).to(dev)

        label = f"{case['name']} (R={R}, use_udma={use_udma})"

        # dispatch (plan=None -> planning kernel runs) then combine
        hidden_nvsh, weights_nvs, _cu, plan_out = buffer.dispatch(
            hidden_sh, route_weights_sk,
            topk_experts_sk=topk_this, tokens_per_expert=tpe_this)
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        hidden_out, route_weights_out, _ = buffer.combine(
            plan=plan_out, hidden_nvsh=hidden_nvsh, route_weights_nvs=weights_nvs)
        torch.npu.synchronize()
        ShmemRuntime.barrier()

        # golden: K-sum of the identical dispatched hidden copies
        golden_hidden = (K * hidden_sh.float()).to(torch.bfloat16)
        assert_close_all_ranks(
            f"{label} combine hidden", hidden_out.cpu().float(),
            golden_hidden.cpu().float(), rank, R, atol=0.05)

        if route_weights_sk is not None:
            ok_w = torch.equal(route_weights_out.cpu(), route_weights_sk.cpu())
            ok_t = torch.tensor([int(ok_w)], dtype=torch.int32, device=hidden_out.device)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            assert int(ok_t.item()) == 1, f"{label} combine weights round-trip"

        if rank == 0:
            print(f"  [PASS] {label}")
        dist.barrier()
    finally:
        ShmemRuntime.finalize()
        if buffer is not None:
            buffer.destroy()
        reset_symmetric_descriptors()


@pytest.mark.parametrize("use_udma", [True, False], ids=["urma", "mte"])
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_buffer_combine(dist_env, case, use_udma):
    rank, local_rank, R = dist_env
    run_case(rank, local_rank, R, case, use_udma)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
