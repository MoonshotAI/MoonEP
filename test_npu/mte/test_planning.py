"""Planning operator correctness and profiling test.

By default all EP ranks are spawned on one host. For multi-node runs, launch
this script once per node with the same --ep-size and rendezvous endpoint, then
pass a unique --node-rank and the number of local processes via
--nproc-per-node.

Run (single node, default ``--bootstrap uid`` -> ``ShmemRuntime.init_with_buffer``;
the script spawns ``--ep-size`` ranks internally with mp.spawn):
    python test_npu/mte/test_planning.py --ep-size 8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from npu_patch._device import local_device_index
from npu_patch import Buffer, ShmemInstance, ShmemRuntime
from npu_patch.buffer_c import create_sym_dist_tensor
from npu_patch.launchers import (
    launch_planning,
    launch_planning_op,
    planning_workspace_sizes,
)
from npu_patch.planning import allocate_planning_outputs
from generate_topk_routing import generate_topk_routing


REFERENCE_ROUTE_SEARCH_WORKSPACE_BYTES = 32 * 1024 * 1024
REFERENCE_ROUTE_SEARCH_MAX_CHUNK = 256 * 1024
MAXVIO_WORKERS = 8


@dataclass(frozen=True)
class PlanningReference:
    dst: torch.Tensor
    raw_dst: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    # Source-dup outputs for dup_rank only. Each process validates its local
    # destination, so materializing all R destination rows is unnecessary.
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor
    dup_rank: int
    order: torch.Tensor
    tokens_per_expert: torch.Tensor
    tpe_prefix_by_rank: torch.Tensor
    balance: torch.Tensor
    z_tensor: torch.Tensor
    alloc_tensor: torch.Tensor
    alloc_cumsum: torch.Tensor
    expert_offsets: torch.Tensor
    NvS: int


def tokens_per_expert_from_topk(topk: torch.Tensor, num_experts: int) -> torch.Tensor:
    topk_cpu = topk.detach().cpu().to(torch.int64).contiguous()
    if topk_cpu.ndim != 3:
        raise ValueError(f"topk must be [R, S, K], got {tuple(topk_cpu.shape)}")
    world_size = int(topk_cpu.shape[0])
    tpe = torch.zeros((world_size, num_experts), dtype=torch.int64)
    for rank in range(world_size):
        tpe[rank] = torch.bincount(topk_cpu[rank].reshape(-1), minlength=num_experts)
    return tpe.to(torch.int32)


def map_maxvio_ranks(
    world_size: int,
    fn,
    executor: Optional[ThreadPoolExecutor] = None,
):
    ranks = range(world_size)
    if world_size <= 1:
        return [fn(rank) for rank in ranks]
    if executor is not None:
        return list(executor.map(fn, ranks))
    with ThreadPoolExecutor(max_workers=min(world_size, MAXVIO_WORKERS)) as pool:
        return list(pool.map(fn, ranks))


def maxvio_for_sigma(
    sigma: float,
    world_size: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    seed: int,
    executor: Optional[ThreadPoolExecutor] = None,
) -> float:
    def tpe_for_rank(rank: int) -> torch.Tensor:
        _, tpe = generate_topk_routing(
            num_tokens,
            num_topk,
            num_experts,
            world_size,
            sigma,
            "cpu",
            seed,
            rank=rank,
        )
        return tpe.to(torch.int64)

    total = torch.zeros(num_experts, dtype=torch.int64)
    for tpe in map_maxvio_ranks(world_size, tpe_for_rank, executor):
        total += tpe
    return float(total.max().item() / (total.sum().item() / num_experts) - 1.0)


def solve_sigma_for_maxvio(
    target: float,
    world_size: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    seed: int,
) -> Tuple[float, float]:
    lo, hi = 1e-5, 10.0
    with ThreadPoolExecutor(max_workers=min(world_size, MAXVIO_WORKERS)) as executor:
        mv_lo = maxvio_for_sigma(
            lo, world_size, num_tokens, num_topk, num_experts, seed, executor
        )
        if mv_lo >= target:
            return lo, mv_lo

        best_sigma, best_mv = lo, mv_lo
        for _ in range(60):
            mid = (lo * hi) ** 0.5
            mv = maxvio_for_sigma(
                mid,
                world_size,
                num_tokens,
                num_topk,
                num_experts,
                seed,
                executor,
            )
            if abs(mv - target) < abs(best_mv - target):
                best_sigma, best_mv = mid, mv
            if abs(mv - target) <= max(0.02 * target, 0.005):
                return mid, mv
            if mv < target:
                lo = mid
            else:
                hi = mid
    return best_sigma, best_mv


def make_maxvio_topk_case(
    world_size: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    sigma: float,
    seed: int,
) -> torch.Tensor:
    def topk_for_rank(rank: int) -> torch.Tensor:
        topk, _ = generate_topk_routing(
            num_tokens,
            num_topk,
            num_experts,
            world_size,
            sigma,
            "cpu",
            seed,
            rank=rank,
        )
        return topk

    return torch.stack(map_maxvio_ranks(world_size, topk_for_rank))


def make_topk_case(
    case: str,
    world_size: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    seed: int,
    maxvio_sigma: Optional[float] = None,
):
    epn = num_experts // world_size
    if case == "balanced":
        rank_base = torch.arange(world_size, dtype=torch.int32).view(-1, 1, 1) * epn
        local_experts = (
            torch.arange(num_tokens * num_topk, dtype=torch.int32)
            .remainder(epn)
            .view(1, num_tokens, num_topk)
        )
        return rank_base + local_experts

    if case == "skewed":
        return torch.zeros((world_size, num_tokens, num_topk), dtype=torch.int32)

    if case == "duplicate":
        rank_base = torch.arange(world_size, dtype=torch.int32).view(-1, 1, 1) * epn
        token_expert = rank_base + (
            torch.arange(num_tokens, dtype=torch.int32).view(1, -1, 1) % epn
        )
        kidx = torch.arange(num_topk, dtype=torch.int32).view(1, 1, -1)
        return torch.where(kidx < 2, token_expert, (token_expert + kidx) % num_experts)

    if case == "random":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(0, num_experts, (world_size, num_tokens, num_topk), generator=gen, dtype=torch.int32)

    if case == "maxvio":
        if maxvio_sigma is None:
            raise ValueError("case=maxvio requires a resolved maxvio sigma")
        return make_maxvio_topk_case(
            world_size,
            num_tokens,
            num_topk,
            num_experts,
            maxvio_sigma,
            seed,
        )

    raise ValueError(f"unknown planning case: {case}")


def make_local_topk_case(
    case: str,
    rank: int,
    world_size: int,
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    seed: int,
    maxvio_sigma: Optional[float] = None,
) -> torch.Tensor:
    """Build only one rank's input for profiling; no global golden is needed."""
    epn = num_experts // world_size
    if case == "balanced":
        values = torch.arange(num_tokens * num_topk, dtype=torch.int32) % epn
        return (values + rank * epn).view(num_tokens, num_topk)

    if case == "skewed":
        return torch.zeros((num_tokens, num_topk), dtype=torch.int32)

    if case == "duplicate":
        token_expert = (
            rank * epn
            + torch.arange(num_tokens, dtype=torch.int32).view(-1, 1) % epn
        )
        kidx = torch.arange(num_topk, dtype=torch.int32).view(1, -1)
        return torch.where(kidx < 2, token_expert, (token_expert + kidx) % num_experts)

    if case == "random":
        gen = torch.Generator(device="cpu").manual_seed(seed + rank)
        return torch.randint(
            0,
            num_experts,
            (num_tokens, num_topk),
            generator=gen,
            dtype=torch.int32,
        )

    if case == "maxvio":
        if maxvio_sigma is None:
            raise ValueError("case=maxvio requires a resolved maxvio sigma")
        topk, _ = generate_topk_routing(
            num_tokens,
            num_topk,
            num_experts,
            world_size,
            maxvio_sigma,
            "cpu",
            seed,
            rank=rank,
        )
        return topk

    raise ValueError(f"unknown planning case: {case}")


def build_route_reference(
    topk: torch.Tensor,
    tpe: torch.Tensor,
    tpe_prefix_by_rank: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    expert_offsets: torch.Tensor,
    NvS: int,
):
    world_size, num_tokens, num_topk = (int(v) for v in topk.shape)
    route_count = num_tokens * num_topk
    flat_topk = topk.reshape(world_size, route_count)

    # Stable expert bucketing gives the exact expert-major, route-stable order.
    order = torch.argsort(flat_topk, dim=1, stable=True)
    sorted_experts = flat_topk.gather(1, order)
    expert_starts = tpe.cumsum(dim=1) - tpe
    sorted_positions = torch.arange(route_count, dtype=torch.int64).unsqueeze(0)
    sorted_local_ordinals = (
        sorted_positions - expert_starts.gather(1, sorted_experts)
    )
    local_ordinals = torch.empty_like(order)
    local_ordinals.scatter_(1, order, sorted_local_ordinals)
    global_ordinals = (
        tpe_prefix_by_rank.gather(1, flat_topk) + local_ordinals
    )

    flat_experts = flat_topk.reshape(-1)
    flat_global_ordinals = global_ordinals.reshape(-1)
    raw_dst_flat = torch.empty_like(flat_global_ordinals)
    boundary_row_bytes = max(world_size, 1) * alloc_cumsum.element_size()
    route_chunk = max(
        1,
        min(
            REFERENCE_ROUTE_SEARCH_MAX_CHUNK,
            REFERENCE_ROUTE_SEARCH_WORKSPACE_BYTES // boundary_row_bytes,
        ),
    )
    for base in range(0, flat_experts.numel(), route_chunk):
        end = min(base + route_chunk, flat_experts.numel())
        experts = flat_experts[base:end]
        ordinals = flat_global_ordinals[base:end]
        boundaries = alloc_cumsum.index_select(0, experts)
        dest_ranks = torch.searchsorted(
            boundaries,
            ordinals.unsqueeze(1),
            right=True,
        ).squeeze(1)
        previous_indices = (dest_ranks - 1).clamp_min(0)
        previous_boundaries = boundaries.gather(
            1,
            previous_indices.unsqueeze(1),
        ).squeeze(1)
        previous_boundaries = torch.where(
            dest_ranks > 0,
            previous_boundaries,
            torch.zeros_like(previous_boundaries),
        )
        loffs = expert_offsets[dest_ranks, experts] + ordinals - previous_boundaries
        raw_dst_flat[base:end] = dest_ranks * NvS + loffs

    raw_dst = raw_dst_flat.reshape(world_size, num_tokens, num_topk)
    dest_ranks = torch.div(raw_dst, NvS, rounding_mode="floor")
    duplicate_mask = torch.zeros_like(dest_ranks, dtype=torch.bool)
    for kidx in range(1, num_topk):
        duplicate_mask[:, :, kidx] = (
            dest_ranks[:, :, :kidx] == dest_ranks[:, :, kidx:kidx + 1]
        ).any(dim=2)
    dst = torch.where(duplicate_mask, -raw_dst - 1, raw_dst)
    return order, raw_dst, dst


def build_dup_reference_for_rank(
    raw_dst_by_route: torch.Tensor,
    NvS: int,
    dup_rank: int,
):
    world_size = int(raw_dst_by_route.shape[0])
    if dup_rank < 0 or dup_rank >= world_size:
        raise ValueError(f"dup_rank must be in [0, {world_size}), got {dup_rank}")

    dest_ranks = torch.div(raw_dst_by_route, NvS, rounding_mode="floor")
    loffs = torch.remainder(raw_dst_by_route, NvS)
    local_mask = dest_ranks == dup_rank
    local_route_ordinals = local_mask.cumsum(dim=2)
    routes_per_group = local_mask.sum(dim=2)
    valid_groups = routes_per_group > 1
    duplicate_counts = routes_per_group[valid_groups] - 1
    group_count = int(duplicate_counts.numel())
    loff_count = int(duplicate_counts.sum().item())
    if group_count > NvS or loff_count > NvS:
        raise ValueError(
            "source-dup reference exceeds NvS capacity: "
            f"groups={group_count}, loffs={loff_count}, NvS={NvS}"
        )

    dup_groups = torch.zeros((NvS, 3), dtype=torch.int64)
    dup_loffs = torch.zeros(NvS, dtype=torch.int64)
    dup_counts = torch.tensor((group_count, loff_count), dtype=torch.int64)
    if group_count > 0:
        primary_indices = local_mask.to(torch.int64).argmax(dim=2)
        primary_loffs = loffs.gather(2, primary_indices.unsqueeze(2)).squeeze(2)
        loff_starts = duplicate_counts.cumsum(dim=0) - duplicate_counts

        dup_groups[:group_count, 0] = primary_loffs[valid_groups]
        dup_groups[:group_count, 1] = loff_starts
        dup_groups[:group_count, 2] = duplicate_counts
        duplicate_mask = local_mask & (local_route_ordinals > 1)
        dup_loffs[:loff_count] = loffs[duplicate_mask]

    return (
        dup_groups.to(torch.int32),
        dup_loffs.to(torch.int32),
        dup_counts.to(torch.int32),
    )


def planning_reference(
    topk: torch.Tensor,
    num_experts: int,
    num_prefetch_slots: int,
    NvS: int,
    dup_rank: int,
) -> PlanningReference:
    topk = topk.detach().cpu().to(torch.int64).contiguous()
    if topk.ndim != 3:
        raise ValueError(f"topk must be [R, S, K], got {tuple(topk.shape)}")

    world_size, num_tokens, num_topk = (int(v) for v in topk.shape)
    route_count = num_tokens * num_topk
    if dup_rank < 0 or dup_rank >= world_size:
        raise ValueError(f"dup_rank must be in [0, {world_size}), got {dup_rank}")
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    epn = num_experts // world_size
    if num_prefetch_slots != epn:
        raise ValueError("current Ascend planning kernel requires B == E / R")
    if NvS < route_count:
        raise ValueError("NvS must be at least S * K")

    tpe = tokens_per_expert_from_topk(topk.to(torch.int32), num_experts).to(torch.int64)
    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[-1]

    tpe_prefix_by_rank = torch.zeros((world_size, num_experts), dtype=torch.int64)
    for rank in range(1, world_size):
        tpe_prefix_by_rank[rank] = tpe_cumsum[rank - 1]

    balance = torch.empty(world_size, dtype=torch.int64)
    for owner in range(world_size):
        beg = owner * epn
        balance[owner] = expert_count[beg:beg + epn].sum() - route_count

    z_tensor = torch.zeros((world_size, world_size), dtype=torch.int64)
    while True:
        surplus_rank = int(torch.argmax(balance).item())
        deficit_rank = int(torch.argmin(balance).item())
        surplus = int(balance[surplus_rank].item())
        deficit = int(balance[deficit_rank].item())
        if surplus <= 0 or deficit >= 0:
            break
        move_tokens = -deficit
        z_tensor[surplus_rank, deficit_rank] = move_tokens
        balance[surplus_rank] -= move_tokens
        balance[deficit_rank] = 0

    alloc_by_expert = torch.zeros((num_experts, world_size), dtype=torch.int64)
    for expert in range(num_experts):
        alloc_by_expert[expert, expert // epn] = expert_count[expert]

    for owner in range(world_size):
        expert_base = owner * epn
        remaining = expert_count[expert_base:expert_base + epn].clone()
        quotas = z_tensor[owner].clone()
        while True:
            target_rank = int(torch.argmax(quotas).item())
            max_quota = int(quotas[target_rank].item())
            if max_quota <= 0:
                break
            local_expert = int(torch.argmax(remaining).item())
            max_remaining = int(remaining[local_expert].item())
            if max_remaining <= 0:
                break
            take = min(max_quota, max_remaining)
            expert = expert_base + local_expert
            alloc_by_expert[expert, target_rank] += take
            alloc_by_expert[expert, owner] -= take
            remaining[local_expert] -= take
            quotas[target_rank] -= take

    alloc_cumsum = alloc_by_expert.cumsum(dim=1)
    alloc_tensor = alloc_by_expert.t().contiguous()

    experts_to_copy = torch.full((world_size, num_prefetch_slots), -1, dtype=torch.int64)
    expert_offsets = torch.full((world_size, num_experts), -1, dtype=torch.int64)
    cu_seqlens = torch.zeros((world_size, num_experts + num_prefetch_slots), dtype=torch.int64)

    for dest_rank in range(world_size):
        local_start = dest_rank * epn
        local_end = local_start + epn
        remote_experts = [
            expert for expert in range(num_experts)
            if int(alloc_by_expert[expert, dest_rank].item()) > 0
            and not (local_start <= expert < local_end)
        ]
        remote_experts.sort(
            key=lambda expert: (int(alloc_by_expert[expert, dest_rank].item()), expert),
            reverse=True,
        )
        for slot, expert in enumerate(remote_experts[:num_prefetch_slots]):
            experts_to_copy[dest_rank, slot] = expert

        prefetched = set(remote_experts[:num_prefetch_slots])
        start = 0
        for group_idx in range(num_experts + num_prefetch_slots):
            token_count = 0
            expert_id = -1
            if group_idx < num_experts:
                if group_idx not in prefetched:
                    expert_id = group_idx
                    token_count = int(alloc_by_expert[group_idx, dest_rank].item())
            else:
                slot = group_idx - num_experts
                expert = int(experts_to_copy[dest_rank, slot].item())
                if expert >= 0:
                    expert_id = expert
                    token_count = int(alloc_by_expert[expert, dest_rank].item())

            start += token_count
            cu_seqlens[dest_rank, group_idx] = start
            if token_count > 0:
                expert_offsets[dest_rank, expert_id] = start - token_count

    order, raw_dst_3d, dst = build_route_reference(
        topk,
        tpe,
        tpe_prefix_by_rank,
        alloc_cumsum,
        expert_offsets,
        NvS,
    )
    raw_dst = raw_dst_3d.reshape(world_size, route_count)

    dup_groups, dup_loffs, dup_counts = build_dup_reference_for_rank(
        raw_dst_3d,
        NvS,
        dup_rank,
    )

    return PlanningReference(
        dst=dst.to(torch.int32),
        raw_dst=raw_dst.reshape(world_size, num_tokens, num_topk).to(torch.int32),
        cu_seqlens=cu_seqlens.to(torch.int32),
        experts_to_copy=experts_to_copy.to(torch.int32),
        dup_groups=dup_groups,
        dup_loffs=dup_loffs,
        dup_counts=dup_counts,
        dup_rank=dup_rank,
        order=order.to(torch.int32),
        tokens_per_expert=tpe.to(torch.int32),
        tpe_prefix_by_rank=tpe_prefix_by_rank.to(torch.int32),
        balance=balance.to(torch.int32),
        z_tensor=z_tensor.to(torch.int32),
        alloc_tensor=alloc_tensor.to(torch.int32),
        alloc_cumsum=alloc_cumsum.to(torch.int32),
        expert_offsets=expert_offsets.to(torch.int32),
        NvS=NvS,
    )


def assert_tensor_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_cpu = actual.detach().cpu().to(torch.int32).contiguous()
    expected_cpu = expected.detach().cpu().to(torch.int32).contiguous()
    if actual_cpu.shape != expected_cpu.shape:
        raise AssertionError(f"{name}: shape mismatch actual={tuple(actual_cpu.shape)} expected={tuple(expected_cpu.shape)}")
    if torch.equal(actual_cpu, expected_cpu):
        return

    diff = actual_cpu != expected_cpu
    first = torch.nonzero(diff, as_tuple=False)[0].tolist()
    first_idx = tuple(int(v) for v in first)
    raise AssertionError(
        f"{name}: mismatch at {first_idx}: "
        f"actual={int(actual_cpu[first_idx].item())}, expected={int(expected_cpu[first_idx].item())}"
    )


def planning_workspace_alloc_views(ctx):
    rank_count = int(ctx["R"])
    expert_count = int(ctx["E"])
    cursor = 0

    def add_workspace_slice(size: int) -> int:
        nonlocal cursor
        cursor = ((cursor + 7) // 8) * 8
        offset = cursor
        cursor += size
        return offset

    add_workspace_slice(rank_count * expert_count)  # tpe_all
    add_workspace_slice(expert_count)  # tpe_prefix
    add_workspace_slice(rank_count)  # balance
    add_workspace_slice(rank_count * rank_count)  # z
    alloc_offset = add_workspace_slice(rank_count * expert_count)
    alloc_cumsum_offset = add_workspace_slice(rank_count * expert_count)

    workspace = ctx["planning_workspace"]
    alloc = workspace[
        alloc_offset:alloc_offset + rank_count * expert_count
    ].reshape(rank_count, expert_count)
    alloc_cumsum = workspace[
        alloc_cumsum_offset:alloc_cumsum_offset + rank_count * expert_count
    ].reshape(expert_count, rank_count)
    return alloc, alloc_cumsum


def describe_experts_to_copy_mismatch(ctx, actual, reference) -> str:
    actual_cpu = actual.detach().cpu().to(torch.int32).contiguous()
    expected_cpu = reference.experts_to_copy.detach().cpu().to(torch.int32).contiguous()
    dest, slot = (
        int(value)
        for value in torch.nonzero(actual_cpu != expected_cpu, as_tuple=False)[0].tolist()
    )
    actual_expert = int(actual_cpu[dest, slot].item())
    expected_expert = int(expected_cpu[dest, slot].item())
    rank_count = int(ctx["R"])
    expert_count = int(ctx["E"])
    experts_per_rank = expert_count // rank_count

    cursor = 0

    def add_workspace_slice(size: int) -> int:
        nonlocal cursor
        cursor = ((cursor + 7) // 8) * 8
        offset = cursor
        cursor += size
        return offset

    tpe_all_offset = add_workspace_slice(rank_count * expert_count)
    add_workspace_slice(expert_count)  # tpe_prefix
    balance_offset = add_workspace_slice(rank_count)
    z_offset = add_workspace_slice(rank_count * rank_count)
    alloc_offset = add_workspace_slice(rank_count * expert_count)

    workspace = ctx["planning_workspace"]
    actual_tpe_all = (
        workspace[tpe_all_offset:tpe_all_offset + rank_count * expert_count]
        .detach()
        .cpu()
        .to(torch.int32)
        .reshape(rank_count, expert_count)
        .contiguous()
    )
    expected_tpe_all = (
        reference.tokens_per_expert.detach().cpu().to(torch.int32).contiguous()
    )
    tpe_diff = actual_tpe_all != expected_tpe_all
    tpe_mismatch_count = int(torch.count_nonzero(tpe_diff).item())
    if tpe_mismatch_count > 0:
        tpe_rank, tpe_expert = (
            int(value)
            for value in torch.nonzero(tpe_diff, as_tuple=False)[0].tolist()
        )
        first_tpe_values = (
            tpe_rank,
            tpe_expert,
            int(actual_tpe_all[tpe_rank, tpe_expert].item()),
            int(expected_tpe_all[tpe_rank, tpe_expert].item()),
        )
    else:
        first_tpe_values = (-1, -1, 0, 0)

    actual_balance = (
        workspace[balance_offset:balance_offset + rank_count]
        .detach()
        .cpu()
        .to(torch.int64)
        .contiguous()
    )
    route_count = int(ctx["S"]) * int(ctx["K"])
    balance_from_actual_tpe = (
        actual_tpe_all.to(torch.int64)
        .sum(dim=0)
        .reshape(rank_count, experts_per_rank)
        .sum(dim=1)
        - route_count
    )
    expected_initial_balance = (
        expected_tpe_all.to(torch.int64)
        .sum(dim=0)
        .reshape(rank_count, experts_per_rank)
        .sum(dim=1)
        - route_count
    )

    def build_z_from_balance(balance: torch.Tensor) -> torch.Tensor:
        work = balance.clone()
        result = torch.zeros((rank_count, rank_count), dtype=torch.int64)
        while True:
            surplus_rank = int(torch.argmax(work).item())
            deficit_rank = int(torch.argmin(work).item())
            surplus = int(work[surplus_rank].item())
            deficit = int(work[deficit_rank].item())
            if surplus <= 0 or deficit >= 0:
                break
            move_tokens = -deficit
            result[surplus_rank, deficit_rank] = move_tokens
            work[surplus_rank] -= move_tokens
            work[deficit_rank] = 0
        return result

    z_from_actual_balance = build_z_from_balance(actual_balance)
    actual_alloc = (
        workspace[
            alloc_offset + dest * expert_count:
            alloc_offset + (dest + 1) * expert_count
        ]
        .detach()
        .cpu()
        .to(torch.int32)
        .contiguous()
    )
    expected_alloc = reference.alloc_tensor[dest].to(torch.int32).contiguous()
    alloc_diff = actual_alloc != expected_alloc
    alloc_mismatch_count = int(torch.count_nonzero(alloc_diff).item())
    if alloc_mismatch_count > 0:
        first_alloc_mismatch = int(
            torch.nonzero(alloc_diff, as_tuple=False)[0].item()
        )
        first_alloc_values = (
            first_alloc_mismatch,
            int(actual_alloc[first_alloc_mismatch].item()),
            int(expected_alloc[first_alloc_mismatch].item()),
        )
    else:
        first_alloc_values = (-1, 0, 0)
    remote_alloc = actual_alloc.clone()
    local_begin = dest * experts_per_rank
    remote_alloc[local_begin:local_begin + experts_per_rank] = 0
    actual_best_count = int(remote_alloc.max().item())
    if actual_best_count > 0:
        actual_best_expert = int(
            torch.nonzero(
                remote_alloc == actual_best_count, as_tuple=False
            )[-1].item()
        )
    else:
        actual_best_expert = -1

    def count_at(row: torch.Tensor, expert: int) -> int:
        return int(row[expert].item()) if expert >= 0 else 0

    actual_z = (
        workspace[
            z_offset + dest * rank_count:
            z_offset + (dest + 1) * rank_count
        ]
        .detach()
        .cpu()
        .to(torch.int32)
        .tolist()
    )
    expected_z = reference.z_tensor[dest].to(torch.int32).tolist()
    return (
        f"diagnostic dest={dest}, slot={slot}, actual_expert={actual_expert}, "
        f"expected_expert={expected_expert}, "
        f"actual_alloc_top1=({actual_best_expert}, {actual_best_count}), "
        f"actual_count_at_actual={count_at(actual_alloc, actual_expert)}, "
        f"actual_count_at_expected={count_at(actual_alloc, expected_expert)}, "
        f"reference_count_at_expected={count_at(expected_alloc, expected_expert)}, "
        f"alloc_mismatch_count={alloc_mismatch_count}, "
        f"first_alloc_mismatch=(expert, actual, expected)={first_alloc_values}, "
        f"tpe_mismatch_count={tpe_mismatch_count}, "
        f"first_tpe_mismatch=(rank, expert, actual, expected)={first_tpe_values}, "
        f"actual_balance={actual_balance.tolist()}, "
        f"balance_from_actual_tpe={balance_from_actual_tpe.tolist()}, "
        f"expected_initial_balance={expected_initial_balance.tolist()}, "
        f"actual_z_row={actual_z}, "
        f"z_from_actual_balance_row={z_from_actual_balance[dest].tolist()}, "
        f"reference_z_row={expected_z}"
    )


def assert_dup_plan_equal(
    name: str,
    actual_groups: torch.Tensor,
    actual_loffs: torch.Tensor,
    actual_counts: torch.Tensor,
    expected_groups: torch.Tensor,
    expected_loffs: torch.Tensor,
    expected_counts: torch.Tensor,
) -> None:
    assert_tensor_equal(f"{name}/dup_counts", actual_counts, expected_counts)

    expected_counts_cpu = expected_counts.detach().cpu().to(torch.int32).contiguous()
    group_count = int(expected_counts_cpu[0].item())
    loff_count = int(expected_counts_cpu[1].item())
    try:
        assert_tensor_equal(
            f"{name}/dup_groups",
            actual_groups[:group_count],
            expected_groups[:group_count],
        )
    except AssertionError as error:
        actual_groups_cpu = (
            actual_groups[:group_count].detach().cpu().to(torch.int32).contiguous()
        )
        actual_loffs_cpu = (
            actual_loffs[:loff_count].detach().cpu().to(torch.int32).contiguous()
        )
        expected_groups_cpu = (
            expected_groups[:group_count].detach().cpu().to(torch.int32).contiguous()
        )
        expected_loffs_cpu = (
            expected_loffs[:loff_count].detach().cpu().to(torch.int32).contiguous()
        )

        def canonical_records(groups: torch.Tensor, loffs: torch.Tensor):
            records = []
            for primary, loff_begin, count in groups.tolist():
                duplicates = tuple(
                    int(value)
                    for value in loffs[loff_begin:loff_begin + count].tolist()
                )
                records.append((int(primary), duplicates))
            return sorted(records)

        semantic_equal = canonical_records(
            actual_groups_cpu, actual_loffs_cpu
        ) == canonical_records(expected_groups_cpu, expected_loffs_cpu)
        diff = actual_groups_cpu != expected_groups_cpu
        row, column = (int(value) for value in torch.nonzero(diff)[0].tolist())
        actual_value = int(actual_groups_cpu[row, column].item())
        expected_rows = []
        if column == 0:
            expected_rows = torch.nonzero(
                expected_groups_cpu[:, 0] == actual_value, as_tuple=False
            ).reshape(-1).tolist()
        raise AssertionError(
            f"{error}; semantic_records_equal={semantic_equal}, "
            f"actual_value_expected_rows={expected_rows}"
        ) from None
    assert_tensor_equal(
        f"{name}/dup_loffs",
        actual_loffs[:loff_count],
        expected_loffs[:loff_count],
    )


def selected_cases(case_arg: str) -> Iterable[str]:
    if case_arg == "all":
        return ("balanced", "skewed", "duplicate", "random")
    return (case_arg,)


def ensure_maxvio_args(args) -> None:
    if args.case != "maxvio":
        if args.maxvio is not None:
            raise ValueError("--maxvio requires --case maxvio")
        return
    if args.maxvio is None:
        raise ValueError("--case maxvio requires --maxvio")
    if args.maxvio < 0:
        raise ValueError("--maxvio must be non-negative")

    world_size = int(args.ep_size)
    num_experts = int(args.num_experts or (world_size * 2))
    if num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by --ep-size")
    if args.num_topk > num_experts:
        raise ValueError("--case maxvio requires --num-topk <= --num-experts")

    sigma, realized = solve_sigma_for_maxvio(
        float(args.maxvio),
        world_size,
        int(args.num_tokens),
        int(args.num_topk),
        num_experts,
        int(args.seed),
    )
    args.maxvio_sigma = sigma
    args.realized_maxvio = realized
    note = (
        ""
        if abs(realized - args.maxvio) <= max(0.05 * args.maxvio, 0.01)
        else " (target may be unreachable for this shape)"
    )
    print(
        f"resolved maxvio: target={args.maxvio:.6g}, "
        f"sigma={sigma:.6g}, realized={realized:.6g}{note}",
        flush=True,
    )


def compact_dup_output(
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
):
    counts_cpu = dup_counts.detach().cpu().to(torch.int32).contiguous()
    group_count = int(counts_cpu[0].item())
    loff_count = int(counts_cpu[1].item())
    return {
        "dup_counts": counts_cpu.clone(),
        "dup_groups": (
            dup_groups[:group_count]
            .detach()
            .cpu()
            .to(torch.int32)
            .contiguous()
            .clone()
        ),
        "dup_loffs": (
            dup_loffs[:loff_count]
            .detach()
            .cpu()
            .to(torch.int32)
            .contiguous()
            .clone()
        ),
    }


def build_golden_case(args, case: str):
    world_size = int(args.ep_size)
    num_experts = int(args.num_experts or (world_size * 2))
    if num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by --ep-size")
    epn = num_experts // world_size
    num_prefetch_slots = int(args.num_prefetch_slots or epn)
    if num_prefetch_slots != epn:
        raise ValueError(
            "current Planning op requires "
            "--num-prefetch-slots == num_experts / ep_size"
        )

    route_count = args.num_tokens * args.num_topk
    NvS = int(args.num_max_tokens_per_rank or route_count)
    if NvS != route_count:
        raise ValueError(
            "ascend-moonep Buffer uses exact NvS=S*K; "
            "--num-max-tokens-per-rank must equal num_tokens*num_topk"
        )

    topk_all = make_topk_case(
        case,
        world_size,
        args.num_tokens,
        args.num_topk,
        num_experts,
        args.seed,
        maxvio_sigma=getattr(args, "maxvio_sigma", None),
    )
    reference = planning_reference(
        topk_all,
        num_experts,
        num_prefetch_slots,
        NvS,
        dup_rank=0,
    )

    dup_by_rank = []
    raw_dst_by_route = reference.raw_dst.detach().cpu().to(torch.int64).contiguous()
    for dup_rank in range(world_size):
        dup_groups, dup_loffs, dup_counts = build_dup_reference_for_rank(
            raw_dst_by_route,
            NvS,
            dup_rank,
        )
        dup_payload = compact_dup_output(dup_groups, dup_loffs, dup_counts)
        dup_payload["rank"] = dup_rank
        dup_by_rank.append(dup_payload)

    meta = {
        "case": case,
        "ep_size": world_size,
        "num_tokens": int(args.num_tokens),
        "num_topk": int(args.num_topk),
        "num_experts": num_experts,
        "num_prefetch_slots": num_prefetch_slots,
        "num_max_tokens_per_rank": NvS,
        "seed": int(args.seed),
    }
    if case == "maxvio":
        meta.update(
            {
                "target_maxvio": float(args.maxvio),
                "sigma": float(args.maxvio_sigma),
                "realized_maxvio": float(args.realized_maxvio),
            }
        )

    return {
        "meta": meta,
        "outputs": {
            "dst": reference.dst.detach().cpu().to(torch.int32).contiguous(),
            "cu_seqlens": (
                reference.cu_seqlens.detach().cpu().to(torch.int32).contiguous()
            ),
            "experts_to_copy": (
                reference.experts_to_copy.detach().cpu().to(torch.int32).contiguous()
            ),
            "dup_by_rank": dup_by_rank,
        },
    }


def write_tensor_bin(output_root: str, relative_path: str, tensor: torch.Tensor):
    output_path = os.path.join(output_root, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tensor_cpu = tensor.detach().cpu().to(torch.int32).contiguous()
    tensor_cpu.numpy().tofile(output_path)
    return {
        "file": relative_path.replace(os.sep, "/"),
        "dtype": "int32",
        "shape": [int(dim) for dim in tensor_cpu.shape],
        "numel": int(tensor_cpu.numel()),
    }


def write_golden_output(args) -> None:
    output_dir = os.path.abspath(args.golden_output)
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "format": "ascend-moonep.test_planning.golden.bin.v1",
        "cases": {},
    }
    for case in selected_cases(args.case):
        print(f"building planning golden for case={case}...", flush=True)
        golden = build_golden_case(args, case)
        meta = golden["meta"]
        outputs = golden["outputs"]
        case_manifest = {
            "meta": meta,
            "ranks": [],
        }
        for rank in range(int(meta["ep_size"])):
            rank_dir = os.path.join(case, f"rank_{rank:05d}")
            dup_payload = outputs["dup_by_rank"][rank]
            rank_manifest = {
                "rank": rank,
                "outputs": {
                    "dst": write_tensor_bin(
                        output_dir,
                        os.path.join(rank_dir, "dst.bin"),
                        outputs["dst"][rank].reshape(-1),
                    ),
                    "cu_seqlens": write_tensor_bin(
                        output_dir,
                        os.path.join(rank_dir, "cu_seqlens.bin"),
                        outputs["cu_seqlens"][rank],
                    ),
                    "experts_to_copy": write_tensor_bin(
                        output_dir,
                        os.path.join(rank_dir, "experts_to_copy.bin"),
                        outputs["experts_to_copy"],
                    ),
                    "dup_counts": write_tensor_bin(
                        output_dir,
                        os.path.join(rank_dir, "dup_counts.bin"),
                        dup_payload["dup_counts"],
                    ),
                    "dup_groups": write_tensor_bin(
                        output_dir,
                        os.path.join(rank_dir, "dup_groups.bin"),
                        dup_payload["dup_groups"],
                    ),
                    "dup_loffs": write_tensor_bin(
                        output_dir,
                        os.path.join(rank_dir, "dup_loffs.bin"),
                        dup_payload["dup_loffs"],
                    ),
                },
            }
            case_manifest["ranks"].append(rank_manifest)
        manifest["cases"][case] = case_manifest

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    print(f"planning golden bin files written to {output_dir}", flush=True)


def validate_launch_args(args) -> None:
    if not args.master_addr:
        raise ValueError("--master-addr must not be empty")
    if args.master_port <= 0 or args.master_port > 65535:
        raise ValueError("--master-port must be in [1, 65535]")
    if args.nproc_per_node is None:
        if args.node_rank != 0:
            raise ValueError("--node-rank requires --nproc-per-node")
        return
    if args.nproc_per_node <= 0:
        raise ValueError("--nproc-per-node must be greater than 0")
    if args.ep_size % args.nproc_per_node != 0:
        raise ValueError("--ep-size must be divisible by --nproc-per-node")

    num_nodes = args.ep_size // args.nproc_per_node
    if args.node_rank < 0 or args.node_rank >= num_nodes:
        raise ValueError(f"--node-rank must be in [0, {num_nodes})")


def global_rank_for_local(local_rank: int, args) -> int:
    if args.nproc_per_node is None:
        return local_rank
    return args.node_rank * args.nproc_per_node + local_rank


def local_process_count(args) -> int:
    if args.nproc_per_node is None:
        return args.ep_size
    return args.nproc_per_node


def effective_shmem_ip_port(args) -> str:
    if args.shmem_ip_port:
        return args.shmem_ip_port
    shmem_port = args.master_port + 1
    if shmem_port > 65535:
        raise ValueError(
            "--shmem-ip-port is required when --master-port is 65535"
        )
    return f"tcp://{args.master_addr}:{shmem_port}"


class IpPortPlanningOp:
    """Standalone Planning launcher backed by the ip-port SHMEM heap."""

    def __init__(self, workspace, workspace_bytes: int, num_sms: int):
        self.workspace = workspace
        self.workspace_bytes = int(workspace_bytes)
        self.num_sms = int(num_sms)

    def compute(
        self,
        ctx,
        topk_flat,
        tokens_per_expert,
        cu_seqlens,
        plan,
        shmem_instance_id=0,
    ) -> None:
        del tokens_per_expert, shmem_instance_id
        launch_planning_op(
            topk_flat,
            plan.dst,
            cu_seqlens,
            plan.experts_to_copy,
            plan.dup_groups,
            plan.dup_loffs,
            plan.dup_counts,
            ctx["planning_workspace"],
            int(self.workspace.data_ptr()),
            self.workspace_bytes,
            int(ctx["S"]),
            int(ctx["K"]),
            int(ctx["E"]),
            int(ctx["NvS"]),
            int(ctx["B"]),
            self.num_sms,
            int(ctx["rank"]),
            int(ctx["R"]),
        )

    def destroy(self) -> None:
        self.workspace = None


def create_ipport_planning_context(rank: int, args, shmem):
    world_size = int(args.ep_size)
    num_experts = int(args.num_experts or (world_size * 2))
    num_prefetch_slots = int(
        args.num_prefetch_slots or (num_experts // world_size)
    )
    NvS = int(args.num_max_tokens_per_rank or (args.num_tokens * args.num_topk))
    local_elements, symmetric_bytes = planning_workspace_sizes(
        args.num_tokens,
        args.num_topk,
        num_experts,
        NvS,
        world_size,
        args.core_num,
    )
    workspace = create_sym_dist_tensor(
        [symmetric_bytes], torch.uint8, torch.npu.current_device()
    )
    planning_op = IpPortPlanningOp(workspace, symmetric_bytes, args.core_num)
    device = f"npu:{torch.npu.current_device()}"
    ctx = {
        "rank": rank,
        "R": world_size,
        "E": num_experts,
        "S": int(args.num_tokens),
        "K": int(args.num_topk),
        "B": num_prefetch_slots,
        "NvS": NvS,
        "weights_buf": torch.empty(0, dtype=torch.int32, device=device),
        "planning_workspace": torch.empty(
            local_elements, dtype=torch.int32, device=device
        ),
        "kernels": {"planning": planning_op},
    }
    return ctx, planning_op


def compute_planning(ctx, topk_local, tpe_local, cu_seqlens, plan) -> None:
    launch_planning(
        ctx,
        topk_local.reshape(-1),
        tpe_local,
        cu_seqlens,
        plan,
    )


def run_profile(
    rank: int,
    ctx,
    args,
    case: str,
    topk_local: torch.Tensor,
    tpe_local: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan,
) -> None:
    os.makedirs(args.profile_dir, exist_ok=True)

    for _ in range(args.profile_warmup):
        compute_planning(ctx, topk_local, tpe_local, cu_seqlens, plan)
    torch_npu.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[
            torch_npu.profiler.ExportType.Text,
        ],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=True,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False,
    )

    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=1, active=1, repeat=1, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )

    prof.start()
    # One profiler warmup step followed by one active step. Each step contains
    # profile_iters kernel launches, matching the original Planning test.
    for _ in range(2):
        for _ in range(args.profile_iters):
            compute_planning(ctx, topk_local, tpe_local, cu_seqlens, plan)
        prof.step()
    prof.stop()

    if rank == 0:
        print(
            f"profile case {case} written to {args.profile_dir}: "
            f"iters_per_step={args.profile_iters}"
        )


def run_case(rank: int, ctx, args, case: str) -> None:
    world_size = int(args.ep_size)
    num_experts = int(args.num_experts or (world_size * 2))
    if num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by --ep-size")
    epn = num_experts // world_size
    num_prefetch_slots = int(args.num_prefetch_slots or epn)
    if num_prefetch_slots != epn:
        raise ValueError("current Planning op requires --num-prefetch-slots == num_experts / ep_size")

    route_count = args.num_tokens * args.num_topk
    NvS = int(args.num_max_tokens_per_rank or route_count)
    if NvS != route_count:
        raise ValueError(
            "ascend-moonep Buffer uses exact NvS=S*K; "
            "--num-max-tokens-per-rank must equal num_tokens*num_topk"
        )

    topk_all = None
    if args.profile or args.skip_assert:
        topk_local_cpu = make_local_topk_case(
            case,
            rank,
            world_size,
            args.num_tokens,
            args.num_topk,
            num_experts,
            args.seed,
            maxvio_sigma=getattr(args, "maxvio_sigma", None),
        )
    else:
        topk_all = make_topk_case(
            case,
            world_size,
            args.num_tokens,
            args.num_topk,
            num_experts,
            args.seed,
            maxvio_sigma=getattr(args, "maxvio_sigma", None),
        )
        topk_local_cpu = topk_all[rank]
    topk_local = topk_local_cpu.to(device="npu", dtype=torch.int32).contiguous()
    # Kept only for MoonEP launcher compatibility. Passing zeros deliberately
    # verifies that Planning derives the real TPE locally from topk_flat.
    tpe_local = torch.zeros(num_experts, device="npu", dtype=torch.int32)
    plan, cu_seqlens = allocate_planning_outputs(ctx)

    if args.profile:
        run_profile(
            rank,
            ctx,
            args,
            case,
            topk_local,
            tpe_local,
            cu_seqlens,
            plan,
        )
        return

    if args.skip_assert:
        print(
            f"rank {rank}: launching planning without golden checks for "
            f"{args.stress_iters} iteration(s)...",
            flush=True,
        )
        for _ in range(args.stress_iters):
            compute_planning(ctx, topk_local, tpe_local, cu_seqlens, plan)
        torch_npu.npu.synchronize()
        if rank == 0:
            print(
                f"planning case {case} completed without assertions: "
                f"R={world_size} S={args.num_tokens} K={args.num_topk} "
                f"E={num_experts} B={num_prefetch_slots} "
                f"iterations={args.stress_iters}",
                flush=True,
            )
        return

    print(
        f"rank {rank}: launching planning before building the reference...",
        flush=True,
    )
    compute_planning(ctx, topk_local, tpe_local, cu_seqlens, plan)
    torch_npu.npu.synchronize()

    print(f"rank {rank}: building planning reference...", flush=True)

    assert topk_all is not None
    reference = planning_reference(
        topk_all,
        num_experts,
        num_prefetch_slots,
        NvS,
        dup_rank=rank,
    )

    first_stress_failure = None
    expected_dst_npu = None
    expected_dup_counts_npu = None
    expected_dup_groups_npu = None
    expected_dup_loffs_npu = None
    stress_group_count = 0
    stress_loff_count = 0
    if args.stress_iters > 1:
        expected_dst_npu = (
            reference.dst[rank]
            .reshape(-1)
            .to(device=plan.dst.device, dtype=plan.dst.dtype)
        )
        expected_dup_counts_npu = reference.dup_counts.to(
            device=plan.dup_counts.device, dtype=plan.dup_counts.dtype
        )
        stress_group_count = int(reference.dup_counts[0].item())
        stress_loff_count = int(reference.dup_counts[1].item())
        expected_dup_groups_npu = reference.dup_groups[:stress_group_count].to(
            device=plan.dup_groups.device, dtype=plan.dup_groups.dtype
        )
        expected_dup_loffs_npu = reference.dup_loffs[:stress_loff_count].to(
            device=plan.dup_loffs.device, dtype=plan.dup_loffs.dtype
        )

    for iteration in range(args.stress_iters):
        if iteration > 0:
            compute_planning(ctx, topk_local, tpe_local, cu_seqlens, plan)
        if expected_dst_npu is None:
            continue

        mismatch_counts = torch.stack(
            (
                torch.count_nonzero(plan.dst != expected_dst_npu),
                torch.count_nonzero(plan.dup_counts != expected_dup_counts_npu),
                torch.count_nonzero(
                    plan.dup_groups[:stress_group_count]
                    != expected_dup_groups_npu
                ),
                torch.count_nonzero(
                    plan.dup_loffs[:stress_loff_count] != expected_dup_loffs_npu
                ),
            )
        ).cpu().tolist()
        if any(mismatch_counts) and first_stress_failure is None:
            dst_mismatches, count_mismatches, group_mismatches, loff_mismatches = (
                int(value) for value in mismatch_counts
            )
            if dst_mismatches == 0:
                try:
                    assert_dup_plan_equal(
                        f"{case}/rank{rank}",
                        plan.dup_groups,
                        plan.dup_loffs,
                        plan.dup_counts,
                        reference.dup_groups,
                        reference.dup_loffs,
                        reference.dup_counts,
                    )
                except AssertionError as error:
                    first_stress_failure = (
                        f"dup stress mismatch at iteration={iteration}, "
                        f"counts=({count_mismatches}, {group_mismatches}, "
                        f"{loff_mismatches}): {error}"
                    )
                continue

            actual_cpu = plan.dst.detach().cpu().to(torch.int32).contiguous()
            expected_cpu = reference.dst[rank].reshape(-1).to(torch.int32)
            first_index = int(
                torch.nonzero(actual_cpu != expected_cpu, as_tuple=False)[0].item()
            )
            actual_value = int(actual_cpu[first_index].item())
            expected_value = int(expected_cpu[first_index].item())
            expert = int(topk_local_cpu.reshape(-1)[first_index].item())
            tpe_prefix_workspace_offset = (
                (int(ctx["R"]) * int(ctx["E"]) + 7) // 8
            ) * 8
            actual_tpe_prefix = int(
                ctx["planning_workspace"][
                    tpe_prefix_workspace_offset + expert
                ].item()
            )
            workspace_cursor = 0
            for workspace_size in (
                int(ctx["R"]) * int(ctx["E"]),
                int(ctx["E"]),
                int(ctx["R"]),
                int(ctx["R"]) * int(ctx["R"]),
                int(ctx["R"]) * int(ctx["E"]),
                int(ctx["R"]) * int(ctx["E"]),
                int(ctx["R"]) * int(ctx["E"]),
            ):
                workspace_cursor = ((workspace_cursor + 7) // 8) * 8
                workspace_cursor += workspace_size
            core_counts_workspace_offset = ((workspace_cursor + 7) // 8) * 8
            route_core_num = min(int(args.core_num), 56)
            items_per_core = (
                int(ctx["S"]) * int(ctx["K"]) + route_core_num - 1
            ) // route_core_num
            owner_core = min(first_index // items_per_core, route_core_num - 1)
            core0_cursor = int(
                ctx["planning_workspace"][
                    core_counts_workspace_offset + expert
                ].item()
            )
            owner_cursor = int(
                ctx["planning_workspace"][
                    core_counts_workspace_offset
                    + owner_core * int(ctx["E"])
                    + expert
                ].item()
            )
            actual_raw = -actual_value - 1 if actual_value < 0 else actual_value
            expected_raw = (
                -expected_value - 1 if expected_value < 0 else expected_value
            )
            first_stress_failure = (
                f"{case}/rank{rank}/dst stress mismatch: iteration={iteration}, "
                f"mismatch_count={dst_mismatches}, index={first_index}, "
                f"token={first_index // args.num_topk}, "
                f"kidx={first_index % args.num_topk}, expert={expert}, "
                f"actual={actual_value} "
                f"(rank={actual_raw // NvS}, loff={actual_raw % NvS}), "
                f"expected={expected_value} "
                f"(rank={expected_raw // NvS}, loff={expected_raw % NvS}), "
                f"actual_tpe_prefix={actual_tpe_prefix}, "
                f"expected_tpe_prefix="
                f"{int(reference.tpe_prefix_by_rank[rank, expert].item())}, "
                f"owner_core={owner_core}, core0_cursor={core0_cursor}, "
                f"owner_cursor={owner_cursor}, "
                f"core_local_prefix={owner_cursor - core0_cursor}, "
                f"expert_tpe_by_rank="
                f"{reference.tokens_per_expert[:, expert].tolist()}"
            )

    torch_npu.npu.synchronize()

    print(f"rank {rank}: checking planning outputs...", flush=True)

    check_error = None
    try:
        actual_alloc, actual_alloc_cumsum = planning_workspace_alloc_views(ctx)
        assert_tensor_equal(
            f"{case}/rank{rank}/alloc",
            actual_alloc,
            reference.alloc_tensor,
        )
        assert_tensor_equal(
            f"{case}/rank{rank}/alloc_cumsum",
            actual_alloc_cumsum,
            reference.alloc_cumsum,
        )
        try:
            assert_tensor_equal(
                f"{case}/rank{rank}/experts_to_copy",
                plan.experts_to_copy,
                reference.experts_to_copy,
            )
        except AssertionError as error:
            diagnostic = describe_experts_to_copy_mismatch(
                ctx, plan.experts_to_copy, reference
            )
            raise AssertionError(f"{error}; {diagnostic}") from None
        assert_tensor_equal(
            f"{case}/rank{rank}/cu_seqlens",
            cu_seqlens,
            reference.cu_seqlens[rank],
        )
        assert_tensor_equal(
            f"{case}/rank{rank}/dst",
            plan.dst,
            reference.dst[rank].reshape(-1),
        )
        assert_dup_plan_equal(
            f"{case}/rank{rank}",
            plan.dup_groups,
            plan.dup_loffs,
            plan.dup_counts,
            reference.dup_groups,
            reference.dup_loffs,
            reference.dup_counts,
        )
    except Exception as error:
        check_error = error
    finally:
        # tokens_per_expert, zero_fill_ranges and remote_stats are compatibility
        # fields in ascend-moonep and are intentionally not precision-checked.
        check_status = "failed" if check_error is not None else "passed"
        print(
            f"rank {rank}: local planning output checks {check_status}",
            flush=True,
        )

    if first_stress_failure is not None:
        raise AssertionError(first_stress_failure)
    if check_error is not None:
        raise check_error
    if rank == 0:
        print(
            f"planning case {case} passed: "
            f"R={world_size} S={args.num_tokens} K={args.num_topk} "
            f"E={num_experts} B={num_prefetch_slots} iterations={args.stress_iters}"
        )


def run_rank(local_rank: int, args) -> None:
    world_size = int(args.ep_size)
    rank = global_rank_for_local(local_rank, args)
    device_id = local_device_index(local_rank)

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    # Planning does not use dispatch/combine symmetric workspaces. Avoid
    # reserving the default 32 MiB in this dedicated test.
    os.environ["MOONEP_ASCEND_OP_WS_BYTES"] = "0"

    torch.npu.set_device(device_id)

    buffer = None
    planning_ctx = None
    standalone_planning_op = None
    shmem = None
    try:
        num_experts = int(args.num_experts or (world_size * 2))
        if args.bootstrap == "ipport":
            mem_size = ShmemRuntime.compute_mem_size(
                S=args.num_tokens,
                H=8,
                K=args.num_topk,
                E=num_experts,
                H_prime=8,
                num_ep_ranks=world_size,
                op_ws_bytes=0,
                num_reduce_buffers=0,
                num_prefetch_slots=0,
            )
            shmem = ShmemInstance(
                rank, world_size, effective_shmem_ip_port(args)
            )
            shmem.init(mem_size)
            planning_ctx, standalone_planning_op = (
                create_ipport_planning_context(rank, args, shmem)
            )
            print("shmem ipport init success")
            for case in selected_cases(args.case):
                run_case(rank, planning_ctx, args, case)
            return

        # Buffer-backed SHMEM exchanges its UID over the default HCCL group.
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=rank,
            world_size=world_size,
        )
        # Buffer-backed only: construct Buffer first (registers the VMM descriptor), then init_with_buffer.
        buffer = Buffer(
            args.num_tokens,
            8,
            args.num_topk,
            num_experts,
            world_size,
            num_sms=args.core_num,
            B=num_experts // world_size,
            explicitly_destroy=True,
        )
        if not ShmemRuntime.is_initialized():
            init_shmem(args=args, group=None)

        print("shmem.init_with_buffer success")

        for case in selected_cases(args.case):
            run_case(rank, buffer._require_ctx(), args, case)
    finally:
        # Buffer-backed only: finalize first (tears down the symmetric mapping), then buffer.destroy (VMM inverse operations).
        if standalone_planning_op is not None:
            torch_npu.npu.synchronize()
            standalone_planning_op.destroy()
            if planning_ctx is not None:
                planning_ctx.clear()
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        if buffer is not None:
            buffer.destroy()
        if shmem is not None:
            shmem.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ascend-moonep Planning correctness and profiling test"
    )
    parser.add_argument("--ep-size", type=int, default=8, help="Global number of worker ranks.")
    parser.add_argument("--first-device", type=int, default=0, help="First local NPU device id used on this node. Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Zero-based node index; requires --nproc-per-node.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Enable multi-node mode and spawn this many local NPU processes.",
    )
    parser.add_argument(
        "--bootstrap",
        choices=("uid", "ipport"),
        default="uid",
        help="SHMEM bootstrap; ipport does not initialize torch.distributed.",
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="Process-group rendezvous host; use the rank-0 host address on every node.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "28889")),
        help="HCCL rendezvous port shared by every rank.",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=None,
        help=(
            "SHMEM ipport endpoint, for example tcp://10.0.0.1:28890; "
            "defaults to tcp://MASTER_ADDR:(MASTER_PORT+1)."
        ),
    )
    parser.add_argument("--num-tokens", type=int, default=16, help="Tokens per rank.")
    parser.add_argument("--num-topk", type=int, default=2, help="Top-k per token.")
    parser.add_argument("--num-experts", type=int, default=None, help="Global expert count. Default: 2 * ep_size.")
    parser.add_argument(
        "--core-num",
        type=int,
        default=32,
        help="Planning kernel core/block dim. Must be in [8, 56].",
    )
    parser.add_argument(
        "--num-prefetch-slots",
        type=int,
        default=None,
        help="Prefetch slots per rank. Default: num_experts / ep_size.",
    )
    parser.add_argument(
        "--num-max-tokens-per-rank",
        type=int,
        default=None,
        help="Planning NvS. Default: num_tokens * num_topk; Ascend planning does not add token padding.",
    )
    parser.add_argument(
        "--case",
        choices=("all", "balanced", "skewed", "duplicate", "random", "maxvio"),
        default="all",
        help="Routing case to test.",
    )
    parser.add_argument(
        "--maxvio",
        type=float,
        default=None,
        help="Target max violation for --case maxvio, computed as max(global_tpe) / avg(global_tpe) - 1.",
    )
    parser.add_argument(
        "--golden-output",
        "--golden-output-dir",
        dest="golden_output",
        default=None,
        help=(
            "Write CPU planning golden .bin files to this directory and exit before "
            "SHMEM initialization or kernel launch."
        ),
    )
    parser.add_argument(
        "--prod-shape",
        action="store_true",
        help="Production 8-card config: S=4096, K=16, E=896, B=112 (requires --ep-size 8).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random routing case.")
    parser.add_argument(
        "--stress-iters",
        type=int,
        default=1,
        help=(
            "Run and check Planning repeatedly after one initialization; "
            "the golden result is built only once."
        ),
    )
    parser.add_argument(
        "--skip-assert",
        "--skip_assert",
        dest="skip_assert",
        action="store_true",
        help=(
            "Run Planning with locally generated input without building the "
            "golden reference or checking outputs."
        ),
    )
    parser.add_argument("--profile", action="store_true", help="Enable torch_npu profiler for planning.")
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=5,
        help="Unrecorded Planning warmup iterations.",
    )
    parser.add_argument(
        "--profile-dir",
        default="./profile",
        help="Profiler output directory.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=10,
        help="Number of planning calls per profiler step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prod_shape:
        if args.ep_size != 8:
            raise ValueError("--prod-shape requires --ep-size 8 (production 8-card config)")
        args.num_tokens = 4096
        args.num_topk = 16
        args.num_experts = 896
        args.num_prefetch_slots = 112
        args.case = "balanced"
    if args.ep_size < 1:
        raise ValueError("--ep-size must be greater than 0")
    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be greater than 0")
    if args.num_topk <= 0:
        raise ValueError("--num-topk must be greater than 0")
    ensure_maxvio_args(args)
    if args.golden_output:
        write_golden_output(args)
        return
    if args.stress_iters <= 0:
        raise ValueError("--stress-iters must be greater than 0")
    if args.stress_iters > 1 and args.case == "all":
        raise ValueError("--stress-iters greater than 1 requires one explicit --case")
    if args.stress_iters > 1 and args.profile:
        raise ValueError("--stress-iters and --profile cannot be enabled together")
    if args.profile_iters <= 0:
        raise ValueError("--profile-iters must be greater than 0")
    if args.profile_warmup < 0:
        raise ValueError("--profile-warmup must be non-negative")
    if args.core_num < 8 or args.core_num > 56:
        raise ValueError("--core-num must be in [8, 56]")
    validate_launch_args(args)

    local_count = local_process_count(args)
    first_global_rank = global_rank_for_local(0, args)
    last_global_rank = global_rank_for_local(local_count - 1, args)

    print(
        f"Launching planning test with ep_size={args.ep_size}, node_rank={args.node_rank}, "
        f"local_processes={local_count}, global_ranks={first_global_rank}-{last_global_rank}, "
        f"first_device={args.first_device}, case={args.case}, core_num={args.core_num}, "
        f"bootstrap={args.bootstrap}, master={args.master_addr}:{args.master_port}, "
        f"shmem_ip_port={effective_shmem_ip_port(args) if args.bootstrap == 'ipport' else 'unused'}, "
        f"skip_assert={args.skip_assert}"
    )
    if local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)

    if args.nproc_per_node is not None:
        print(f"Node {args.node_rank} local ranks completed")


if __name__ == "__main__":
    main()

# python test_npu/mte/test_planning.py --ep-size 8 --case all
# python test_npu/mte/test_planning.py --ep-size 64 --num-tokens 8192 --num-topk 8 --num-experts 384 --case random --golden-output planning_golden
# python test_npu/mte/test_planning.py --ep-size 64 --num-tokens 8192 --num-topk 8 --num-experts 384 --case maxvio --maxvio 1.0
# Two-node example (run one command on each node with the same master endpoint):
# node 0: python test_npu/mte/test_planning.py --ep-size 16 --nproc-per-node 8 --node-rank 0 --master-addr 10.0.0.1
# node 1: python test_npu/mte/test_planning.py --ep-size 16 --nproc-per-node 8 --node-rank 1 --master-addr 10.0.0.1
