"""Plan-driven MoonEP combine + prologue test.

Mirrors test_moonep_dispatch_with_plan.py. Uses the same make_plan contract
(dst / dup_groups / dup_loffs / dup_counts), supports deterministic all-to-all
peer regions, dispatch's cyclic plan, and a seeded random permutation, and
verifies:

  - golden: CPU prologue + gather reference
  - staged kernel combine output against the CPU reference
  - optional weight gather invertibility

By default all R ranks are spawned on one host.

Examples:

  python test_npu/urma/test_moonep_combine_udma_with_plan.py --mode kernel \
      --R 8 --S 8192 --K 8 --H 7168 --E 384 --plan-mode cyclic
  python test_npu/urma/test_moonep_combine_udma_with_plan.py --mode kernel \
      --R 16 --S 8192 --K 8 --H 7168 --E 384 --plan-mode cyclic \
      --nproc-per-node 8 --node-rank 0
"""
from __future__ import annotations


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import argparse
import gc
import os
import sys
from dataclasses import dataclass

import torch
import torch.multiprocessing as mp

# Keep the cyclic plan bit-for-bit identical to the dispatch+combine joint
# test. The joint test imports this same module for its dst/duplicate plan.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moonep_dispatch_udma_with_plan as dispatch_test

MAX_UB_ROW_BYTES = 128 * 1024
SEED = 20260803


def validate_args(args) -> None:
    if args.R <= 0 or args.S <= 0 or args.K <= 0 or args.H <= 0 or args.E <= 0:
        raise ValueError("R/S/K/H/E must be positive")
    if args.B < 0:
        raise ValueError("B must be non-negative")
    if args.H % 8 != 0:
        raise ValueError("H must be multiple of 8")
    if args.H * 2 > MAX_UB_ROW_BYTES:
        raise ValueError(f"H is too large for UB row buffer: H*2={args.H * 2} > {MAX_UB_ROW_BYTES}")
    if args.R * args.S * args.K >= 2**31:
        raise ValueError("R*S*K must fit in positive int32 dst encoding")
    if args.nproc_per_node is None:
        if args.node_rank != 0:
            raise ValueError("--node-rank requires --nproc-per-node")
        return
    if args.nproc_per_node <= 0:
        raise ValueError("nproc-per-node must be positive")
    if args.R % args.nproc_per_node != 0:
        raise ValueError("R must be divisible by nproc-per-node")
    num_nodes = args.R // args.nproc_per_node
    if args.node_rank < 0 or args.node_rank >= num_nodes:
        raise ValueError(f"node-rank must be in [0, {num_nodes})")


def global_rank_for_local(local_rank: int, args) -> int:
    if args.nproc_per_node is None:
        return local_rank
    return args.node_rank * args.nproc_per_node + local_rank


def local_process_count(args) -> int:
    return args.R if args.nproc_per_node is None else args.nproc_per_node


def num_output_slots(args) -> int:
    return args.S * args.K


@dataclass(frozen=True)
class RouteEntry:
    src_rank: int
    token_id: int
    topk_id: int


def make_hidden(rank: int, args) -> torch.Tensor:
    row_base = torch.arange(args.S, dtype=torch.float32).unsqueeze(1) + float(rank * args.S)
    col = (torch.arange(args.H, dtype=torch.float32) % 16).unsqueeze(0) / 64.0
    return (row_base + col).to(torch.bfloat16).contiguous()


def expected_hidden_row(src_rank: int, token_id: int, args) -> torch.Tensor:
    base = float(src_rank * args.S + token_id)
    col = (torch.arange(args.H, dtype=torch.float32) % 16) / 64.0
    return (base + col).to(torch.bfloat16)


def make_weights(rank: int, args) -> torch.Tensor:
    base = rank * args.S * args.K
    return (torch.arange(args.S * args.K, dtype=torch.float32) + float(base)).reshape(args.S, args.K).contiguous()


def make_all_to_all_destinations(args) -> torch.Tensor:
    """Build the same peer-region layout used by udma_aggregate_perf.

    A source rank stores routes in ascending destination-peer regions. A
    destination rank stores incoming routes in ascending source-peer regions.
    Self routes are excluded when R > 1. When ``slots`` is not divisible by
    ``R - 1``, cyclic peer distance distributes the remainder while preserving
    exactly ``slots`` outgoing and incoming routes on every rank.
    """
    slots = num_output_slots(args)
    if args.R == 1:
        return torch.arange(slots, dtype=torch.int32)

    base_count, remainder = divmod(slots, args.R - 1)

    def route_count(src_rank: int, dest_rank: int) -> int:
        if src_rank == dest_rank:
            return 0
        cyclic_distance = (dest_rank - src_rank) % args.R
        return base_count + int(cyclic_distance <= remainder)

    # The remote region offset is the prefix of incoming source-peer regions,
    # matching udma_aggregate_perf's reverse-link region selection.
    remote_offsets = [[0] * args.R for _ in range(args.R)]
    for dest_rank in range(args.R):
        remote_offset = 0
        for src_rank in range(args.R):
            if src_rank == dest_rank:
                continue
            remote_offsets[src_rank][dest_rank] = remote_offset
            remote_offset += route_count(src_rank, dest_rank)
        if remote_offset != slots:
            raise AssertionError(
                f"destination rank {dest_rank} receives {remote_offset} routes, "
                f"expected {slots}"
            )

    destination_by_route = torch.empty(args.R * slots, dtype=torch.int32)
    for src_rank in range(args.R):
        local_offset = 0
        source_start = src_rank * slots
        for dest_rank in range(args.R):
            if dest_rank == src_rank:
                continue
            count = route_count(src_rank, dest_rank)
            remote_offset = remote_offsets[src_rank][dest_rank]
            if count > 0:
                destination_by_route[
                    source_start + local_offset:
                    source_start + local_offset + count
                ] = (
                    dest_rank * slots
                    + torch.arange(
                        remote_offset,
                        remote_offset + count,
                        dtype=torch.int32,
                    )
                )
            local_offset += count
        if local_offset != slots:
            raise AssertionError(
                f"source rank {src_rank} sends {local_offset} routes, "
                f"expected {slots}"
            )
    return destination_by_route


def make_plan(args, local_rank: int):
    slots = num_output_slots(args)
    dst_local = torch.empty(args.S * args.K, dtype=torch.int32)
    dup_groups_local = torch.full((slots, 3), -1, dtype=torch.int32)
    dup_loffs_local = torch.full((slots,), -1, dtype=torch.int32)
    dup_counts_local = torch.zeros((2,), dtype=torch.int32)
    expected_local = {}

    plan_mode = getattr(args, "plan_mode", "all-to-all")
    if plan_mode == "all-to-all":
        destination_by_route = make_all_to_all_destinations(args)
    elif plan_mode == "cyclic":
        return dispatch_test.make_plan(args, local_rank)
    elif plan_mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        destination_by_route = torch.randperm(
            args.R * slots,
            generator=generator,
            dtype=torch.int32,
        )
    else:
        raise ValueError(f"unsupported plan mode: {plan_mode}")

    group_count = 0
    dup_offset = 0

    for src_rank in range(args.R):
        for token_id in range(args.S):
            routes_by_dest = {}
            for topk_id in range(args.K):
                route_index = (src_rank * args.S + token_id) * args.K + topk_id
                raw_destination = int(destination_by_route[route_index])
                dest_rank, loff = divmod(raw_destination, slots)
                routes_by_dest.setdefault(dest_rank, []).append((topk_id, loff))
                if dest_rank == local_rank:
                    expected_local[loff] = RouteEntry(src_rank, token_id, topk_id)

            for dest_rank, members in routes_by_dest.items():
                members.sort()
                if dest_rank == local_rank and len(members) > 1:
                    primary = members[0][1]
                    dup_n = len(members) - 1
                    dup_groups_local[group_count] = torch.tensor(
                        [primary, dup_offset, dup_n], dtype=torch.int32
                    )
                    for duplicate_index, (_, loff) in enumerate(members[1:]):
                        dup_loffs_local[dup_offset + duplicate_index] = loff
                    group_count += 1
                    dup_offset += dup_n

                if src_rank == local_rank:
                    for member_index, (topk_id, loff) in enumerate(members):
                        raw = dest_rank * slots + loff
                        flat = token_id * args.K + topk_id
                        dst_local[flat] = raw if member_index == 0 else -raw - 1

    dup_counts_local[0] = group_count
    dup_counts_local[1] = dup_offset

    return dst_local, dup_groups_local, dup_loffs_local, dup_counts_local, expected_local


def build_staged_hidden(args, expected: dict[int, RouteEntry]) -> torch.Tensor:
    slots = num_output_slots(args)
    hidden = torch.zeros((slots, args.H), dtype=torch.bfloat16)
    for loff, entry in expected.items():
        hidden[loff] = expected_hidden_row(entry.src_rank, entry.token_id, args)
    return hidden.contiguous()


def build_staged_weights(args, expected: dict[int, RouteEntry]) -> torch.Tensor:
    slots = num_output_slots(args)
    weights = torch.zeros((slots,), dtype=torch.float32)
    for loff, entry in expected.items():
        weights[loff] = float(
            entry.src_rank * args.S * args.K + entry.token_id * args.K + entry.topk_id
        )
    return weights.contiguous()


def combine_identity_reference(
    hidden: torch.Tensor, dst: torch.Tensor, args
) -> torch.Tensor:
    """Model the staged BF16 reduction instead of assuming hidden * K exactly.

    Routes for one token are first reduced per destination rank and rounded to
    BF16 by the duplicate prologue. Combine then accumulates those per-rank
    partials in FP32 and rounds the final row back to BF16.
    """
    encoded = dst.reshape(args.S, args.K).to(torch.int64)
    raw = torch.where(encoded >= 0, encoded, -encoded - 1)
    destination_rank = torch.div(raw, num_output_slots(args), rounding_mode="floor")

    hidden_fp32 = hidden.float()
    reference_fp32 = torch.zeros_like(hidden_fp32)
    for rank in range(args.R):
        route_count = (destination_rank == rank).sum(dim=1, dtype=torch.int32)
        active_tokens = torch.nonzero(route_count, as_tuple=False).flatten()
        if active_tokens.numel() == 0:
            continue
        active_hidden = hidden_fp32.index_select(0, active_tokens)
        active_count = route_count.index_select(0, active_tokens).unsqueeze(1)
        partial = (active_hidden * active_count).to(torch.bfloat16).float()
        reference_fp32.index_add_(0, active_tokens, partial)
    return reference_fp32.to(torch.bfloat16)


def print_precision_metrics(
    rank: int,
    label: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    chunk_rows: int = 256,
) -> None:
    """Print exact-match and floating-point error metrics with bounded memory."""
    if actual.shape != reference.shape:
        raise ValueError(
            f"precision shape mismatch: actual={tuple(actual.shape)} "
            f"reference={tuple(reference.shape)}"
        )

    total = actual.numel()
    exact = 0
    sum_abs = 0.0
    max_abs = 0.0
    sum_rel = 0.0
    max_rel = 0.0
    tiny = torch.finfo(torch.float32).tiny

    for start in range(0, actual.shape[0], chunk_rows):
        end = min(start + chunk_rows, actual.shape[0])
        actual_chunk = actual[start:end].float()
        reference_chunk = reference[start:end].float()
        abs_error = (actual_chunk - reference_chunk).abs()
        relative_error = abs_error / reference_chunk.abs().clamp_min(tiny)

        exact += int((actual_chunk == reference_chunk).sum().item())
        sum_abs += float(abs_error.sum(dtype=torch.float64).item())
        max_abs = max(max_abs, float(abs_error.max().item()))
        sum_rel += float(relative_error.sum(dtype=torch.float64).item())
        max_rel = max(max_rel, float(relative_error.max().item()))

    print(
        f"[rank {rank}] precision {label}: "
        f"exact={exact}/{total} ({exact / total:.6%}), "
        f"max_abs={max_abs:.6g}, mean_abs={sum_abs / total:.6g}, "
        f"max_rel={max_rel:.6e}, mean_rel={sum_rel / total:.6e}",
        flush=True,
    )


def print_staged_mismatch_diagnostics(
    args,
    rank: int,
    actual: torch.Tensor,
    reference: torch.Tensor,
    dst: torch.Tensor,
    hidden_after: torch.Tensor,
    expected_local_hidden_after: torch.Tensor,
    actual_weights: torch.Tensor | None,
    expected_weights: torch.Tensor | None,
) -> None:
    """Emit bounded data that separates prologue, UDMA gather, and reduce failures."""
    exact_bad = actual != reference
    severe_bad = ~torch.isclose(
        actual.float(),
        reference.float(),
        rtol=torch.finfo(torch.bfloat16).eps,
        atol=0,
    )
    bad_rows = torch.nonzero(exact_bad.any(dim=1), as_tuple=False).flatten().tolist()
    severe_rows = torch.nonzero(severe_bad.any(dim=1), as_tuple=False).flatten().tolist()

    local_hidden_bad = hidden_after != expected_local_hidden_after
    local_hidden_bad_rows = torch.nonzero(
        local_hidden_bad.any(dim=1), as_tuple=False
    ).flatten().tolist()
    print(
        f"[MCH][summary] rank={rank} exact_bad_elements={int(exact_bad.sum())} "
        f"bad_rows={bad_rows} severe_bad_elements={int(severe_bad.sum())} "
        f"severe_rows={severe_rows} "
        f"local_prologue_bad_elements={int(local_hidden_bad.sum())} "
        f"local_prologue_bad_rows={local_hidden_bad_rows[:16]}",
        flush=True,
    )

    rows_to_print = severe_rows if severe_rows else bad_rows
    shown_h = min(args.H, 32)
    for token_id in rows_to_print[:12]:
        bad_cols = torch.nonzero(
            exact_bad[token_id], as_tuple=False
        ).flatten().tolist()
        severe_cols = torch.nonzero(
            severe_bad[token_id], as_tuple=False
        ).flatten().tolist()
        print(
            f"[MCH][row] rank={rank} token={token_id} "
            f"bad_cols={bad_cols[:64]} severe_cols={severe_cols[:64]} "
            f"actual_first={actual[token_id, :shown_h].float().tolist()} "
            f"expected_first={reference[token_id, :shown_h].float().tolist()}",
            flush=True,
        )

        positive_contributions = []
        token_routes = dst[
            token_id * args.K:(token_id + 1) * args.K
        ].to(torch.int64)
        token_raw = torch.where(token_routes >= 0, token_routes, -token_routes - 1)
        token_peers = torch.div(
            token_raw, num_output_slots(args), rounding_mode="floor"
        )
        source_row = expected_hidden_row(rank, token_id, args).float()
        for topk_id in range(args.K):
            route_idx = token_id * args.K + topk_id
            encoded = int(dst[route_idx])
            raw = encoded if encoded >= 0 else -encoded - 1
            peer, loff = divmod(raw, num_output_slots(args))
            peer_route_count = int((token_peers == peer).sum().item())
            contribution = (source_row * peer_route_count).to(torch.bfloat16)
            if encoded >= 0:
                positive_contributions.append((topk_id, peer, loff, contribution))

            weight_actual = None
            weight_expected = None
            if actual_weights is not None:
                weight_actual = float(actual_weights[token_id, topk_id])
            if expected_weights is not None:
                weight_expected = float(expected_weights[token_id, topk_id])
            print(
                f"[MCH][route] rank={rank} token={token_id} topk={topk_id} "
                f"route_idx={route_idx} encoded={encoded} positive={int(encoded >= 0)} "
                f"peer={peer} loff={loff} weight_actual={weight_actual} "
                f"weight_expected={weight_expected} "
                f"expected_contribution_first={contribution[:shown_h].float().tolist()}",
                flush=True,
            )

            if encoded >= 0 and peer == rank:
                local_row_bad = int(
                    (hidden_after[loff] != contribution).sum().item()
                )
                print(
                    f"[MCH][local-source] rank={rank} token={token_id} "
                    f"topk={topk_id} loff={loff} bad_elements={local_row_bad} "
                    f"actual_first={hidden_after[loff, :shown_h].float().tolist()}",
                    flush=True,
                )

        # If the actual row matches the reference with one positive route
        # removed, the missing peer/loff is identified directly.
        for missing_topk, missing_peer, missing_loff, _ in positive_contributions:
            candidate_fp32 = torch.zeros(args.H, dtype=torch.float32)
            for topk_id, _, _, contribution in positive_contributions:
                if topk_id != missing_topk:
                    candidate_fp32 += contribution.float()
            candidate = candidate_fp32.to(torch.bfloat16)
            candidate_bad = int((actual[token_id] != candidate).sum().item())
            candidate_max_abs = float(
                (actual[token_id].float() - candidate.float()).abs().max().item()
            )
            print(
                f"[MCH][missing-candidate] rank={rank} token={token_id} "
                f"drop_topk={missing_topk} peer={missing_peer} loff={missing_loff} "
                f"bad_elements={candidate_bad} max_abs={candidate_max_abs}",
                flush=True,
            )


def apply_prologue_cpu(
    hidden: torch.Tensor,
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
) -> torch.Tensor:
    out = hidden.clone().to(torch.float32)
    n_groups = int(dup_counts[0])
    for gid in range(n_groups):
        primary, dup_start, dup_n = map(int, dup_groups[gid].tolist())
        if primary < 0 or dup_n <= 0:
            continue
        acc = out[primary].clone()
        for j in range(dup_n):
            loff = int(dup_loffs[dup_start + j])
            if loff >= 0:
                acc += out[loff]
        out[primary] = acc
    return out.to(torch.bfloat16)


def combine_reference_all_ranks(args):
    """CPU golden: stage every rank, prologue, then gather for each rank."""
    slots = num_output_slots(args)
    staged = []
    staged_w = []
    plans = []
    for rank in range(args.R):
        dst, dup_groups, dup_loffs, dup_counts, expected = make_plan(args, rank)
        hidden = build_staged_hidden(args, expected)
        weights = build_staged_weights(args, expected)
        hidden = apply_prologue_cpu(hidden, dup_groups, dup_loffs, dup_counts)
        staged.append(hidden)
        staged_w.append(weights)
        plans.append((dst, dup_groups, dup_loffs, dup_counts, expected))

    outputs = []
    outputs_sk = []
    for rank in range(args.R):
        dst, _, _, _, _ = plans[rank]
        out = torch.zeros((args.S, args.H), dtype=torch.float32)
        out_sk = torch.zeros((args.S, args.K), dtype=torch.float32)
        for token_id in range(args.S):
            for topk_id in range(args.K):
                encoded = int(dst[token_id * args.K + topk_id])
                raw = encoded if encoded >= 0 else -encoded - 1
                dest_rank, loff = divmod(raw, slots)
                if encoded >= 0:
                    out[token_id] += staged[dest_rank][loff].float()
                out_sk[token_id, topk_id] = float(staged_w[dest_rank][loff])
        outputs.append(out.to(torch.bfloat16))
        outputs_sk.append(out_sk)
    return outputs, outputs_sk, plans


def combine_reference_for_rank(
    args, rank: int, dst: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build only one process's expected output for multi-rank kernel tests.

    The all-rank golden path proves that the staged plan is equivalent to the
    source rank's hidden/weight tensors. Reusing that identity here avoids
    materializing R staged tensors and R output tensors in every process.
    """
    hidden = make_hidden(rank, args)
    output = combine_identity_reference(hidden, dst, args)
    output_sk = None if args.no_weights else make_weights(rank, args)
    return output, output_sk


def validate_golden_combine(args) -> None:
    outputs, outputs_sk, plans = combine_reference_all_ranks(args)
    for rank in range(args.R):
        hidden = make_hidden(rank, args)
        ref = combine_identity_reference(hidden, plans[rank][0], args)
        if not torch.equal(outputs[rank], ref):
            mismatch = int((outputs[rank] != ref).sum().item())
            raise AssertionError(
                f"rank={rank} golden combine identity mismatch {mismatch}/{args.S * args.H}"
            )
        if not args.no_weights:
            weights = make_weights(rank, args)
            if not torch.allclose(outputs_sk[rank], weights):
                raise AssertionError(f"rank={rank} golden weight gather mismatch")
        _, _, _, dup_counts, expected = plans[rank]
        if len(expected) != num_output_slots(args):
            raise AssertionError(f"rank={rank} expected {num_output_slots(args)} slots")
        print(
            f"[rank {rank}] golden combine validated: groups={int(dup_counts[0])} "
            f"dups={int(dup_counts[1])} identity=hidden*{args.K}",
            flush=True,
        )


def run_rank(local_rank: int, args) -> None:
    validate_args(args)
    rank = global_rank_for_local(local_rank, args)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.R)
    os.environ["ASCEND_MOONEP_SHMEM_IP_PORT"] = args.shmem_ip_port
    os.environ["ASCEND_MOONEP_COMBINE_USE_UDMA"] = "1"

    import torch_npu
    from npu_patch._device import local_device_index
    from npu_patch import Buffer, MoonEPCommPlan, ShmemRuntime

    # Must come *after* the import above: a function-local ``import`` makes the
    # name local to this whole scope, so any earlier call would raise
    # UnboundLocalError.
    device_id = local_device_index(local_rank)
    torch.npu.set_device(device_id)
    buffer = None
    hidden_shared = None
    weights_shared = None
    output_sh = None
    output_sk = None
    try:
        slots = num_output_slots(args)
        buffer = Buffer(
            args.S,
            args.H,
            args.K,
            args.E,
            args.R,
            B=args.B,
            explicitly_destroy=True,
            rank=rank,
        )
        if not ShmemRuntime.is_initialized():
            init_shmem(args=args, 
                ip_port=args.shmem_ip_port,
                rank=rank,
                world_size=args.R,
                enable_udma=True,
                qp_count=2,
            )

        buffer._ensure_shmem_views()
        ctx = buffer._require_ctx()
        hidden_shared = ctx["hidden_buf_local"]
        weights_shared = ctx["weights_buf_local"].view(torch.float32)

        dst, dup_groups, dup_loffs, dup_counts, expected = make_plan(args, rank)
        device = torch.device(f"npu:{device_id}")
        hidden_input = build_staged_hidden(args, expected).to(device=device)
        weights_input = None
        if not args.no_weights:
            weights_input = build_staged_weights(args, expected).to(device=device)

        plan = MoonEPCommPlan(
            dst=dst.to(device=device),
            experts_to_copy=torch.full(
                (args.R, args.B), -1, dtype=torch.int32, device=device
            ),
            zero_fill_ranges=torch.zeros(
                (args.E + args.B, 2), dtype=torch.int32, device=device
            ),
            remote_stats=torch.zeros(2, dtype=torch.int32, device=device),
            dup_groups=dup_groups.to(device=device),
            dup_loffs=dup_loffs.to(device=device),
            dup_counts=dup_counts.to(device=device),
            N=slots,
            R=args.R,
            E=args.E,
            B=args.B,
            NvS=slots,
            K=args.K,
        )

        output_sh, output_sk, done = buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_input,
            route_weights_nvs=weights_input,
            async_finish=False,
            inter_rank_sync=True,
            zero_copy=False,
        )
        if done is not None:
            raise AssertionError("synchronous UDMA combine must return done=None")
        torch_npu.npu.synchronize()
        ShmemRuntime.barrier()

        reference, expected_weights = combine_reference_for_rank(args, rank, dst)
        actual = output_sh.cpu()
        print_precision_metrics(rank, "kernel_vs_staged_ref", actual, reference)
        if not torch.equal(actual, reference):
            # Permit one BF16 ULP for hardware-dependent FP32 accumulation order.
            if not torch.allclose(
                actual.float(),
                reference.float(),
                rtol=torch.finfo(torch.bfloat16).eps,
                atol=0,
            ):
                hidden_after = hidden_shared.cpu()
                expected_local_hidden_after = apply_prologue_cpu(
                    build_staged_hidden(args, expected),
                    dup_groups,
                    dup_loffs,
                    dup_counts,
                )
                actual_weights_diag = (
                    None if args.no_weights else output_sk.cpu()
                )
                expected_weights_diag = (
                    None if args.no_weights else expected_weights
                )
                print_staged_mismatch_diagnostics(
                    args,
                    rank,
                    actual,
                    reference,
                    dst,
                    hidden_after,
                    expected_local_hidden_after,
                    actual_weights_diag,
                    expected_weights_diag,
                )
                mismatch = int((actual != reference).sum().item())
                raise AssertionError(
                    f"rank={rank} staged combine mismatch "
                    f"{mismatch}/{args.S * args.H}; "
                    f"expect_first={float(reference[0, 0])} "
                    f"got_first={float(actual[0, 0])}"
                )

        if args.no_weights:
            if output_sk is not None:
                raise AssertionError(
                    "output_sk must be None when route weights are omitted"
                )
        else:
            actual_weights = output_sk.cpu()
            if slots <= 64:
                print(
                    f"[MCW][buffer-after] rank={rank} "
                    f"symmetric={weights_shared.cpu().tolist()} "
                    f"actual={actual_weights.tolist()} "
                    f"expected={expected_weights.tolist()}",
                    flush=True,
                )
            if not torch.equal(actual_weights, expected_weights):
                raise AssertionError(f"rank={rank} weight gather mismatch")

    finally:
        output_sk = None
        output_sh = None
        weights_shared = None
        hidden_shared = None
        try:
            if ShmemRuntime.is_initialized():
                ShmemRuntime.finalize()
        finally:
            if buffer is not None and not buffer.destroyed:
                buffer.destroy()
        gc.collect()

    print(
        f"[rank {rank}] MoonEPCombine PASS method=Buffer.combine(env=UDMA) input=staged",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan-driven staged MoonEP combine test"
    )
    parser.add_argument("--mode", choices=("golden", "kernel"), default="golden")
    parser.add_argument("--first-device", type=int, default=0, help="Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get(
            "ASCEND_MOONEP_SHMEM_IP_PORT", "tcp://127.0.0.1:8899"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--plan-mode",
        choices=("all-to-all", "cyclic", "random"),
        default="all-to-all",
        help=(
            "route layout: aggregate-perf-style deterministic peer regions, "
            "the exact dispatch/joint cyclic plan, or a seeded global random "
            "permutation"
        ),
    )
    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--S", type=int, default=2)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--E", type=int, default=4)
    parser.add_argument("--B", type=int, default=0)
    parser.add_argument("--no-weights", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.B:
        args.B = args.E // args.R
    validate_args(args)

    local_count = local_process_count(args)
    first_global_rank = global_rank_for_local(0, args)
    last_global_rank = global_rank_for_local(local_count - 1, args)
    print(
        f"MoonEPCombine Test: mode={args.mode}, "
        f"plan_mode={args.plan_mode}, seed={args.seed}, "
        f"R={args.R}, S={args.S}, K={args.K}, H={args.H}, E={args.E}, "
        f"B={args.B}, slots={num_output_slots(args)}, with_weights={not args.no_weights}, "
        f"node_rank={args.node_rank}, local_processes={local_count}, "
        f"global_ranks={first_global_rank}-{last_global_rank}",
        flush=True,
    )
    if args.mode == "golden":
        validate_golden_combine(args)
    elif local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)

    if args.nproc_per_node is None:
        print("\nALL RANKS PASS")
    else:
        print(f"\nNODE {args.node_rank} LOCAL RANKS PASS")


if __name__ == "__main__":
    main()
