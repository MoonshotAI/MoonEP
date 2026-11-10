"""Plan-driven fused MoonEP dispatch test.

The test generates planning outputs directly, so the dispatch kernel does not
read src_info or build dedup tables.

By default all R ranks are spawned on one host. For multi-node runs, launch this
script once per node with the same UID rendezvous or SHMEM ipport endpoint, then
pass a unique --node-rank and the local process count via --nproc-per-node.

Run (single node, default ``--bootstrap uid`` -> ``ShmemRuntime.init_with_buffer``;
the script spawns R ranks internally with mp.spawn):
    python test_npu/mte/test_moonep_dispatch_with_plan.py --mode kernel --R 8 --S 64 --K 8 --H 128 --max-checks 8
(Multi-node example: launch once per node, each with a unique --node-rank:)
    python test_npu/mte/test_moonep_dispatch_with_plan.py --mode kernel --R 16 --S 64 --K 8 --H 128 \
        --nproc-per-node 8 --node-rank ${NODE_RANK} --master-addr <node0_ip> --master-port 29521
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_test_utils import init_shmem
import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SHMEM_HEAP_ALIGN_SIZE = 1024 * 1024 * 1024
OP_WORKSPACE_BYTES = 16 * 1024 * 1024
MAX_UB_ROW_BYTES = 128 * 1024
SEED = 20260803


def align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def num_experts(args) -> int:
    return args.E if args.E is not None else args.R * 4


def num_reserved_slots(args) -> int:
    return num_experts(args) // args.R


def required_shmem_bytes(args) -> int:
    """Exact Buffer symmetric chunks plus op-owned workspaces."""
    N = args.S * args.K
    E = num_experts(args)
    align8 = lambda value: align_up(value, 8)
    dup_rank_counts_off = align8(E)
    dup_groups_off = dup_rank_counts_off + 2 * args.R * 8
    dup_loffs_off = dup_groups_off + align8(N * 3)
    planning_workspace_bytes = (dup_loffs_off + align8(N)) * 4
    # combine_sync_buffer: independent combine grant/done cacheline pool
    # [R * kCombineMaxGroupWidth(16) * 2 signal slots * 512].
    combine_sync_buffer_bytes = args.R * 16 * 2 * 512
    # dispatch_sync_buffer: the Token control area plus four rounds of local sync area for 16 Token AIVs.
    dispatch_sync_buffer_bytes = args.R * 16 * 6 * 512 + 16 * 4 * 512
    return (
        N * args.H * 2
        + N * 4
        + 3 * 4
        + planning_workspace_bytes
        + combine_sync_buffer_bytes
        + dispatch_sync_buffer_bytes
        + 2 * OP_WORKSPACE_BYTES
    )


def effective_shmem_bytes(args) -> int:
    required = required_shmem_bytes(args)
    if required <= SHMEM_HEAP_ALIGN_SIZE:
        return SHMEM_HEAP_ALIGN_SIZE
    return align_up(required + SHMEM_HEAP_ALIGN_SIZE, SHMEM_HEAP_ALIGN_SIZE)


def effective_shmem_ip_port(args) -> str:
    if args.shmem_ip_port:
        return args.shmem_ip_port
    shmem_port = args.master_port + 1
    if shmem_port > 65535:
        raise ValueError(
            "--shmem-ip-port is required when --master-port is 65535"
        )
    return f"tcp://{args.master_addr}:{shmem_port}"


def validate_args(args) -> None:
    if args.R <= 0 or args.S <= 0 or args.K <= 0 or args.H <= 0:
        raise ValueError("R/S/K/H must be positive")
    E = num_experts(args)
    if E <= 0 or E % args.R != 0:
        raise ValueError("E must be positive and divisible by R")
    if (3 * E * args.R) % 4 != 0:
        raise ValueError("3*E*R must be divisible by 4 for Buffer planning scratch")
    if args.H % 8 != 0:
        raise ValueError("H must be multiple of 8")
    if args.H * 2 > MAX_UB_ROW_BYTES:
        raise ValueError(f"H is too large for UB row buffer: H*2={args.H * 2} > {MAX_UB_ROW_BYTES}")
    if args.R * args.S * args.K >= 2**31:
        raise ValueError("R*S*K must fit in positive int32 dst encoding")
    if args.profile and args.mode != "kernel":
        raise ValueError("--profile requires --mode kernel")
    if args.profile_iters <= 0:
        raise ValueError("profile-iters must be positive")
    if args.profile_warmup < 0:
        raise ValueError("profile-warmup must be non-negative")
    if not args.master_addr:
        raise ValueError("master-addr must not be empty")
    if args.master_port <= 0 or args.master_port > 65535:
        raise ValueError("master-port must be in [1, 65535]")
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


def selected_output_modes(args) -> tuple[str, ...]:
    if args.output_mode == "both":
        return ("zero-copy", "copy")
    return (args.output_mode,)


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


def expected_weight(src_rank: int, token_id: int, topk_id: int, args) -> float:
    """Reproduce make_weights' fp32 arithmetic for large global route ids."""
    base = torch.tensor(src_rank * args.S * args.K, dtype=torch.float32)
    local_route = torch.tensor(token_id * args.K + topk_id, dtype=torch.float32)
    return float((base + local_route).item())


def make_plan(args, local_rank: int):
    slots = num_output_slots(args)
    total_routes = args.R * slots

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    destination_by_route = torch.randperm(
        total_routes,
        generator=generator,
        dtype=torch.int32,
    )

    # Build this source rank's dst directly from its contiguous route slice.
    source_start = local_rank * slots
    local_raw_dst = destination_by_route[source_start:source_start + slots]
    local_dest_ranks = torch.div(
        local_raw_dst,
        slots,
        rounding_mode="floor",
    ).reshape(args.S, args.K)
    canonical = torch.ones((args.S, args.K), dtype=torch.bool)
    for topk_id in range(1, args.K):
        canonical[:, topk_id] = torch.all(
            local_dest_ranks[:, :topk_id]
            != local_dest_ranks[:, topk_id].unsqueeze(1),
            dim=1,
        )
    dst_local = torch.where(
        canonical.reshape(-1),
        local_raw_dst,
        -local_raw_dst - 1,
    ).contiguous()

    # Invert only this destination rank's NvS entries. The previous reference
    # scanned all R*NvS routes in Python for every process.
    destination_start = local_rank * slots
    destination_end = destination_start + slots
    local_destination_mask = (
        (destination_by_route >= destination_start)
        & (destination_by_route < destination_end)
    )
    source_route_indices = torch.nonzero(
        local_destination_mask,
        as_tuple=False,
    ).reshape(-1)
    if int(source_route_indices.numel()) != slots:
        raise AssertionError("random route permutation does not cover every local output slot")
    local_loffs = (
        destination_by_route[source_route_indices] - destination_start
    ).to(torch.int64)
    received_source_routes = torch.empty(slots, dtype=torch.int32)
    received_source_routes[local_loffs] = source_route_indices.to(torch.int32)

    expected_local = {}
    for loff, source_route in enumerate(received_source_routes.tolist()):
        src_rank, route_in_rank = divmod(source_route, slots)
        token_id, topk_id = divmod(route_in_rank, args.K)
        expected_local[loff] = RouteEntry(src_rank, token_id, topk_id)

    # Sorting source route ids orders rows by (src_rank, token_id, topk_id),
    # giving the deterministic group and duplicate ordering carried by plan.
    sorted_source_routes, sorted_loffs = torch.sort(received_source_routes)
    global_tokens = torch.div(
        sorted_source_routes,
        args.K,
        rounding_mode="floor",
    )
    _, token_counts = torch.unique_consecutive(
        global_tokens,
        return_counts=True,
    )
    token_starts = torch.cumsum(token_counts, dim=0) - token_counts
    duplicate_groups = token_counts > 1
    group_counts = token_counts[duplicate_groups]
    group_starts = token_starts[duplicate_groups]
    duplicate_counts = group_counts - 1
    duplicate_starts = torch.cumsum(duplicate_counts, dim=0) - duplicate_counts

    dup_groups_local = torch.full((slots, 3), -1, dtype=torch.int32)
    group_count = int(group_counts.numel())
    if group_count > 0:
        dup_groups_local[:group_count, 0] = sorted_loffs[group_starts].to(torch.int32)
        dup_groups_local[:group_count, 1] = duplicate_starts.to(torch.int32)
        dup_groups_local[:group_count, 2] = duplicate_counts.to(torch.int32)

    duplicate_members = torch.zeros(slots, dtype=torch.bool)
    if slots > 1:
        duplicate_members[1:] = global_tokens[1:] == global_tokens[:-1]
    duplicate_loffs = sorted_loffs[duplicate_members].to(torch.int32)
    duplicate_count = int(duplicate_loffs.numel())
    dup_loffs_local = torch.full((slots,), -1, dtype=torch.int32)
    dup_loffs_local[:duplicate_count] = duplicate_loffs
    dup_counts_local = torch.tensor(
        [group_count, duplicate_count],
        dtype=torch.int32,
    )

    return dst_local, dup_groups_local, dup_loffs_local, dup_counts_local, expected_local


def selected_loffs(expected_for_rank: dict[int, RouteEntry], args) -> list[int]:
    loffs = sorted(expected_for_rank)
    if args.max_checks <= 0 or len(loffs) <= args.max_checks:
        return loffs
    rng = random.Random(args.seed + 911)
    return sorted(rng.sample(loffs, args.max_checks))


def collect_dispatch_profile(args, rank: int, shmem, dispatch_once):
    import torch_npu
    import torch_npu.profiler

    profile_dir = os.path.abspath(os.path.join(args.profile_dir, f"rank_{rank}"))
    os.makedirs(profile_dir, exist_ok=True)

    hidden_out = None
    weight_out = None
    for _ in range(args.profile_warmup):
        hidden_out, weight_out = dispatch_once()
        torch_npu.npu.synchronize()

    shmem.barrier()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=args.profile_iters,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(args.profile_iters):
            hidden_out, weight_out = dispatch_once()
            torch_npu.npu.synchronize()
            prof.step()

    shmem.barrier()
    print(
        f"[rank {rank}] profile saved: warmup={args.profile_warmup} "
        f"active={args.profile_iters} path={profile_dir}",
        flush=True,
    )
    return hidden_out, weight_out


def validate_golden_plan(
    args,
    dst: torch.Tensor,
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
    expected: dict[int, RouteEntry],
) -> None:
    slots = num_output_slots(args)
    seen_destinations = set()
    for token_id in range(args.S):
        routes_by_dest = {}
        for topk_id in range(args.K):
            encoded = int(dst[token_id * args.K + topk_id])
            raw = encoded if encoded >= 0 else -encoded - 1
            dest_rank, loff = divmod(raw, slots)
            if dest_rank >= args.R or raw in seen_destinations:
                raise AssertionError(f"invalid or repeated dst entry at token={token_id}, topk={topk_id}")
            seen_destinations.add(raw)
            routes_by_dest.setdefault(dest_rank, []).append((topk_id, encoded))
        for members in routes_by_dest.values():
            members.sort()
            if members[0][1] < 0 or any(encoded >= 0 for _, encoded in members[1:]):
                raise AssertionError(f"invalid canonical dst encoding for token={token_id}")

    group_count = int(dup_counts[0])
    dup_count = int(dup_counts[1])
    if not (0 <= group_count <= slots and 0 <= dup_count <= slots):
        raise AssertionError(f"invalid dup_counts={dup_counts.tolist()}")
    if len(expected) != slots:
        raise AssertionError(f"expected {slots} populated rows, got {len(expected)}")

    consumed_loffs = set()
    for gid in range(group_count):
        primary, dup_start, dup_n = map(int, dup_groups[gid].tolist())
        if primary not in expected:
            raise AssertionError(f"group {gid} has invalid primary loff {primary}")
        if dup_start < 0 or dup_n <= 0 or dup_start + dup_n > dup_count:
            raise AssertionError(f"group {gid} has invalid duplicate range")

        primary_entry = expected[primary]
        group_entries = [primary_entry]
        for index in range(dup_start, dup_start + dup_n):
            duplicate = int(dup_loffs[index])
            if duplicate in consumed_loffs or duplicate not in expected:
                raise AssertionError(f"group {gid} has invalid duplicate loff {duplicate}")
            duplicate_entry = expected[duplicate]
            if (duplicate_entry.src_rank, duplicate_entry.token_id) != (
                primary_entry.src_rank,
                primary_entry.token_id,
            ):
                raise AssertionError(f"group {gid} mixes rows from different source tokens")
            group_entries.append(duplicate_entry)
            consumed_loffs.add(duplicate)
        if primary_entry.topk_id != min(entry.topk_id for entry in group_entries):
            raise AssertionError(f"group {gid} primary is not the canonical top-k route")

    if len(consumed_loffs) != dup_count:
        raise AssertionError(
            f"dup_counts reports {dup_count} duplicate rows, validated {len(consumed_loffs)}"
        )


def run_rank(local_rank: int, args) -> None:
    validate_args(args)
    rank = global_rank_for_local(local_rank, args)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.R)

    if args.mode == "golden":
        dst, dup_groups, dup_loffs, dup_counts, expected = make_plan(args, rank)
        validate_golden_plan(args, dst, dup_groups, dup_loffs, dup_counts, expected)
        print(
            f"[rank {rank}] golden plan validated: groups={int(dup_counts[0])} "
            f"dups={int(dup_counts[1])} checked_rows={len(selected_loffs(expected, args))}",
            flush=True,
        )
        return

    import torch_npu
    from npu_patch._device import local_device_index
    from npu_patch import (
        Buffer,
        MoonEPCommPlan,
        ShmemRuntime,
    )
    from npu_patch.dispatch import launch_dispatch, materialize_dispatch_outputs

    # Must come *after* the import above: a function-local ``import`` makes the
    # name local to this whole scope, so any earlier call would raise
    # UnboundLocalError.  Kept out of the golden path so plan-only runs need no NPU.
    device_id = local_device_index(local_rank)
    torch.npu.set_device(device_id)
    # UID bootstrap uses HCCL to broadcast the UID. The ipport path supplies
    # rank/world explicitly and creates no torch.distributed process group.
    if args.bootstrap == "uid":
        dist.init_process_group(
            backend="hccl",
            init_method=f"tcp://{args.master_addr}:{args.master_port}",
            rank=rank,
            world_size=args.R,
        )

    buffer = None
    hidden_shared = None
    weights_shared = None
    hidden_out = None
    weight_out = None
    ctx = None
    try:
        # Buffer-backed only: construct Buffer first (registers the VMM descriptor), then
        # ShmemRuntime.init_with_ip_port (wrapped by init_shmem). shmem initialization always goes through buffer-backed.
        E = num_experts(args)
        B = num_reserved_slots(args)
        buffer = Buffer(
            args.S,
            args.H,
            args.K,
            E,
            args.R,
            B=B,
            explicitly_destroy=True,
            rank=rank if args.bootstrap == "ipport" else None,
            use_udma_dispatch=False,
        )

        if not ShmemRuntime.is_initialized():
            if args.bootstrap == "ipport":
                init_shmem(args=args, 
                    ip_port=effective_shmem_ip_port(args),
                    rank=rank,
                    world_size=args.R,
                )
            else:
                init_shmem(args=args, group=None)

        # This test uses the low-level launch_dispatch (not buffer.dispatch), so the Buffer
        # communication chunks must be translated into the symmetric addresses the kernel expects.
        buffer._ensure_shmem_views()
        ctx = buffer._require_ctx()
        hidden_shared = ctx["hidden_buf_local"]
        weights_shared = ctx["weights_buf_local"].view(torch.float32)

        dst, dup_groups, dup_loffs, dup_counts, expected = make_plan(args, rank)
        hidden = make_hidden(rank, args)
        hidden_npu = hidden.to(device=f"npu:{device_id}")
        weights_npu = None
        if not args.no_weights:
            weights_npu = make_weights(rank, args).to(device=f"npu:{device_id}")
        device = hidden_npu.device

        plan = MoonEPCommPlan(
            dst=dst.to(device=device),
            experts_to_copy=torch.full(
                (args.R, B), -1, dtype=torch.int32, device=device
            ),
            zero_fill_ranges=torch.zeros(
                (E + B, 2), dtype=torch.int32, device=device
            ),
            remote_stats=torch.zeros(2, dtype=torch.int32, device=device),
            dup_groups=dup_groups.to(device=device),
            dup_loffs=dup_loffs.to(device=device),
            dup_counts=dup_counts.to(device=device),
            N=num_output_slots(args),
            R=args.R,
            E=E,
            B=B,
            NvS=num_output_slots(args),
            K=args.K,
        )

        if args.clear_symmetric:
            hidden_shared.zero_()
            weights_shared.zero_()
            torch_npu.npu.synchronize()
        ShmemRuntime.barrier()
        launch_dispatch(ctx, hidden_npu, weights_npu, plan)
        torch_npu.npu.synchronize()
        ShmemRuntime.barrier()

        if args.profile:
            def profile_dispatch_once():
                launch_dispatch(ctx, hidden_npu, weights_npu, plan)
                return materialize_dispatch_outputs(
                    ctx,
                    with_weights=weights_npu is not None,
                    zero_copy=True,
                )

            collect_dispatch_profile(
                args, rank, ShmemRuntime, profile_dispatch_once
            )

        for output_mode in selected_output_modes(args):
            zero_copy = output_mode == "zero-copy"
            hidden_out, weight_out = materialize_dispatch_outputs(
                ctx,
                with_weights=weights_npu is not None,
                zero_copy=zero_copy,
            )
            torch_npu.npu.synchronize()
            if zero_copy:
                if hidden_out.data_ptr() != hidden_shared.data_ptr():
                    raise AssertionError("zero-copy hidden output must alias SHMEM")
                if (
                    not args.no_weights
                    and weight_out.data_ptr() != weights_shared.data_ptr()
                ):
                    raise AssertionError("zero-copy weight output must alias SHMEM")
            else:
                if hidden_out.data_ptr() == hidden_shared.data_ptr():
                    raise AssertionError("copy hidden output must use ordinary NPU storage")
                if (
                    not args.no_weights
                    and weight_out.data_ptr() == weights_shared.data_ptr()
                ):
                    raise AssertionError("copy weight output must use ordinary NPU storage")

            for loff in selected_loffs(expected, args):
                entry = expected[loff]
                actual_h = hidden_out[loff].cpu()
                expect_h = expected_hidden_row(entry.src_rank, entry.token_id, args)
                if not torch.equal(actual_h, expect_h):
                    mismatch_count = int((actual_h != expect_h).sum().item())
                    raise AssertionError(
                        f"rank={rank} output_mode={output_mode} loff={loff} "
                        f"hidden mismatch {mismatch_count}/{args.H}; "
                        f"source=(rank={entry.src_rank}, token={entry.token_id}, "
                        f"topk={entry.topk_id}) expect_first={float(expect_h[0])} "
                        f"got_first={float(actual_h[0])}"
                    )

                if args.no_weights:
                    if weight_out is not None:
                        raise AssertionError(
                            "weight_out must be None when route weights are omitted"
                        )
                else:
                    actual_w = float(weight_out[loff].cpu().item())
                    expect_w = expected_weight(
                        entry.src_rank, entry.token_id, entry.topk_id, args
                    )
                    if actual_w != expect_w:
                        raise AssertionError(
                            f"output_mode={output_mode} loff={loff} "
                            f"weight expected {expect_w}, got {actual_w}"
                        )

    finally:
        hidden_out = None
        weight_out = None
        hidden_shared = None
        weights_shared = None
        ctx = None
        try:
            # Buffer-backed only: finalize first (tears down the symmetric mapping), then buffer.destroy (VMM inverse operations).
            # The C-side acl_shmem_finalize is idempotent based on init_status.
            ShmemRuntime.finalize()
        finally:
            if buffer is not None:
                buffer.destroy()
            if dist.is_initialized():
                dist.destroy_process_group()
    print(f"[rank {rank}] MoonEPDispatch PASS", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Plan-driven fused MoonEP dispatch test")
    parser.add_argument("--mode", choices=("golden", "kernel"), default="golden")
    parser.add_argument(
        "--bootstrap",
        choices=("uid", "ipport"),
        default="uid",
        help="SHMEM bootstrap; ipport does not initialize torch.distributed",
    )
    parser.add_argument("--first-device", type=int, default=0, help="Deprecated and ignored: the per-rank device index is resolved from ASCEND_RT_VISIBLE_DEVICES via npu_patch._device.")
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="zero-based node index; requires --nproc-per-node",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="enable multi-node mode and spawn this many local NPU processes",
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help=(
            "UID rendezvous host and default SHMEM ipport host; "
            "use the rank-0 host address on every node"
        ),
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "28889")),
        help="UID HCCL rendezvous port; default SHMEM ipport uses this port plus one",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=None,
        help=(
            "SHMEM ipport endpoint, for example tcp://10.0.0.1:28890; "
            "defaults to tcp://MASTER_ADDR:(MASTER_PORT+1)"
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--prod-shape",
        action="store_true",
        help="Production 8-card config: R=8, S=4096, K=16, H=7168, E=896.",
    )
    parser.add_argument("--R", type=int, default=2)
    parser.add_argument("--S", type=int, default=2)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument(
        "--E",
        type=int,
        default=None,
        help="expert count used to size Buffer metadata; defaults to 4*R",
    )
    parser.add_argument("--no-weights", action="store_true")
    parser.add_argument(
        "--output-mode",
        choices=("zero-copy", "copy", "both"),
        default="zero-copy",
        help="validate SHMEM aliases, ordinary copied outputs, or both",
    )
    parser.add_argument(
        "--clear-symmetric",
        action="store_true",
        help="clear the SHMEM output tensors before dispatch (debug only)",
    )
    parser.add_argument(
        "--max-checks",
        type=int,
        default=0,
        help="0 means check every populated local row; use a positive value for large shapes.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="collect an Ascend PyTorch profile for repeated dispatch iterations",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=5,
        help="number of unrecorded dispatch warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=20,
        help="number of dispatch iterations recorded by the profiler (default: 20)",
    )
    parser.add_argument(
        "--profile-dir",
        default="dispatch_perf",
        help="profile output directory, relative to the launch working directory by default",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prod_shape:
        args.R = 8
        args.S = 4096
        args.K = 16
        args.H = 7168
        args.E = 896
        args.max_checks = 0
    validate_args(args)
    local_count = local_process_count(args)
    first_global_rank = global_rank_for_local(0, args)
    last_global_rank = global_rank_for_local(local_count - 1, args)
    print(
        f"MoonEPDispatch Test: mode={args.mode}, seed={args.seed}, "
        f"R={args.R}, S={args.S}, K={args.K}, H={args.H}, "
        f"E={num_experts(args)}, B={num_reserved_slots(args)}, "
        f"slots={num_output_slots(args)}, "
        f"with_weights={not args.no_weights}, output_mode={args.output_mode}, "
        f"clear_symmetric={args.clear_symmetric}, "
        f"profile={args.profile}, profile_warmup={args.profile_warmup}, profile_iters={args.profile_iters}, "
        f"effective_shmem={effective_shmem_bytes(args)}, bootstrap={args.bootstrap}, "
        f"shmem_ip_port={effective_shmem_ip_port(args) if args.bootstrap == 'ipport' else 'unused'}, "
        f"master={args.master_addr}:{args.master_port}, node_rank={args.node_rank}, "
        f"local_processes={local_count}, global_ranks={first_global_rank}-{last_global_rank}",
        flush=True,
    )
    if local_count == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=local_count, join=True)
    if args.nproc_per_node is None:
        print("\nALL RANKS PASS")
    else:
        print(f"\nNODE {args.node_rank} LOCAL RANKS PASS")


if __name__ == "__main__":
    main()
