"""Benchmark a producer GEMM with MoonEP direct-out publication.

The control writes a token-major local output and then runs ordinary MoonEP
payload dispatch. The candidate replaces that local store with peer-VMM stores
to representative final-owner rows and runs metadata publication plus duplicate
expansion. The CUDA Tile producer is benchmark-only and is not a runtime
dependency of ``moonep``.

Example:

    torchrun --nproc_per_node=4 --module benchmarks.bench_producer_direct_out \
        --ep 4 --hidden 3584 --topk 8 --unbalance-ratio 1.0 \
        --out results/direct_out.csv
"""

import argparse
import csv
import os

import torch
import torch.distributed as dist

from benchmarks.bench_comm import setup, time_gpu_op
from moonep import Buffer
from moonep.dispatch import launch_dispatch
from moonep.dispatch_epilogue import launch_dispatch_epilogue
from moonep.inter_rank_sync import launch_inter_rank_sync
from tests.generate_topk_routing import generate_topk_routing


def distributed_mean(local_value: float, group) -> float:
    """Average one timing value across the EP group."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    value = torch.tensor([local_value], dtype=torch.float64, device=device)
    gathered = [
        torch.empty_like(value)
        for _ in range(dist.get_world_size(group=group))
    ]
    dist.all_gather(gathered, value, group=group)
    return torch.cat(gathered).mean().item()


def time_schedule(schedule, *, warmup: int, iters: int, group) -> float:
    """Time a captured multi-stream schedule."""
    schedule(warmup)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        schedule(iters)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return distributed_mean(
        start.elapsed_time(end) * 1e3 / iters,
        group,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=896)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--producer-k", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--unbalance-ratio", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from benchmarks.producer_direct_out_kernels import (
            launch_producer_gemm_direct_out,
            launch_producer_gemm_local,
        )
    except ImportError as exc:
        raise RuntimeError(
            "bench_producer_direct_out requires the optional cuda.tile package"
        ) from exc

    rank, world, local_rank = setup()
    if world != args.ep:
        raise ValueError(f"--ep={args.ep} requires world_size={args.ep}, got {world}")
    if args.experts % args.ep != 0:
        raise ValueError("--experts must be divisible by --ep")
    if args.topk > args.experts:
        raise ValueError("--topk must not exceed --experts")
    if args.seq_len % args.tile_m != 0:
        raise ValueError("--seq-len must be divisible by --tile-m")
    if args.hidden % args.tile_n != 0:
        raise ValueError("--hidden must be divisible by --tile-n")
    if args.producer_k % args.tile_k != 0:
        raise ValueError("--producer-k must be divisible by --tile-k")

    group = dist.group.WORLD
    device = torch.device(f"cuda:{local_rank}")
    buffer = Buffer(
        args.seq_len,
        args.hidden,
        args.topk,
        args.experts,
        args.ep,
        num_sms=32,
        group=group,
        explicitly_destroy=True,
    )
    ctx = buffer._require_ctx()
    comm = buffer._comm_stream
    if comm is None:
        raise RuntimeError("MoonEP communication stream is unavailable")
    topk, tpe = generate_topk_routing(
        args.seq_len,
        args.topk,
        args.experts,
        args.ep,
        args.unbalance_ratio,
        device,
        1234,
        rank=rank,
    )
    target, _, plan = buffer.prepare_dispatch(
        topk,
        tpe,
        inter_rank_sync=True,
    )

    torch.manual_seed(3000 + rank)
    producer_input = torch.empty(
        (args.seq_len, args.producer_k),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.1)
    producer_weight = torch.empty(
        (args.producer_k, args.hidden),
        dtype=torch.bfloat16,
        device=device,
    ).normal_(mean=0.0, std=0.01)
    local_output = torch.empty(
        (args.seq_len, args.hidden),
        dtype=torch.bfloat16,
        device=device,
    )
    route_weights = torch.rand(
        (args.seq_len, args.topk),
        dtype=torch.float32,
        device=device,
    )

    def producer_local() -> None:
        launch_producer_gemm_local(
            producer_input,
            producer_weight,
            local_output,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
        )

    def producer_direct() -> None:
        launch_producer_gemm_direct_out(
            producer_input,
            producer_weight,
            plan.dst,
            target.mapped_hidden,
            nvs=int(ctx["NvS"]),
            nvs_padded=int(ctx["NvS_padded"]),
            topk=args.topk,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
        )

    def dispatch_payload() -> None:
        launch_dispatch(
            ctx,
            local_output,
            None,
            plan,
            build_dedup_map=True,
        )
        launch_dispatch_epilogue(ctx, plan)

    def publish_only() -> None:
        launch_dispatch(
            ctx,
            None,
            None,
            plan,
            write_payload=False,
            build_dedup_map=True,
        )
        launch_dispatch_epilogue(ctx, plan)

    def prepare_metadata() -> None:
        launch_dispatch(
            ctx,
            None,
            None,
            plan,
            write_payload=False,
            build_dedup_map=True,
        )

    def publish_visibility_and_expand() -> None:
        launch_inter_rank_sync(ctx)
        launch_dispatch_epilogue(ctx, plan)

    def baseline_boundary() -> None:
        producer_local()
        dispatch_payload()

    def direct_boundary() -> None:
        producer_direct()
        publish_only()

    ready_events = [
        torch.cuda.Event(enable_timing=False)
        for _ in range(max(args.warmup, args.iters))
    ]
    metadata_done_events = [
        torch.cuda.Event(enable_timing=False)
        for _ in range(max(args.warmup, args.iters))
    ]

    def overlapped_direct_boundary(count: int) -> None:
        main_stream = torch.cuda.current_stream()
        for idx in range(count):
            ready_events[idx].record(main_stream)
            comm.wait_event(ready_events[idx])
            with torch.cuda.stream(comm):
                prepare_metadata()
                metadata_done_events[idx].record(comm)
            producer_direct()
            main_stream.wait_event(metadata_done_events[idx])
            publish_visibility_and_expand()

    baseline_boundary()
    torch.cuda.synchronize()
    baseline = ctx["hidden_buf_local"].clone()
    direct_boundary()
    torch.cuda.synchronize()
    torch.testing.assert_close(ctx["hidden_buf_local"], baseline, rtol=0, atol=0)
    overlapped_direct_boundary(1)
    torch.cuda.synchronize()
    torch.testing.assert_close(ctx["hidden_buf_local"], baseline, rtol=0, atol=0)

    producer_local_us = time_gpu_op(
        producer_local, args.warmup, args.iters, group, cudagraph=True
    )
    producer_direct_us = time_gpu_op(
        producer_direct, args.warmup, args.iters, group, cudagraph=True
    )
    dispatch_payload_us = time_gpu_op(
        dispatch_payload, args.warmup, args.iters, group, cudagraph=True
    )
    publish_us = time_gpu_op(
        publish_only, args.warmup, args.iters, group, cudagraph=True
    )
    baseline_us = time_gpu_op(
        baseline_boundary, args.warmup, args.iters, group, cudagraph=True
    )
    direct_us = time_gpu_op(
        direct_boundary, args.warmup, args.iters, group, cudagraph=True
    )
    overlapped_direct_us = time_schedule(
        overlapped_direct_boundary,
        warmup=args.warmup,
        iters=args.iters,
        group=group,
    )
    direct_improvement = (baseline_us - direct_us) / baseline_us * 100.0
    overlap_improvement = (
        (baseline_us - overlapped_direct_us) / baseline_us * 100.0
    )
    further_improvement = (
        (direct_us - overlapped_direct_us) / direct_us * 100.0
    )

    producer_local()
    launch_dispatch(
        ctx,
        local_output,
        route_weights,
        plan,
        build_dedup_map=True,
    )
    launch_dispatch_epilogue(ctx, plan)
    torch.cuda.synchronize()
    public_hidden_reference = ctx["hidden_buf_local"].clone()
    public_weights_reference = (
        ctx["weights_buf_local"].view(torch.float32).clone()
    )
    prepared_publication = buffer.prepare_dispatch_publication(
        target,
        plan,
        route_weights_sk=route_weights,
    )
    producer_direct()
    public_output, public_weights, public_done = buffer.publish_dispatch(
        target,
        plan,
        async_finish=True,
        zero_copy=True,
        prepared_publication=prepared_publication,
    )
    if public_done is None:
        raise AssertionError("async publication did not return an event")
    public_done.synchronize()
    torch.testing.assert_close(
        public_output,
        public_hidden_reference,
        rtol=0,
        atol=0,
    )
    if public_weights is None:
        raise AssertionError("route-weight publication returned no view")
    torch.testing.assert_close(
        public_weights,
        public_weights_reference,
        rtol=0,
        atol=0,
    )

    if rank == 0:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        row = {
            "ep": args.ep,
            "H": args.hidden,
            "producer_K": args.producer_k,
            "S": args.seq_len,
            "E": args.experts,
            "topk": args.topk,
            "unbalance_ratio": args.unbalance_ratio,
            "producer_local_us": f"{producer_local_us:.4f}",
            "producer_direct_us": f"{producer_direct_us:.4f}",
            "dispatch_payload_us": f"{dispatch_payload_us:.4f}",
            "publish_us": f"{publish_us:.4f}",
            "baseline_boundary_us": f"{baseline_us:.4f}",
            "direct_boundary_us": f"{direct_us:.4f}",
            "direct_improvement_pct": f"{direct_improvement:.4f}",
            "overlapped_direct_boundary_us": f"{overlapped_direct_us:.4f}",
            "overlapped_direct_improvement_pct": (
                f"{overlap_improvement:.4f}"
            ),
            "further_improvement_pct": f"{further_improvement:.4f}",
        }
        with open(args.out, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        print(
            f"H={args.hidden} K={args.topk} "
            f"skew={args.unbalance_ratio:.2f} "
            f"baseline={baseline_us:.2f} us "
            f"direct={direct_us:.2f} us "
            f"overlap_direct={overlapped_direct_us:.2f} us "
            f"improvement={overlap_improvement:.2f}% "
            f"further={further_improvement:.2f}%",
            flush=True,
        )

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
