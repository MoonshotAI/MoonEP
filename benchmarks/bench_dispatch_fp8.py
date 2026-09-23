"""Compare BF16 and DeepSeek 1x128 FP8 dispatch, including scale traffic.

Example:
    PYTHONPATH=. torchrun --nproc_per_node=8 benchmarks/bench_dispatch_fp8.py \
        --tokens 2048 8192 --hidden 3584 7168 --out fp8_dispatch.csv

Inputs are prequantized. Times include planning (fresh mode), inter-rank
synchronization, dispatch, duplicate expansion, and boundary copies when
zero_copy=False. They exclude quantization, GEMM, weight prefetch and combine.
CUDA graphs remove Python launch overhead. Each sample uses the slowest rank;
formats alternate measurement order and the reported time is the sample median.
speedup_vs_run_bf16 compares formats in this run of the loaded MoonEP code;
it is empty when BF16 is not measured. To compare against upstream, run this
script with --dtypes bf16 against an upstream checkout and compare matched rows.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import statistics
import subprocess

import torch
import torch.distributed as dist

import moonep
from moonep import Buffer
from tests.generate_topk_routing import generate_topk_routing
from tests.quantization_reference import quantize_fp8


def source_metadata():
    """Identify the loaded package, including edits and the compiled extension."""
    package = Path(moonep.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".so"}:
            digest.update(str(path.relative_to(package)).encode() + b"\0")
            digest.update(path.read_bytes())
    revision = "unknown"
    try:
        root = subprocess.check_output(
            ["git", "-C", str(package.parent), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        # A source export may live under an unrelated parent repository.
        if Path(root).resolve() == package.parent:
            revision = subprocess.check_output(
                ["git", "-C", root, "rev-parse", "HEAD"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return dict(git_revision=revision, source_sha256=digest.hexdigest(),
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def time_dispatch(fn, warmup, iters, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        dist.barrier()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed = torch.tensor([begin.elapsed_time(end) * 1000 / iters],
                               dtype=torch.float64, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        values.append(elapsed.item())
    return statistics.median(values)


def remote_counts(plan, R, rank):
    dst = plan.dst.long()
    raw = torch.where(dst < 0, -dst - 1, dst)
    dest_rank = raw // plan.NvS
    unique = torch.bincount(dest_rank[dst >= 0], minlength=R)
    weights = torch.bincount(dest_rank, minlength=R)
    counts = torch.stack((unique, weights))
    counts[:, rank] = 0
    all_counts = [torch.empty_like(counts) for _ in range(R)]
    dist.all_gather(all_counts, counts)
    return torch.stack(all_counts).cpu()  # [sender, data/weights, receiver]


def remote_bytes(counts, row_bytes):
    traffic = counts[:, 0, :] * row_bytes + counts[:, 1, :] * 4
    return int(torch.maximum(traffic.sum(0).max(), traffic.sum(1).max()).item())


def bench_case(args, rank, R, S, H, bias, metadata):
    torch.manual_seed(1234 + rank)
    buffer = Buffer(S, H, args.topk, args.experts, R, num_sms=args.num_sms,
                    token_padding=args.token_padding, enable_pdl=not args.no_pdl)
    try:
        hidden = torch.randn(S, H, dtype=torch.bfloat16, device="cuda")
        inputs = {"bf16": hidden}
        if "fp8" in args.dtypes:
            inputs["fp8"] = quantize_fp8(hidden)
        weights = torch.rand(S, args.topk, dtype=torch.float32, device="cuda")
        topk, tpe = generate_topk_routing(S, args.topk, args.experts, R, bias,
                                         "cuda", 1234, rank=rank)
        _, _, _, plan = buffer.dispatch(hidden, weights, topk, tpe, zero_copy=True)
        counts = remote_counts(plan, R, rank)
        results = []
        for mode in args.modes:
            for zero_copy in args.zero_copy:
                samples = {dtype: [] for dtype in args.dtypes}
                for sample in range(args.samples):
                    order = args.dtypes if sample % 2 == 0 else args.dtypes[::-1]
                    for dtype in order:
                        def dispatch():
                            buffer.dispatch(inputs[dtype], weights, topk, tpe,
                                            plan=plan if mode == "cached" else None,
                                            zero_copy=bool(zero_copy))
                        us = time_dispatch(dispatch, args.warmup, args.iters, args.repeats)
                        samples[dtype].append(us)
                medians = {dtype: statistics.median(values) for dtype, values in samples.items()}
                for dtype in args.dtypes:
                    row_bytes = H * 2 if dtype == "bf16" else H + H // 128 * 4
                    nbytes = remote_bytes(counts, row_bytes)
                    us = medians[dtype]
                    speedup = medians["bf16"] / us if "bf16" in medians else None
                    row = dict(metadata, R=R, S=S, H=H, K=args.topk, E=args.experts,
                               bias=bias, mode=mode, zero_copy=zero_copy, dtype=dtype,
                               num_sms=args.num_sms, token_padding=args.token_padding,
                               num_sms_dedup=buffer._require_ctx()['num_sms_dedup'],
                               worst_rank_us=us, min_sample_us=min(samples[dtype]),
                               max_sample_us=max(samples[dtype]),
                               speedup_vs_run_bf16=speedup,
                               remote_MB=nbytes / 1e6, remote_GBps=nbytes / us / 1e3,
                               row_bytes=row_bytes)
                    results.append(row)
                    if rank == 0:
                        speedup_text = f"{speedup:.3f}" if speedup is not None else "n/a"
                        print(f"{S:6d} {H:5d} {bias:5.1f} {mode:6s} {zero_copy:2d} "
                              f"{dtype:4s} {us:11.2f} {speedup_text:>12s} "
                              f"{row['remote_MB']:10.2f} {row['remote_GBps']:10.2f}", flush=True)
        return results
    finally:
        buffer.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 2048, 8192])
    parser.add_argument("--hidden", type=int, nargs="+", default=[3584, 7168])
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--bias", type=float, nargs="+", default=[0.0, 1.0, 5.0])
    parser.add_argument("--num-sms", type=int, default=32)
    parser.add_argument("--token-padding", type=int, default=128)
    parser.add_argument("--dtypes", choices=["bf16", "fp8"], nargs="+", default=["bf16", "fp8"])
    parser.add_argument("--modes", choices=["fresh", "cached"], nargs="+", default=["fresh", "cached"])
    parser.add_argument("--zero-copy", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--no-pdl", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if any(h <= 0 or h % 128 for h in args.hidden):
        parser.error("hidden sizes must be positive multiples of 128")
    if min(*args.tokens, args.warmup, args.iters, args.repeats, args.samples) <= 0:
        parser.error("tokens and timing counts must be positive")
    if not 1 <= args.topk <= min(args.experts, 32):
        parser.error("topk must be in [1, min(experts, 32)]")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, R = dist.get_rank(), dist.get_world_size()
    try:
        if args.experts % R:
            raise ValueError("experts must be divisible by EP world size")
        metadata = dict(source_metadata(), gpu=torch.cuda.get_device_name(),
                        torch_version=torch.__version__, cuda_version=torch.version.cuda,
                        pdl=not args.no_pdl, warmup=args.warmup, iters=args.iters,
                        repeats=args.repeats, samples=args.samples)
        if rank == 0:
            print(f"MoonEP BF16/FP8 dispatch: R={R}, GPU={torch.cuda.get_device_name()}, "
                  f"torch={torch.__version__}, CUDA={torch.version.cuda}", flush=True)
            print("Prequantized input; includes scales, excludes quantization/GEMM/prefetch/combine.")
            print("CUDA graph timing: median of samples, slowest rank per replay.")
            print(f"Loaded source: revision={metadata['git_revision']}, "
                  f"sha256={metadata['source_sha256']}")
            print("vs run BF16 = BF16 latency / dtype latency in this run; n/a without BF16.")
            print("     S     H  bias   mode ZC   dt   Worst(us)  vs run BF16 Remote(MB)   BW(GB/s)")
        results = []
        for S in args.tokens:
            for H in args.hidden:
                for bias in args.bias:
                    results.extend(bench_case(args, rank, R, S, H, bias, metadata))
                    torch.cuda.empty_cache()
                    if rank == 0 and args.out:
                        args.out.parent.mkdir(parents=True, exist_ok=True)
                        with args.out.open("w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
                            writer.writeheader()
                            writer.writerows(results)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
