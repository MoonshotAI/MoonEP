"""Measure device-side ``aclshmemx_barrier_all_vec`` latency.

The timed kernel contains only a loop of device barriers. One NPU event pair
times the whole loop, and the result is divided by ``--iters`` so per-launch
host overhead is amortized rather than charged to every barrier.

Engine (``--engine``): the barrier kernel itself has no transport variant
(device barriers use the SHMEM internal sync pool), but ``urma`` initializes
the symmetric heap in UDMA mode (with UDMA QP), so this benchmark also smokes
"urma heap usability"; ``ubmem`` is the default MTE/UB path.

Build first:
    bash npu_patch/scripts/build_a5.sh

Single node, 8 ranks:
    bash npu_patch/benchmarks/run_bench.sh bench_barrier.py \
        "--iters 100 --warmup 10 --samples 5 --block-dim 32"

Multi-node (run once on every node; example uses 8 NPUs per node):
    NODES=16 NPROC=8 NODE_RANK=0 MASTER_ADDR=<node0-ip> \
      bash npu_patch/benchmarks/run_bench.sh bench_barrier.py \
      "--iters 100 --warmup 10 --samples 5 --block-dim 32"

Use NODE_RANK=1..15 on the remaining nodes with the same MASTER_ADDR,
MASTER_PORT and arguments. The SHMEM bootstrap endpoint defaults to
tcp://MASTER_ADDR:(MASTER_PORT+1); ``--ip-port`` can override it.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist
import torch_npu

from _bench_common import add_engine_args, apply_engine, setup
from npu_patch import ShmemRuntime
from npu_patch._C import launch_device_barrier_bench
from npu_patch.buffer_c import (
    create_vmm_physical,
    reset_symmetric_descriptors,
)
from npu_patch.shmem_runtime import default_shmem_ip_port


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    parser.add_argument("--iters", type=int, default=100,
                        help="Device barriers in one timed kernel.")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Device barriers in the untimed warmup kernel.")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of independently timed kernels.")
    parser.add_argument("--block-dim", type=int, default=32,
                        help="AIV block count; use 32 to match planning.")
    parser.add_argument("--ip-port", default=None,
                        help="Multi-node SHMEM bootstrap, for example "
                             "tcp://10.0.0.1:29501. Defaults to "
                             "MASTER_ADDR:(MASTER_PORT+1) on multi-node runs.")
    return parser.parse_args()


def gather_rank_times(local_us: float, world_size: int):
    device = torch.device(f"npu:{torch.npu.current_device()}")
    value = torch.tensor([local_us], dtype=torch.float64, device=device)
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value)
    return [float(item.item()) for item in gathered]


def main():
    args = parse_args()
    if args.iters <= 0 or args.warmup < 0 or args.samples <= 0:
        raise SystemExit(
            "--iters and --samples must be positive; --warmup must be non-negative")
    if args.block_dim <= 0:
        raise SystemExit("--block-dim must be positive")

    rank, world_size, local_rank = setup()
    args.use_udma = apply_engine(args.engine)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size
    ip_port = args.ip_port
    if multi_node and ip_port is None:
        ip_port = default_shmem_ip_port(
            os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]))

    # Buffer-backed SHMEM requires at least one identically sized registered
    # descriptor on every rank. The benchmark kernel itself does not touch it;
    # SHMEM owns and uses its internal synchronization pool.
    backing, _, _ = create_vmm_physical(
        [4096], torch.uint8,
        torch.npu.current_device())

    try:
        ShmemRuntime.init_with_buffer(group=None, ip_port=ip_port,
                                      enable_udma=args.use_udma)
        dist.barrier()

        if args.warmup:
            launch_device_barrier_bench(args.warmup, args.block_dim)
            torch.npu.synchronize()
        dist.barrier()

        sample_max_us = []
        if rank == 0:
            print(
                f"device barrier benchmark: ranks={world_size}, "
                f"block_dim={args.block_dim}, barriers/kernel={args.iters}, "
                f"samples={args.samples}",
                flush=True,
            )

        for sample in range(args.samples):
            dist.barrier()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            launch_device_barrier_bench(args.iters, args.block_dim)
            end.record()
            end.synchronize()

            local_us = start.elapsed_time(end) * 1e3 / args.iters
            rank_times = gather_rank_times(local_us, world_size)
            sample_max_us.append(max(rank_times))
            if rank == 0:
                print(
                    f"sample {sample + 1}: per-barrier us "
                    f"min={min(rank_times):.3f} "
                    f"mean={statistics.fmean(rank_times):.3f} "
                    f"max={max(rank_times):.3f}",
                    flush=True,
                )

        if rank == 0:
            print(
                "result (max rank, averaged across samples): "
                f"{statistics.fmean(sample_max_us):.3f} us/barrier",
                flush=True,
            )
    finally:
        torch.npu.synchronize()
        dist.barrier()
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        allocation = getattr(backing, "_vmm_allocation", None)
        if allocation is not None:
            allocation.destroy()
        reset_symmetric_descriptors()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
