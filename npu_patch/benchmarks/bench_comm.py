"""
MoonEP-Ascend dispatch/combine communication benchmark (Buffer API).

Ascend port of ``MoonEP/benchmarks/bench_comm.py``. Benchmarks the MoE
expert-parallel communication critical path through the public ``Buffer`` API
(``npu_patch.Buffer``) instead of the CUDA ``launch_*`` kernels:

  - planning:    launch_planning on the current rank's topk/tpe (PLAN broadcast
                 orchestrated by the API layer).
  - dispatch_fwd: Buffer.dispatch(hidden, route_weights, topk, tpe) with a
                 fresh plan, zero_copy=True -> shard aliasing (planning +
                 dispatch + dispatch-epilogue, no boundary copies).
  - dispatch_bwd: Buffer.dispatch(grad_output, None, plan=plan) reusing the
                 saved plan, hidden-only (dispatch + epilogue).
  - combine_fwd:  Buffer.combine(plan, shard, None) hidden-only, zero_copy.
  - combine_bwd:  Buffer.combine(plan, shard, route_weights) with the route
                 weights gathered back.

The epilogue/prologue kernels are internal to ``dispatch``/``combine`` on
Ascend, so they are not timed separately (unlike the CUDA reference). All
timings are cross-rank mean us. Bandwidth = S*K*H*2 / t (logical payload,
identical formula to the reference).

Shmem init (buffer-backed only; analysis doc §4): inside each config's
``bench_one``: construct Buffer (registers CANN VMM descriptors) ->
ShmemRuntime.init_with_buffer -> run -> destroy -> finalize -> reset (re-init
per config; setup is untimed).

Sweep mirrors the reference defaults: H in {3584, 7168}, K in {8, 16},
S = 8192, unbalance ratio in {0.1, 1.0, 5.0}, E = 896 total experts. EP group
size defaults to the launched world size (8 on a single node, 16 across two
nodes) — override with ``--ep-size`` for a subgroup of ranks [0, ep). ``--prod``
runs just the production 8-card config (S=4096, K=16, E=896, H=7168).

Every flow runs a live Ascend kernel and reports a cross-rank mean latency.

Run:
    # single node, 8 cards
    bash npu_patch/benchmarks/run_bench.sh bench_comm.py
    # 2 nodes / 16 cards (run separately on each node)
    bash npu_patch/benchmarks/run_bench.sh bench_comm.py --nnodes 2 --nproc-per-node 8 \
        --node-rank 0 --master-addr <node0-ip>
    bash npu_patch/benchmarks/run_bench.sh bench_comm.py --nnodes 2 --nproc-per-node 8 \
        --node-rank 1 --master-addr <node0-ip>
"""

import argparse
import csv
import os
import traceback

import torch
import torch.distributed as dist
import torch_npu

from _bench_common import (
    PROD_E,
    PROD_H,
    PROD_K,
    PROD_S,
    add_common_args,
    apply_engine,
    fmt4,
    measure_flow,
    setup,
    time_text,
)
from _bench_common import generate_topk_routing  # re-export

from npu_patch import Buffer, ShmemRuntime
from npu_patch.launchers import launch_combine, launch_dispatch, launch_planning
from npu_patch.planning import allocate_planning_outputs

# Reference sweep defaults (experts-per-rank shrinks as R grows).
HIDDENS = [3584, 7168]
TOPKS = [8, 16]
SEQLENS = [8192]
RATIOS = [0.1, 1.0, 5.0]
TOTAL_EXPERTS = 896
FLOWS = ("planning", "dispatch_fwd", "dispatch_bwd", "combine_fwd", "combine_bwd")


def build_configs(ep: int, experts: int | None):
    E = experts or TOTAL_EXPERTS
    assert E % ep == 0, f"E ({E}) must be divisible by ep ({ep})"
    configs = []
    for H in HIDDENS:
        for K in TOPKS:
            for S in SEQLENS:
                for ratio in RATIOS:
                    if K > E:
                        continue
                    configs.append(dict(ep=ep, H=H, K=K, S=S, E=E,
                                        unbalance_ratio=ratio))
    return configs


def bench_one(group, group_rank, R, S, K, E, H, bias_ratio, args, flows):
    """Run the five comm flows for one config; return {flow: result}.

    Plan/output tensors are pre-allocated once and the timed flows call the
    low-level ``launch_planning`` / ``launch_dispatch`` / ``launch_combine``
    directly (as the unit tests do), so the per-iteration host allocation and
    Buffer orchestration are out of the timed path. CUDA-graph capture (default)
    removes the remaining per-launch host overhead.
    """
    dev = f"npu:{torch.npu.current_device()}"
    use_udma = args.use_udma
    # Buffer-backed only: construct the Buffer first (registers VMM
    # descriptors), then init_with_buffer. Buffer itself gates use_udma on A5
    # capability and writes ASCEND_MOONEP_*_USE_UDMA, so init_with_buffer
    # below just inherits that policy.
    buffer = Buffer(S, H, K, E, R, num_sms=args.num_sms, group=group,
                    use_udma=use_udma, explicitly_destroy=True)
    ctx = buffer._require_ctx()
    if not ShmemRuntime.is_initialized():
        ShmemRuntime.init_with_buffer(group=group, enable_udma=use_udma)
    # This script uses the low-level launch_* (not buffer.dispatch), so map
    # the Buffer comm chunks to the symmetric addresses the kernels expect.
    buffer._ensure_shmem_views()

    # Shared routing (seed 1234 -> rank-shared expert popularity, rank -> per-token draws).
    topk, tpe = generate_topk_routing(S, K, E, R, bias_ratio, dev, 1234,
                                      rank=group_rank)
    hidden_sh = torch.randn(S, H, dtype=torch.bfloat16, device=dev)
    route_weights_sk = torch.rand(S, K, dtype=torch.float32, device=dev)
    grad_output = torch.randn(S, H, dtype=torch.bfloat16, device=dev)

    # Pre-allocate plan + outputs once (test_planning.py pattern): no per-call
    # allocate_planning_outputs / torch.empty in the timed path.
    plan, cu_seqlens = allocate_planning_outputs(ctx)
    topk_flat = topk.reshape(-1).contiguous()
    output_sh = torch.empty(S, H, dtype=torch.bfloat16, device=dev)
    output_sk = torch.empty(S, K, dtype=torch.float32, device=dev)

    # Global expert-load imbalance (max / mean over experts, all ranks summed).
    global_tpe = tpe.to(torch.int64)
    dist.all_reduce(global_tpe, group=group)
    load_max_mean = (
        float(global_tpe.max().item())
        / max(float(global_tpe.sum().item()) / E, 1.0)
        if int(global_tpe.sum().item()) > 0 else 0.0
    )

    results = {}
    try:
        if "planning" in flows:
            results['planning'] = measure_flow(
                "planning", "PlanningOp",
                lambda: launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan),
                group, group_rank, allow_stub=args.allow_stub,
                warmup=args.warmup, iters=args.iters, cudagraph=args.cudagraph)

        if "dispatch_fwd" in flows:
            results['dispatch_fwd'] = measure_flow(
                "dispatch_fwd", "PlanningOp",
                lambda: (launch_planning(ctx, topk_flat, tpe, cu_seqlens, plan),
                         launch_dispatch(ctx, hidden_sh, route_weights_sk, plan,
                                         build_dedup_map=True)),
                group, group_rank, allow_stub=args.allow_stub,
                warmup=args.warmup, iters=args.iters, cudagraph=args.cudagraph)

        if "dispatch_bwd" in flows:
            results['dispatch_bwd'] = measure_flow(
                "dispatch_bwd", "DispatchOp",
                lambda: launch_dispatch(ctx, grad_output, None, plan,
                                        build_dedup_map=False),
                group, group_rank, allow_stub=args.allow_stub,
                warmup=args.warmup, iters=args.iters, cudagraph=args.cudagraph)

        if "combine_fwd" in flows:
            results['combine_fwd'] = measure_flow(
                "combine_fwd", "CombineOp",
                lambda: launch_combine(ctx, output_sh, plan, output_sk=None),
                group, group_rank, allow_stub=args.allow_stub,
                warmup=args.warmup, iters=args.iters, cudagraph=args.cudagraph)

        if "combine_bwd" in flows:
            results['combine_bwd'] = measure_flow(
                "combine_bwd", "CombineOp",
                lambda: launch_combine(ctx, output_sh, plan, output_sk=output_sk),
                group, group_rank, allow_stub=args.allow_stub,
                warmup=args.warmup, iters=args.iters, cudagraph=args.cudagraph)

        # Bandwidth accounting (payload bytes / time for the timed dispatch/combine ops).
        payload_bytes = S * K * H * 2
        for key in ("dispatch_fwd", "dispatch_bwd", "combine_fwd", "combine_bwd"):
            res = results.get(key)
            if res is not None and res["status"] == "ok":
                res["GBps"] = payload_bytes / res["us"] * 1e6 / 1e9
            elif res is not None:
                res["GBps"] = float("nan")

        results["_load_max_mean"] = load_max_mean
        return results
    finally:
        # Re-init per config: finalize first (tear down symmetric mappings),
        # then buffer.destroy (inverse VMM ops), then clear the registry so the
        # next config re-registers. Must be in finally: if a flow raises
        # without cleanup, the next config's new Buffer skips
        # init_with_buffer (ShmemRuntime still marked initialized) and
        # _ensure_shmem_views misses the newly registered chunks -- the whole
        # sweep then fails in a cascade.
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        buffer.destroy()
        from npu_patch.buffer_c import reset_symmetric_descriptors
        reset_symmetric_descriptors()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--hidden", type=int, default=None,
                    help="Restrict sweep to this H.")
    ap.add_argument("--topk", type=int, default=None,
                    help="Restrict sweep to this K.")
    ap.add_argument("--unbalance-ratio", type=float, default=None,
                    help="Restrict sweep to this lognormal sigma.")
    ap.add_argument("--experts", type=int, default=None,
                    help="Override total expert count E.")
    ap.add_argument("--flows", type=str, default=",".join(FLOWS),
                    help="Comma-separated flows to run: " + ",".join(FLOWS))
    ap.add_argument("--prod", action="store_true",
                    help="Run only the production 8-card config: "
                         "S=4096, K=16, E=896, H=7168, unbalance_ratio=1.0.")
    return ap.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup()
    args.use_udma = apply_engine(args.engine)

    ep = args.ep_size if args.ep_size is not None else world_size
    assert ep >= 1, f"--ep-size must be >= 1, got {ep}"
    assert ep <= world_size, f"--ep-size {ep} exceeds world size {world_size}"
    in_group = rank < ep

    # Subgroup over ranks [0, ep); every world rank participates in new_group.
    group = dist.new_group(ranks=list(range(ep)))

    flows = [f.strip() for f in args.flows.split(",") if f.strip()]
    for f in flows:
        assert f in FLOWS, f"unknown flow {f!r}; choose from {FLOWS}"

    if args.prod:
        if PROD_E % ep != 0:
            raise SystemExit(
                f"--prod requires --ep-size dividing E={PROD_E}; got ep={ep}")
        configs = [dict(ep=ep, H=PROD_H, K=PROD_K, S=PROD_S, E=PROD_E,
                        unbalance_ratio=1.0)]
    else:
        configs = build_configs(ep, args.experts)
        configs = [c for c in configs if (
            (args.hidden is None or c["H"] == args.hidden)
            and (args.topk is None or c["K"] == args.topk)
            and (args.unbalance_ratio is None or c["unbalance_ratio"] == args.unbalance_ratio)
        )]
    if not configs:
        raise SystemExit("no configs selected (check --hidden/--topk/--unbalance-ratio)")

    if rank == 0:
        print(f"=== bench_comm: world={world_size} ep={ep} nodesize={local_rank} "
              f"flows={flows} configs={len(configs)} ===", flush=True)

    # shmem init lives inside each config's bench_one: construct Buffer ->
    # ShmemRuntime.init_with_buffer -> run -> destroy -> finalize -> reset
    # (buffer-backed only; re-init per config, setup untimed).

    csv_writer = None
    csv_file = None
    csv_header = [
        "ep", "H", "K", "S", "E", "unbalance_ratio", "load_max_mean",
        "planning_us", "dispatch_fwd_us", "dispatch_bwd_us",
        "combine_fwd_us", "combine_bwd_us",
        "dispatch_fwd_GBps", "dispatch_bwd_GBps",
        "combine_fwd_GBps", "combine_bwd_GBps",
    ]
    if rank == 0 and args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        csv_file = open(args.out, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

    if rank == 0:
        header = (f"{'ep':>3} {'H':>5} {'K':>3} {'S':>5} {'E':>4} {'unb_r':>5} "
                  f"{'mx/mean':>7} "
                  f"{'plan_us':>9} {'d_fwd':>9} {'d_bwd':>9} "
                  f"{'c_fwd':>9} {'c_bwd':>9} "
                  f"{'DfGB/s':>8} {'DbGB/s':>8} {'CfGB/s':>8} {'CbGB/s':>8}")
        print(header, flush=True)
        print("-" * len(header), flush=True)

    failures = 0
    for cfg in configs:
        res = None
        if in_group:
            try:
                res = bench_one(
                    group=group, group_rank=rank,
                    R=ep, S=cfg["S"], K=cfg["K"], E=cfg["E"], H=cfg["H"],
                    bias_ratio=cfg["unbalance_ratio"], args=args, flows=flows)
            except Exception:
                if rank == 0:
                    traceback.print_exc()
                res = None
        # Global sync so idle ranks don't race ahead / failures don't wedge the sweep.
        dist.barrier()

        if rank == 0:
            if res is None:
                failures += 1
                line = (f"{ep:>3} {cfg['H']:>5} {cfg['K']:>3} {cfg['S']:>5} "
                        f"{cfg['E']:>4} {cfg['unbalance_ratio']:>5.2f} "
                        f"{'FAILED':>25}")
                print(line, flush=True)
                continue

            def _us(key):
                return time_text(res[key]) if key in res else "  n/a"

            def _gb(key):
                return "  n/a" if key not in res else fmt4(res[key]["GBps"])

            line = (f"{ep:>3} {cfg['H']:>5} {cfg['K']:>3} {cfg['S']:>5} "
                    f"{cfg['E']:>4} {cfg['unbalance_ratio']:>5.2f} "
                    f"{res['_load_max_mean']:>7.3f} "
                    f"{_us('planning'):>9} {_us('dispatch_fwd'):>9} "
                    f"{_us('dispatch_bwd'):>9} {_us('combine_fwd'):>9} "
                    f"{_us('combine_bwd'):>9} "
                    f"{_gb('dispatch_fwd'):>8} {_gb('dispatch_bwd'):>8} "
                    f"{_gb('combine_fwd'):>8} {_gb('combine_bwd'):>8}")
            print(line, flush=True)

            if csv_writer:
                def _cv_us(key):
                    if key not in res or res[key]["status"] != "ok":
                        return ""
                    return fmt4(res[key]["us"])

                def _cv_gb(key):
                    if key not in res or res[key]["status"] != "ok":
                        return ""
                    return fmt4(res[key]["GBps"])

                csv_writer.writerow([
                    ep, cfg["H"], cfg["K"], cfg["S"], cfg["E"],
                    f"{cfg['unbalance_ratio']:.2f}",
                    f"{res['_load_max_mean']:.3f}",
                    _cv_us("planning"), _cv_us("dispatch_fwd"),
                    _cv_us("dispatch_bwd"), _cv_us("combine_fwd"),
                    _cv_us("combine_bwd"),
                    _cv_gb("dispatch_fwd"), _cv_gb("dispatch_bwd"),
                    _cv_gb("combine_fwd"), _cv_gb("combine_bwd"),
                ])
                csv_file.flush()

    dist.destroy_process_group()

    if rank == 0 and csv_file:
        csv_file.close()
        print(f"Wrote CSV to {args.out}", flush=True)
    if rank == 0 and failures:
        print(f"{failures}/{len(configs)} configs FAILED", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
