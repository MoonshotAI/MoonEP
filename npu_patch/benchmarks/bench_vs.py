"""
MoonEP-Ascend dispatch/combine benchmark: MoonEP maxvio sweep.

Ascend port of ``MoonEP/benchmarks/bench_vs_deepep.py`` with the same CLI and
methodology. DeepEP v2 is CUDA-only and unavailable on Ascend, so this benchmark
measures the MoonEP-Ascend public Buffer API alone across a MaxVio sweep
(identical parameters: S=8192, seed-1234 lognormal routing, the lognormal sigma
reverse-solved per target MaxVio, eager npu-event timing, cross-rank mean).

Reported ops (public Buffer API end-to-end):
  - plan: launch_planning (exact, separate)
  - d_f:  buffer.dispatch(zero_copy=True) [inter_rank_sync + planning + dispatch
          + epilogue], then launch_prefetch
  - d_b:  buffer.dispatch(saved plan, zero_copy=True) + launch_prefetch
  - c_f:  buffer.combine(hidden only)
  - c_b:  buffer.combine(with route_weights gather)
  - pf:   launch_prefetch separately (for its share of d_f/d_b)
grad_reduce is NOT included (overlappable, not on the MoE critical path).

Prefetch is timed through the low-level ``launch_prefetch`` on per-rank
``[epn+B, H, H']`` symmetric chunks (``make_symmetric_experts``, same as
``test_prefetch.py``). ``Buffer.prefetch_weight`` is not used here: its full
``[E+B, H, H']`` weight pools need an E+B VMM per rank that the current shmem
implementation does not support. Prefetch reads home rows locally rather than
across ranks; the dispatch/combine comm paths are the cross-rank measurements.

Bandwidth = S*K*H*2 / t (logical payload bytes, same formula as the reference).
Results are saved to a CSV on rank 0 (--out, default bench_vs_results.csv) for
later plotting; --plot optionally renders a single-library figure.

EP group = whole world (R == world_size). Single- and multi-node both work:

    single node (default; UID bootstrap, no ip_port needed):
        torchrun --nproc_per_node=8 npu_patch/benchmarks/bench_vs.py [--out r.csv] [--plot r.png]
    multi node (ip_port/DEFAULT bootstrap; when ip_port is omitted it is
    derived from MASTER_ADDR/MASTER_PORT, port + 1):
        # node 0
        MASTER_ADDR=192.168.1.10 bash npu_patch/benchmarks/run_bench.sh bench_vs.py   # NODES=2 NODE_RANK=0
        # node 1
        MASTER_ADDR=192.168.1.10 NODES=2 NODE_RANK=1 bash npu_patch/benchmarks/run_bench.sh bench_vs.py
    explicit shmem ip_port:
        torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
            --master-addr=192.168.1.10 --master-port=29500 \
            npu_patch/benchmarks/bench_vs.py --ip-port tcp://192.168.1.10:29501
"""

import argparse
import csv
import os

import torch
import torch_npu
import torch.distributed as dist

from _bench_common import (
    add_engine_args,
    apply_engine,
    setup,
    time_kernel,
)
from _bench_common import generate_topk_routing  # re-export

from npu_patch import Buffer, ShmemRuntime, launch_prefetch
from npu_patch._device import local_device_index
from npu_patch.buffer_c import (
    create_sym_tensor_from_ptr,
    create_vmm_physical,
    map_to_sym_ptr,
    reset_symmetric_descriptors,
)
from npu_patch.launchers import (
    launch_combine,
    launch_dispatch,
    launch_planning,
)
from npu_patch.planning import allocate_planning_outputs


def make_symmetric_experts(device_id, R, E, H, Hp, B):
    """Allocate a per-rank ``[epn + B, H, Hp]`` VMM chunk registered with the
    buffer-backed init. Returns the source tensor (``data_ptr`` = source VA);
    ``remote_expert`` (logical ``[E]``) and ``prefetch_buffers`` (local ``[B]``)
    are derived with ``map_to_sym_ptr`` after init_with_buffer/init_with_ip_port
    (see ``_prepare_shmem_views``).

    ``device_id`` is the node-local logical device id (``local_rank``: the
    0..N-1 visible index from ASCEND_RT_VISIBLE_DEVICES as seen by torch.npu),
    NOT the global rank (>= the local card count in multi-node) or the physical
    card id. The VMM physical allocation (``location.id``) uses this visible
    index and must be the current device.
    """
    epn = (E + R - 1) // R
    return create_vmm_physical([epn + B, H, Hp], torch.bfloat16,
                               device_id)[0]


# ---------------------------------------------------------------------------
# Shared timing harness — eager npu-event timing, cross-rank mean us
# ---------------------------------------------------------------------------

def time_op(fn, group, rank, warmup=20, iters=50, cudagraph=True):
    return time_kernel(fn, group, rank, warmup=warmup, iters=iters,
                       agg="mean", cudagraph=cudagraph)["us"]


def gbps(S, K, H, us):
    return S * K * H * 2 / (us * 1e-6) / 1e9


def _maxvio_for_sigma(sigma, S, K, E, R, dev):
    """Realized MaxVio of the lognormal routing at the given sigma, using the
    benchmark's fixed seeds (1234 shared, per-rank local) — deterministic."""
    total = torch.zeros(E, dtype=torch.int64, device=dev)
    for r in range(R):
        _, tpe = generate_topk_routing(S, K, E, R, sigma, dev, 1234, rank=r)
        total += tpe.to(torch.int64)
    return total.max().item() / (total.sum().item() / E) - 1.0


def solve_sigma_for_maxvio(target, S, K, E, R, dev):
    """Log-scale bisection for the lognormal sigma whose realized MaxVio best
    matches ``target``. Returns ``(sigma, realized_maxvio)``; if the target is
    below the uniform-sampling noise floor, returns the smallest sigma tried."""
    lo, hi = 1e-5, 10.0
    mv_lo = _maxvio_for_sigma(lo, S, K, E, R, dev)
    if mv_lo >= target:
        return lo, mv_lo
    best_sigma, best_mv = lo, mv_lo
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        mv = _maxvio_for_sigma(mid, S, K, E, R, dev)
        if abs(mv - target) < abs(best_mv - target):
            best_sigma, best_mv = mid, mv
        if abs(mv - target) <= max(0.02 * target, 0.005):
            return mid, mv
        if mv < target:
            lo = mid
        else:
            hi = mid
    return best_sigma, best_mv


# ---------------------------------------------------------------------------
# MoonEP — public Buffer API for dispatch/combine; kernel-level launches for
# the separate planning measurement and launch_prefetch for prefetch
# ---------------------------------------------------------------------------

class MoonEPRunner:
    NAME = 'moonep'

    def __init__(self, group, R, S, K, E, H, num_sms, local_rank, hp=2048,
                 use_udma=False):
        self.group = group
        self.num_sms = num_sms
        self.rank = dist.get_rank(group)
        self.R = R
        self.local_rank = local_rank
        self.hp = hp
        self.use_udma = use_udma
        self.buffer = Buffer(S, H, K, E, R, num_sms=num_sms, group=group,
                             use_udma=use_udma, explicitly_destroy=True)
        self.ctx = self.buffer._require_ctx()
        # Pre-allocate the plan + outputs once (test_planning.py pattern): the
        # timed ops reuse these instead of allocating per call.
        self.plan, self.cu_seqlens = allocate_planning_outputs(self.ctx)
        dev = f"npu:{local_rank}"
        self.output_sh = torch.empty(S, H, dtype=torch.bfloat16, device=dev)
        self.output_sk = torch.empty(S, K, dtype=torch.float32, device=dev)

        # Composite [E+B] layout expects B == epn (native precondition).
        self.epn = E // R
        self.B = int(self.ctx['B'])
        assert self.B == self.epn, (
            f"composite [E+B] layout expects B == epn, got B={self.B}, "
            f"epn={self.epn}")

        # Per-rank [epn+B, H, hp] VMM chunks registered with the buffer-backed
        # init; the logical [E] remote_expert views and local [B] slots are
        # derived by _prepare_shmem_views after init_with_buffer.
        self._weight_chunks = []
        for _ in range(3):
            self._weight_chunks.append(
                make_symmetric_experts(self.local_rank, R, E, H, hp, self.B))
        self.remote_experts = []       # 3 logical [E, H, hp] home-weight views
        self.prefetch_buffers = []     # 3 local [B, H, hp] prefetch slots
        g = torch.Generator(device=dev).manual_seed(5000 + self.rank)
        # Each rank initializes its OWN home rows: the local chunk rows [0:epn)
        # (global experts [rank*epn:(rank+1)*epn]), exactly as test_prefetch.py.
        for chunk in self._weight_chunks:
            chunk[:self.epn].copy_(
                torch.randn(self.epn, H, hp, dtype=torch.bfloat16,
                            device=dev, generator=g))
        torch.npu.synchronize()

    def _prepare_shmem_views(self):
        """After init_with_buffer: map Buffer comm chunks and weight chunks to symmetric tensors."""
        self.buffer._ensure_shmem_views()
        dev = f"npu:{self.local_rank}"
        E = self.epn * self.R
        H = int(self.ctx['H'])
        hp = self.hp
        row_bytes = H * hp * 2  # bf16
        self.remote_experts = []
        self.prefetch_buffers = []
        for chunk in self._weight_chunks:
            sym_base = map_to_sym_ptr(chunk.data_ptr(), 'weight_chunk')
            self.remote_experts.append(create_sym_tensor_from_ptr(
                sym_base, [E, H, hp], torch.bfloat16, dev))
            self.prefetch_buffers.append(create_sym_tensor_from_ptr(
                sym_base + self.epn * row_bytes,
                [self.B, H, hp], torch.bfloat16, dev))

    def prepare(self, topk, tpe, hidden, weights):
        self.hidden, self.weights = hidden, weights
        self.topk, self.tpe = topk, tpe
        self._topk_flat = topk.reshape(-1).contiguous()
        # Untimed full dispatch (low-level): materializes the plan + dedup
        # structures used by saved-plan dispatch/combine below.
        launch_planning(self.ctx, self._topk_flat, tpe, self.cu_seqlens,
                        self.plan)
        launch_dispatch(self.ctx, hidden, weights, self.plan,
                        build_dedup_map=True)
        # Byte accounting from the plan's slot table (global expert ids).
        etc_cpu = self.plan.experts_to_copy.cpu()   # [R, B] int32, -1 = idle
        self.max_recv = int((etc_cpu >= 0).sum(dim=1).max().item())
        self.shard_view = self.ctx['hidden_buf_local']
        self.weights_view = self.ctx['weights_buf_local'].view(torch.float32)
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def planning(self):
        launch_planning(self.ctx, self._topk_flat, self.tpe,
                        self.cu_seqlens, self.plan)

    def _prefetch(self):
        # Low-level launch_prefetch per weight tensor (gate/up/down), pulling
        # this rank's plan row into the local [B, H, hp] slots. Avoids
        # Buffer.prefetch_weight's full [E+B, H, hp] pools, which need an E+B
        # VMM the current shmem implementation does not support.
        local_experts = self.plan.experts_to_copy[self.rank].contiguous()
        for remote_expert, slot_buf in zip(self.remote_experts,
                                           self.prefetch_buffers):
            launch_prefetch(remote_expert, slot_buf, local_experts,
                            num_sms=self.num_sms)

    def prefetch(self):
        # launch_prefetch has no cross-rank sync of its own; pair it with a
        # small sync so timed iterations stay serialized across ranks.
        self._prefetch()

    def dispatch_fwd(self):
        # full fwd (low-level): planning + dispatch + epilogue, then prefetch.
        launch_planning(self.ctx, self._topk_flat, self.tpe,
                        self.cu_seqlens, self.plan)
        launch_dispatch(self.ctx, self.hidden, self.weights, self.plan,
                        build_dedup_map=True)
        self._prefetch()

    def dispatch_bwd(self):
        # bwd: saved plan (no planning, no dedup builder); the backward expert
        # GEMMs need the same weights, so prefetch again.
        launch_dispatch(self.ctx, self.hidden, None, self.plan,
                        build_dedup_map=False)
        self._prefetch()

    def combine_fwd(self):
        # c_f: hidden only
        launch_combine(self.ctx, self.output_sh, self.plan, output_sk=None)

    def combine_bwd(self):
        # c_b: with droute_weights gather. grad_reduce is not on the critical
        # path and is not included in this benchmark.
        launch_combine(self.ctx, self.output_sh, self.plan,
                       output_sk=self.output_sk)

    def destroy(self):
        # Release this runner's 3 weight VMM chunks via the inverse VMM ops.
        for chunk in self._weight_chunks:
            alloc = getattr(chunk, '_vmm_allocation', None)
            if alloc is not None:
                alloc.destroy()
        self.buffer.destroy()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_engine_args(ap)
    ap.add_argument('--libs', nargs='+', default=['moonep'], choices=['moonep'],
                    help='Libraries to benchmark (Ascend: only moonep).')
    ap.add_argument('--out', default='bench_vs_results.csv',
                    help='CSV output path (rank 0). Default: %(default)s')
    ap.add_argument('--plot', default=None,
                    help='If set, save the stacked-bar figure to this path '
                         '(rank 0 only).')
    ap.add_argument('--maxvios', default='0.2,1,10,20',
                    help='Comma-separated target MaxVio values; the lognormal '
                         'sigma is reverse-solved for each (default: %(default)s)')
    ap.add_argument('--experts', type=int, default=1024)
    ap.add_argument('--hidden', type=int, default=3584)
    ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--hp', type=int, default=3072,
                    help='Expert weight inner dim H\' (MoE intermediate size) '
                         'for the MoonEP prefetch op (default: %(default)s)')
    ap.add_argument('--num-sms', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--ip-port', default=None,
                    help='shmem DEFAULT-bootstrap ip_port URL for the '
                         'buffer-backed shmem init ("tcp://<rank0_ip>:<port>" '
                         'or "tcp6://[ipv6]:<port>"). Required for multi-node; '
                         'when omitted in a multi-node launch it is derived '
                         'from MASTER_ADDR/MASTER_PORT (port + 1). Single-node '
                         'launches keep the UID bootstrap and ignore this.')
    ap.add_argument('--no-cudagraph', dest='cudagraph', action='store_false',
                    help='Use the eager timing loop instead of CUDA-graph '
                         'capture (CUDA-graph removes per-launch host overhead).')
    ap.set_defaults(cudagraph=True)
    args = ap.parse_args()

    rank, world_size, local_rank = setup()
    local_rank = local_device_index(local_rank)
    use_udma = apply_engine(args.engine)
    R = world_size
    group = dist.group.WORLD
    dev = f"npu:{local_rank}"

    S, E = 8192, args.experts
    H, K = args.hidden, args.topk
    assert R >= 1 and E % R == 0, (
        f'E ({E}) must be divisible by world_size (EP size) {R}')
    B = E // R
    targets = [float(x) for x in args.maxvios.split(',')]

    # Multi-node detection: under torchrun, WORLD_SIZE > LOCAL_WORLD_SIZE
    # (procs per node) means multi-node.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size

    # Buffer-backed only: construct the runner (Buffer + 3 weight VMM chunks,
    # descriptors registered) -> init (single-node UID bootstrap / multi-node
    # ip_port bootstrap) -> _prepare_shmem_views maps symmetric tensors.
    runner = MoonEPRunner(group, R, S, K, E, H, args.num_sms, local_rank,
                          hp=args.hp, use_udma=use_udma)
    if not ShmemRuntime.is_initialized():
        if args.ip_port is not None or multi_node:
            # Multi-node: ip_port/DEFAULT bootstrap; when ip_port is omitted it
            # is derived from MASTER_ADDR/MASTER_PORT (port + 1, see
            # ShmemRuntime.init_with_ip_port).
            ShmemRuntime.init_with_ip_port(group=group, ip_port=args.ip_port,
                                           enable_udma=use_udma)
        else:
            ShmemRuntime.init_with_buffer(group=group, enable_udma=use_udma)
    runner._prepare_shmem_views()

    rows = []
    try:
        for target in targets:
            # Rank 0 reverse-solves the lognormal sigma for the target MaxVio
            # (deterministic given the fixed seeds), then broadcasts it.
            if rank == 0:
                sigma, realized = solve_sigma_for_maxvio(target, S, K, E, R, dev)
            else:
                sigma, realized = None, None
            obj = [sigma, realized]
            dist.broadcast_object_list(obj, src=0, group=group)
            sigma, realized = obj
            topk, tpe = generate_topk_routing(S, K, E, R, sigma, dev, 1234,
                                              rank=rank)
            global_tpe = tpe.to(torch.int64)
            dist.all_reduce(global_tpe, group=group)
            maxvio = global_tpe.max().item() / (global_tpe.sum().item() / E) - 1.0
            if rank == 0:
                note = '' if abs(maxvio - target) <= max(0.05 * target, 0.01) \
                    else f' (target {target} unreachable, floored)'
                print(f'--- target MaxVio={target}: sigma={sigma:.5f}, '
                      f'realized MaxVio={maxvio:.3f}{note}', flush=True)
            g = torch.Generator(device=dev).manual_seed(7777 + rank)
            hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev,
                                 generator=g)
            weights = torch.rand(S, K, dtype=torch.float32, device=dev,
                                 generator=g)

            runner.prepare(topk, tpe, hidden, weights)
            d_f = time_op(runner.dispatch_fwd, group, rank, args.warmup,
                          args.iters, cudagraph=args.cudagraph)
            d_b = time_op(runner.dispatch_bwd, group, rank, args.warmup,
                          args.iters, cudagraph=args.cudagraph)
            c_f = time_op(runner.combine_fwd, group, rank, args.warmup,
                          args.iters, cudagraph=args.cudagraph)
            c_b = time_op(runner.combine_bwd, group, rank, args.warmup,
                          args.iters, cudagraph=args.cudagraph)
            plan_us = time_op(runner.planning, group, rank, args.warmup,
                              args.iters, cudagraph=args.cudagraph)
            pf = time_op(runner.prefetch, group, rank, args.warmup, args.iters,
                         cudagraph=args.cudagraph)
            pf_gbps = (3 * runner.max_recv * H * runner.hp * 2
                       / (pf * 1e-6) / 1e9) if pf > 0 else 0.0
            rows.append(dict(lib='moonep', ep=R, E=E, H=H, K=K, sigma=sigma,
                             target_maxvio=target, maxvio=maxvio,
                             plan_us=plan_us, pf_us=pf,
                             d_f_us=d_f, d_b_us=d_b, c_f_us=c_f, c_b_us=c_b,
                             d_f_gbps=gbps(S, K, H, d_f),
                             d_b_gbps=gbps(S, K, H, d_b),
                             c_f_gbps=gbps(S, K, H, c_f),
                             c_b_gbps=gbps(S, K, H, c_b)))
            if rank == 0:
                print(f'{"moonep":>8} ep={R} E={E} H={H} K={K} sigma={sigma:.5f}: '
                      f'maxvio={maxvio:6.2f} | '
                      f'plan {plan_us:6.1f} us ({plan_us / d_f * 100:4.1f}%) | '
                      f'd_f {d_f:7.1f} us ({gbps(S, K, H, d_f):5.0f} GB/s) | '
                      f'd_b {d_b:7.1f} us ({gbps(S, K, H, d_b):5.0f} GB/s) | '
                      f'c_f {c_f:7.1f} us ({gbps(S, K, H, c_f):5.0f} GB/s) | '
                      f'c_b {c_b:7.1f} us ({gbps(S, K, H, c_b):5.0f} GB/s) | '
                      f'pf {pf:6.1f} us ({pf_gbps:4.0f} GB/s, '
                      f'{pf / d_f * 100:4.1f}%)', flush=True)
            del hidden, weights
    finally:
        # Buffer-backed teardown order: finalize (drop symmetric mappings),
        # then inverse VMM ops, then clear the descriptor registry. Kept in
        # finally so a failing target still frees shmem/VMM resources.
        if ShmemRuntime.is_initialized():
            ShmemRuntime.finalize()
        runner.destroy()
        reset_symmetric_descriptors()

    if rank == 0 and args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.out}')
    if rank == 0 and args.plot:
        plot_rows(rows, args.plot)
        print(f'wrote {args.plot}')

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Plotting (inline): stacked-bar figure, MoonEP only
# ---------------------------------------------------------------------------

_COLORS = {'comm': '#2563eb', 'planning': '#22c55e', 'prefetch': '#a855f7'}


def plot_rows(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    xs = sorted({r['maxvio'] for r in rows})
    xpos = np.arange(len(xs))
    labels = [f'{x:g}' for x in xs]
    C = _COLORS

    def get(op, x):
        return next(r[f'{op}_us'] for r in rows if r['maxvio'] == x)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    def stacked(ax, comm_fn, segs, title):
        comm = [comm_fn(x) for x in xs]
        ax.bar(xpos, comm, 0.5, color=C['comm'])
        bottoms = comm[:]
        for sname, fn in segs:
            vals = [fn(x) for x in xs]
            ax.bar(xpos, vals, 0.5, bottom=bottoms, color=C[sname])
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        for x0, t in zip(xpos, bottoms):
            ax.text(x0, t + 12, f'{t:.0f}', ha='center', va='bottom',
                    fontsize=7.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel('latency (us)')

    stacked(axes[0, 0],
            lambda x: get('d_f', x) - get('plan', x) - get('pf', x),
            [('planning', lambda x: get('plan', x)),
             ('prefetch', lambda x: get('pf', x))],
            'dispatch fwd')
    stacked(axes[0, 1],
            lambda x: get('d_b', x) - get('pf', x),
            [('prefetch', lambda x: get('pf', x))],
            'dispatch bwd')

    for ax, op, title in ((axes[1, 0], 'c_f', 'combine fwd'),
                          (axes[1, 1], 'c_b', 'combine bwd')):
        ys = [get(op, x) for x in xs]
        ax.bar(xpos, ys, 0.5, color=C['comm'])
        ax.bar_label(ax.containers[0], fmt='%.0f', fontsize=7.5, padding=2)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel('latency (us)')

    for ax in axes[-1]:
        ax.set_xlabel('MaxVio (0 = balanced, k = hottest expert at (k+1)x load)')
    E = rows[0]['E']
    H = rows[0]['H']
    K = rows[0]['K']
    fig.suptitle(f'MoonEP-Ascend — ep=8, E={E}, H={H}, K={K}, S=8192/rank',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    main()
