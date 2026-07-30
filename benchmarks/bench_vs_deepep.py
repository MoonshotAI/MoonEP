"""Aligned dispatch/combine benchmark: MoonEP vs DeepEP v2 (elastic).

Fused-permute edition. MoonEP's dispatch epilogue fuses the expert permute
into the comm path (and combine_prologue fuses the unpermute reduction), so
the only apples-to-apples DeepEP comparison is the v2 elastic path with
``do_expand=True`` (expanded dispatch) and the reduced combine. (DeepEP v1's
recv_x is grouped by source rank — the expert permute/unpermute is a separate
unfused step outside the library — so it is not compared here.)

Everything shared across both libraries:
  - identical routing matrix (tests/generate_topk_routing.py, seed 1234 shared
    across ranks for expert popularity, rank seed for per-token draws)
  - identical input hidden/weights tensors (fixed generator seeds)
  - identical timing harness (eager CUDA-event timing over back-to-back
    iterations, cross-rank mean), warmup=20, iters=50
  - identical SM budget (32) and expert_alignment/token_padding (128)

Reported ops (MoonEP runs the public Buffer API end-to-end; the [E+B]
weight pools are assembled from per-rank chunks with nvl_dist_map since
B == epn, so prefetch_weight issues real NVLink remote reads):
  - plan: MoonEP launch_planning (exact, separate)  |  v2 layout is fused
          into dispatch_impl and cannot be timed in isolation; estimated as
          full-dispatch minus cached-dispatch (comes out ~= 0)
  - d_f:  buffer.dispatch(zero_copy=True) = inter_rank_sync + planning +
          dispatch + epilogue (fused permute), then buffer.prefetch_weight
          |  v2 expanded dispatch (with topk_weights)
  - d_b:  buffer.dispatch(saved plan, zero_copy=True) + prefetch_weight
          |  v2 cached expanded dispatch (hidden-only)
  - c_f:  buffer.combine(hidden only)  |  v2 reduced combine (no topk_weights)
  - c_b:  buffer.combine(with droute_weights gather)  |  v2 reduced combine
          (with topk_weights)
  - pf:   buffer.prefetch_weight separately (for its share of d_f/d_b)
  grad_reduce is NOT included: it is overlappable with subsequent compute
  and not on the MoE critical path.

Route weights are transferred in both directions on both libraries (dispatch
scatters them, combine gathers them back).

Bandwidth = S*K*H*2 / t (logical payload bytes, same formula for both).

Single node only (ep == world size). Launch:
  torchrun --nproc_per_node=8 benchmarks/bench_vs_deepep.py [--out r.csv] [--plot r.png]
"""

import argparse
import csv
import os
import time

import torch
import torch.distributed as dist

from tests.generate_topk_routing import generate_topk_routing


# ---------------------------------------------------------------------------
# Shared timing harness — the ONLY timing method used for all three libraries
# ---------------------------------------------------------------------------

def time_op(fn, group, warmup=20, iters=50):
    """Eager CUDA-event timing: back-to-back launches, cross-rank mean us."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3
    t = torch.tensor([local_us], dtype=torch.float64, device='cuda')
    outs = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(outs, t, group=group)
    return torch.cat(outs).mean().item()


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
    matches `target`. Returns (sigma, realized_maxvio). If the target is below
    the uniform-sampling noise floor, returns the smallest sigma tried."""
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
# the separate planning/prefetch measurements (prefetch uses the same
# launch_prefetch the public prefetch_weight wraps, over a sharded NVL pool)
# ---------------------------------------------------------------------------

class MoonEPRunner:
    NAME = 'moonep'

    def __init__(self, group, R, S, K, E, H, num_sms, hp=2048):
        from moonep import Buffer
        from moonep._C import (
            FABRIC_HANDLE_BYTES as _FABRIC_HANDLE_BYTES,
            nvl_dist_alloc, nvl_release_mem_handle,
            nvl_dist_map, get_vmm_granularity
        )
        from moonep.buffer import (_all_gather_shareables, _exchange_ipc_fds,
                                   _use_fabric_for_group)
        from moonep.planning import allocate_planning_outputs, launch_planning
        from moonep.inter_rank_sync import launch_inter_rank_sync
        self._launch_planning = launch_planning
        self._launch_sync = launch_inter_rank_sync
        self._use_fabric = _use_fabric_for_group(group)
        self._alloc_chunk = lambda shape, dtype: nvl_dist_alloc(
            shape=shape, dtype=dtype, use_fabric=self._use_fabric)
        self._release = nvl_release_mem_handle
        self._dist_map = nvl_dist_map
        self._gather_shareables = _all_gather_shareables
        self._exchange_fds = _exchange_ipc_fds
        self.num_sms = num_sms
        self.rank = dist.get_rank(group)
        self.R = R
        self.buffer = Buffer(S, H, K, E, R, num_sms=num_sms, group=group,
                             explicitly_destroy=True)
        self.ctx = self.buffer._require_ctx()
        self.plan_scratch, self.cu_seqlens = allocate_planning_outputs(self.ctx)

        # --------------------------------------------------------------
        # Training-framework [E+B] weight/grad tensors (public API inputs).
        # B == epn,
        # so the buffer chunk has the same shape as an expert chunk and
        # nvl_dist_map is reused directly: the [epn, H, Hp] expert chunks of
        # the R ranks (dense global pool, remote rows walk NVLink) plus this
        # rank's own [epn] buffer chunk appended last, mapped with
        # world_size=R+1, give [E+B, H, Hp].
        # --------------------------------------------------------------
        self.epn = E // R
        self.B = int(self.ctx['B'])
        assert self.B == self.epn, (
            f"composite [E+B] layout expects B == epn, got B={self.B}, "
            f"epn={self.epn}")
        self.hp = hp
        gran = get_vmm_granularity()
        self._keepalives = []

        def build_full(dtype: torch.dtype):
            itemsize = 2 if dtype == torch.bfloat16 else 4
            chunk_bytes = self.epn * H * hp * itemsize
            assert chunk_bytes % gran == 0, (
                f"chunk bytes {chunk_bytes} not VMM-aligned ({gran})")
            # per-rank expert chunk (shared) + this rank's buffer chunk (local)
            ka_w, w_sh, w_owned = self._alloc_chunk([self.epn, H, hp], dtype)
            ka_b, b_sh, b_owned = self._alloc_chunk([self.B, H, hp], dtype)
            for ka, owned in ((ka_w, w_owned), (ka_b, b_owned)):
                self._keepalives.append(ka)
                self._release(owned)
            # exchange the expert chunk handles, then append this rank's own
            # buffer chunk as the trailing (R+1)-th chunk
            if self._use_fabric:
                all_w = self._gather_shareables(w_sh, group)
                full = self._dist_map(
                    chunk_shape=[self.epn, H, hp], dtype=dtype,
                    shareables=torch.cat(
                        [all_w, b_sh.view(1, _FABRIC_HANDLE_BYTES)], dim=0),
                    local_rank=self.rank, world_size=R + 1, use_fabric=True)
            else:
                w_fd, b_fd = int(w_sh.item()), int(b_sh.item())
                fds = self._exchange_fds(w_fd, list(range(R)), self.rank, R,
                                         group)
                os.close(w_fd)
                all_w_fds = [fds[r] for r in range(R)]
                try:
                    full = self._dist_map(
                        chunk_shape=[self.epn, H, hp], dtype=dtype,
                        shareables=torch.tensor(all_w_fds + [b_fd],
                                                dtype=torch.int64),
                        local_rank=self.rank, world_size=R + 1,
                        use_fabric=False)
                finally:
                    for fd in all_w_fds:
                        os.close(fd)
                os.close(b_fd)
            return full

        # full_weight for the 3 projections (bf16). grad_reduce is not on the
        # MoE critical path (overlappable with subsequent compute) and is not
        # included in this benchmark, so no full_grad / reduce_buffer is built.
        self.full_weights = [build_full(torch.bfloat16) for _ in range(3)]
        # fill: random weights
        for fw in self.full_weights:
            fw[self.rank * self.epn:(self.rank + 1) * self.epn].normal_()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def prepare(self, topk, tpe, hidden, weights):
        self.hidden, self.weights = hidden, weights
        self.topk, self.tpe = topk, tpe
        self._topk_flat = topk.reshape(-1).contiguous()
        # Untimed full dispatch (public API): materializes the plan + dedup
        # structures used by saved-plan dispatch/combine below.
        _, _, _, self.plan = self.buffer.dispatch(
            hidden, weights, topk, tpe, zero_copy=True, router_weights_zero_copy=True)
        # Scratch plan for the separate (exact) planning measurement.
        self._launch_planning(self.ctx, self._topk_flat, tpe,
                              self.cu_seqlens, self.plan_scratch)
        # Byte accounting from the plan's slot table (global expert ids).
        etc_cpu = self.plan.experts_to_copy.cpu()   # [R, B] int32, -1 = idle
        self.max_recv = int((etc_cpu >= 0).sum(dim=1).max().item())
        self.shard_view = self.ctx['hidden_buf_local']
        self.weights_view = self.ctx['weights_buf_local'].view(torch.float32)
        torch.cuda.synchronize()
        dist.barrier()

    def planning(self):
        self._launch_planning(self.ctx, self._topk_flat, self.tpe,
                              self.cu_seqlens, self.plan_scratch)

    def _prefetch(self):
        self.buffer.prefetch_weight(
            plan=self.plan,
            full_gate_weight=self.full_weights[0],
            full_up_weight=self.full_weights[1],
            full_down_weight=self.full_weights[2])

    def prefetch(self):
        # prefetch_weight has no cross-rank sync of its own; pair it with a
        # small sync kernel so timed iterations stay serialized across ranks
        # (same as bench_comm).
        self._prefetch()
        self._launch_sync(self.ctx)

    def dispatch_fwd(self):
        # full fwd (public API): inter_rank_sync + planning + dispatch +
        # epilogue (zero_copy, no boundary copy), then prefetch weight.
        _, _, _, self.plan = self.buffer.dispatch(
            self.hidden, self.weights, self.topk, self.tpe,
            zero_copy=True, router_weights_zero_copy=True)
        self._prefetch()

    def dispatch_bwd(self):
        # bwd: saved plan (no planning, no dedup builder), zero_copy; the
        # backward expert GEMMs need the same weights, so prefetch again.
        self.buffer.dispatch(self.hidden, plan=self.plan, zero_copy=True)
        self._prefetch()

    def combine_fwd(self):
        # c_f: hidden only
        self.buffer.combine(plan=self.plan, hidden_nvsh=self.shard_view,
                            zero_copy=True)

    def combine_bwd(self):
        # c_b: with droute_weights gather. grad_reduce is not on the critical
        # path and is not included in this benchmark.
        self.buffer.combine(plan=self.plan, hidden_nvsh=self.shard_view,
                            route_weights_nvs=self.weights_view,
                            zero_copy=True, router_weights_zero_copy=True)

    def destroy(self):
        self.buffer.destroy()


# ---------------------------------------------------------------------------
# DeepEP v2 (elastic) — expanded dispatch / reduced combine (fused permute)
# ---------------------------------------------------------------------------

class DeepEPV2Runner:
    NAME = 'deepep_v2'

    def __init__(self, group, R, S, K, E, H, num_sms):
        import deep_ep
        self.deep_ep = deep_ep
        self.S, self.E, self.num_sms = S, E, num_sms
        self.buffer = deep_ep.ElasticBuffer(
            group, num_max_tokens_per_rank=S, hidden=H,
            deterministic=False, allow_hybrid_mode=True,
            allow_multiple_reduction=True, prefer_overlap_with_compute=False,
            sl_idx=0, num_allocated_qps=0, explicitly_destroy=True,
            num_gpu_timeout_secs=100, num_cpu_timeout_secs=100)
        self.num_qps = self.buffer.get_theoretical_num_qps(num_sms)

    def _expanded_args(self):
        return dict(num_sms=self.num_sms, num_qps=self.num_qps,
                    num_max_tokens_per_rank=self.S, num_experts=self.E,
                    expert_alignment=128, async_with_compute_stream=0,
                    allocate_on_comm_stream=0, do_handle_copy=1, do_cpu_sync=0,
                    do_expand=True, use_tma_aligned_col_major_sf=True)

    def prepare(self, topk, tpe, hidden, weights):
        self.topk_idx = topk.to(self.deep_ep.topk_idx_t)
        self.hidden, self.weights = hidden, weights
        recv = self.buffer.dispatch(x=hidden, topk_idx=self.topk_idx,
                                    topk_weights=weights, **self._expanded_args())
        self.recv_x, self.recv_topk_weights, self.handle = recv[0], recv[2], recv[3]
        torch.cuda.synchronize()
        dist.barrier()

    def dispatch_fwd(self):
        self.buffer.dispatch(x=self.hidden, topk_idx=self.topk_idx,
                             topk_weights=self.weights, **self._expanded_args())

    def dispatch_bwd(self):
        self.buffer.dispatch(x=self.hidden, num_sms=self.num_sms,
                             num_qps=self.num_qps, async_with_compute_stream=0,
                             allocate_on_comm_stream=0,
                             do_expand=True, use_tma_aligned_col_major_sf=True,
                             do_zero_padding=True, handle=self.handle)

    def combine_fwd(self):
        # c_f: hidden only
        self.buffer.combine(x=self.recv_x, handle=self.handle,
                            num_sms=self.num_sms, num_qps=self.num_qps,
                            async_with_compute_stream=0,
                            allocate_on_comm_stream=0)

    def combine_bwd(self):
        # c_b: with topk_weights reduction
        self.buffer.combine(x=self.recv_x, topk_weights=self.recv_topk_weights,
                            handle=self.handle, num_sms=self.num_sms,
                            num_qps=self.num_qps, async_with_compute_stream=0,
                            allocate_on_comm_stream=0)

    def destroy(self):
        self.buffer.destroy()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

RUNNERS = {
    'moonep': MoonEPRunner,
    'v2': DeepEPV2Runner,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--libs', nargs='+', default=['moonep', 'v2'],
                    choices=list(RUNNERS))
    ap.add_argument('--out', default=None)
    ap.add_argument('--plot', default=None,
                    help='If set, save the stacked-bar comparison figure to '
                         'this path (rank 0 only).')
    ap.add_argument('--maxvios', default='0.2,1,10,20',
                    help='Comma-separated target MaxVio values; the lognormal '
                         'sigma is reverse-solved for each (default: %(default)s)')
    ap.add_argument('--experts', type=int, default=384)
    ap.add_argument('--hidden', type=int, default=7168)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--hp', type=int, default=2048,
                    help='Expert weight inner dim H\' (MoE intermediate size) '
                         'for the MoonEP prefetch op (default: %(default)s)')
    ap.add_argument('--num-sms', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=50)
    args = ap.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    R = dist.get_world_size()
    assert R == 8, f'ep=8 only: world_size must be 8, got {R}'
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{local_rank}')

    S, E = 8192, args.experts
    H, K = args.hidden, args.topk
    targets = [float(x) for x in args.maxvios.split(',')]

    def make_runner(lib):
        if lib == 'moonep':
            return MoonEPRunner(group, R, S, K, E, H, args.num_sms, hp=args.hp)
        return RUNNERS[lib](group, R, S, K, E, H, args.num_sms)

    # Buffers depend only on (S, H, K, E, num_sms) — not on the routing —
    # so each library's buffer is created ONCE and reused across MaxVio points.
    t0 = time.perf_counter()
    runners = {lib: make_runner(lib) for lib in args.libs}
    t_create = time.perf_counter() - t0
    if rank == 0:
        print(f'[phase] create buffers (once): {t_create:.2f}s', flush=True)

    rows = []
    for target in targets:
        # ---- shared inputs: identical routing + data for all libraries ----
        # Rank 0 reverse-solves the lognormal sigma for the target MaxVio
        # (deterministic given the fixed seeds), then broadcasts it.
        t0 = time.perf_counter()
        if rank == 0:
            sigma, realized = solve_sigma_for_maxvio(target, S, K, E, R, dev)
        else:
            sigma, realized = None, None
        obj = [sigma, realized]
        dist.broadcast_object_list(obj, src=0)
        sigma, realized = obj
        t_solve = time.perf_counter() - t0
        topk, tpe = generate_topk_routing(S, K, E, R, sigma, dev, 1234, rank=rank)
        # MaxVio = (max_i Load_i - mean_i Load_i) / mean_i Load_i over the
        # global per-expert routed-token counts: 0 = perfectly balanced,
        # k = the hottest expert carries (k+1)x the balanced load.
        global_tpe = tpe.to(torch.int64)
        dist.all_reduce(global_tpe, group=group)
        maxvio = global_tpe.max().item() / (global_tpe.sum().item() / E) - 1.0
        if rank == 0:
            note = '' if abs(maxvio - target) <= max(0.05 * target, 0.01) \
                else f' (target {target} unreachable, floored)'
            print(f'--- target MaxVio={target}: sigma={sigma:.5f}, '
                  f'realized MaxVio={maxvio:.3f}{note} '
                  f'[solve {t_solve:.2f}s]', flush=True)
        g = torch.Generator(device=dev).manual_seed(7777 + rank)
        hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=g)
        weights = torch.rand(S, K, dtype=torch.float32, device=dev, generator=g)

        for lib in args.libs:
            runner = runners[lib]
            t0 = time.perf_counter()
            runner.prepare(topk, tpe, hidden, weights)
            t_prepare = time.perf_counter() - t0
            t0 = time.perf_counter()
            d_f = time_op(runner.dispatch_fwd, group, args.warmup, args.iters)
            d_b = time_op(runner.dispatch_bwd, group, args.warmup, args.iters)
            c_f = time_op(runner.combine_fwd, group, args.warmup, args.iters)
            c_b = time_op(runner.combine_bwd, group, args.warmup, args.iters)
            t_ops = time.perf_counter() - t0
            if hasattr(runner, 'planning'):
                # MoonEP: planning measured exactly and separately.
                plan_us = time_op(runner.planning, group, args.warmup, args.iters)
                pf = time_op(runner.prefetch, group, args.warmup, args.iters)
                pf_gbps = (3 * runner.max_recv * H * runner.hp * 2
                           / (pf * 1e-6) / 1e9) if pf > 0 else 0.0
            else:
                # v2: layout is FUSED into dispatch_impl — the public API has
                # no separate layout call (unlike v1's get_dispatch_layout),
                # so it cannot be timed in isolation. The best available
                # estimate is full-dispatch minus cached-dispatch, which comes
                # out ~= 0 — i.e. v2's layout cost is negligible. No prefetch op.
                plan_us = max(d_f - d_b, 0.0)
                pf, pf_gbps = 0.0, 0.0
            rows.append(dict(lib=lib, ep=R, E=E, H=H, K=K, sigma=sigma,
                             target_maxvio=target, maxvio=maxvio,
                             plan_us=plan_us, pf_us=pf,
                             d_f_us=d_f, d_b_us=d_b, c_f_us=c_f, c_b_us=c_b,
                             d_f_gbps=gbps(S, K, H, d_f),
                             d_b_gbps=gbps(S, K, H, d_b),
                             c_f_gbps=gbps(S, K, H, c_f),
                             c_b_gbps=gbps(S, K, H, c_b)))
            if rank == 0:
                print(f'{lib:>8} ep={R} E={E} H={H} K={K} sigma={sigma:.5f}: '
                      f'maxvio={maxvio:6.2f} | '
                      f'plan {plan_us:6.1f} us ({plan_us / d_f * 100:4.1f}%) | '
                      f'd_f {d_f:7.1f} us ({gbps(S, K, H, d_f):5.0f} GB/s) | '
                      f'd_b {d_b:7.1f} us ({gbps(S, K, H, d_b):5.0f} GB/s) | '
                      f'c_f {c_f:7.1f} us ({gbps(S, K, H, c_f):5.0f} GB/s) | '
                      f'c_b {c_b:7.1f} us ({gbps(S, K, H, c_b):5.0f} GB/s) | '
                      f'pf {pf:6.1f} us ({pf_gbps:4.0f} GB/s, '
                      f'{pf / d_f * 100:4.1f}%) '
                      f'[prepare {t_prepare:.2f}s ops {t_ops:.2f}s]',
                      flush=True)
        del hidden, weights

    t0 = time.perf_counter()
    for runner in runners.values():
        runner.destroy()
    if rank == 0:
        print(f'[phase] destroy buffers: {time.perf_counter() - t0:.2f}s',
              flush=True)

    if rank == 0 and args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.out}')

    if rank == 0 and args.plot:
        plot_rows(rows, args.plot)
        print(f'wrote {args.plot}')

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Plotting (inline): stacked-bar comparison figure
# ---------------------------------------------------------------------------

_COLORS = {'comm': '#2563eb', 'planning': '#22c55e', 'prefetch': '#a855f7',
           'v2': '#f59e0b'}


def plot_rows(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np  # noqa: F401 (used by bar positions)

    pairs = sorted({(r['maxvio'], r['target_maxvio']) for r in rows})
    xs = [m for m, _ in pairs]
    labels = [f'{t:g}' for _, t in pairs]
    C = _COLORS

    def get(lib, op, x):
        return next(r[f'{op}_us'] for r in rows
                    if r['lib'] == lib and r['maxvio'] == x)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True)
    width = 0.35
    xpos = np.arange(len(xs))
    legend_patches = {}

    def stacked(ax, comm_fn, segs, title, total_fn):
        ymin_candidates = []
        for i, lib in enumerate(['moonep', 'v2']):
            comm_vals = [comm_fn(lib, x) for x in xs]
            ymin_candidates += comm_vals
            color = C['comm'] if lib == 'moonep' else C['v2']
            ax.bar(xpos + (i - 0.5) * width, comm_vals, width, color=color)
            legend_patches.setdefault(
                'MoonEP comm' if lib == 'moonep' else 'DeepEP v2',
                plt.Rectangle((0, 0), 1, 1, fc=color))
            bottoms = comm_vals[:]
            for sname, fn in segs:
                vals = [fn(lib, x) for x in xs]
                ax.bar(xpos + (i - 0.5) * width, vals, width, bottom=bottoms,
                       color=C[sname])
                bottoms = [b + v for b, v in zip(bottoms, vals)]
                if lib == 'moonep':
                    legend_patches.setdefault(
                        f'MoonEP {sname}',
                        plt.Rectangle((0, 0), 1, 1, fc=C[sname]))
            for x0, t in zip(xpos + (i - 0.5) * width,
                             [total_fn(lib, x) for x in xs]):
                ax.text(x0, t + 12, f'{t:.0f}', ha='center', va='bottom',
                        fontsize=7.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel('latency (us)')
        ymax = max(total_fn(l, x) for l in ('moonep', 'v2') for x in xs)
        ax.set_ylim(min(ymin_candidates) * 0.85, ymax * 1.12)

    stacked(axes.flat[0],
            lambda lib, x: get(lib, 'd_f', x) - get(lib, 'plan', x) - get(lib, 'pf', x),
            [('planning', lambda lib, x: get(lib, 'plan', x)),
             ('prefetch', lambda lib, x: get(lib, 'pf', x))],
            'dispatch fwd', lambda lib, x: get(lib, 'd_f', x))

    stacked(axes.flat[1],
            lambda lib, x: get(lib, 'd_b', x) - get(lib, 'pf', x),
            [('prefetch', lambda lib, x: get(lib, 'pf', x))],
            'dispatch bwd', lambda lib, x: get(lib, 'd_b', x))

    ax = axes.flat[2]
    for i, lib in enumerate(['moonep', 'v2']):
        ys = [get(lib, 'c_f', x) for x in xs]
        bars = ax.bar(xpos + (i - 0.5) * width, ys, width,
                      color=C['comm'] if lib == 'moonep' else C['v2'])
        ax.bar_label(bars, fmt='%.0f', fontsize=7.5, padding=2)
    ax.set_title('combine fwd', fontsize=11)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylabel('latency (us)')
    all_y = [r['c_f_us'] for r in rows]
    ax.set_ylim(min(all_y) * 0.92, max(all_y) * 1.06)

    ax = axes.flat[3]
    for i, lib in enumerate(['moonep', 'v2']):
        ys = [get(lib, 'c_b', x) for x in xs]
        bars = ax.bar(xpos + (i - 0.5) * width, ys, width,
                      color=C['comm'] if lib == 'moonep' else C['v2'])
        ax.bar_label(bars, fmt='%.0f', fontsize=7.5, padding=2)
    ax.set_title('combine bwd', fontsize=11)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylabel('latency (us)')
    all_y = [r['c_b_us'] for r in rows]
    ax.set_ylim(min(all_y) * 0.92, max(all_y) * 1.06)

    for ax in axes[-1]:
        ax.set_xlabel('MaxVio (0 = balanced, k = hottest expert at (k+1)x load)')

    order = ['MoonEP comm', 'MoonEP planning', 'MoonEP prefetch', 'DeepEP v2']
    fig.legend([legend_patches[k] for k in order], order,
               loc='lower center', ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.028))
    fig.text(0.5, 0.005,
             '* grad_reduce is not on the MoE critical path '
             '(overlappable with subsequent compute) and is not included '
             'in this benchmark.',
             ha='center', fontsize=8.5, style='italic', color='#444444')
    E = rows[0]['E']
    H = rows[0]['H']
    K = rows[0]['K']
    fig.suptitle(f'MoonEP vs DeepEP v2 — ep=8, E={E}, H={H}, K={K}, '
                 f'S=8192/rank, 32 SMs', fontsize=11)
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    main()
