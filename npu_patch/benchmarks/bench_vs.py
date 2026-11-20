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
grad_reduce is not timed but is checked under ``--verify``.

``--pf-overlap`` overlaps async prefetch (no rank barriers) with synchronous
dispatch on the main stream; the standalone pf number is unchanged.

``--verify`` runs a correctness check per MaxVio point before timing (plan,
dispatch hidden/weights, combine hidden/weights, prefetch slots, grad_reduce)
plus a latency sanity check. ``--verify-stages`` selects a subset.

Prefetch is timed through the low-level ``launch_prefetch`` on per-rank
``[epn+B, H, H']`` symmetric chunks (``make_symmetric_experts``, same as
``test_prefetch.py``). ``Buffer.prefetch_weight`` is not used here: its full
``[E+B, H, H']`` weight pools need an E+B VMM per rank that the current shmem
implementation does not support. Prefetch reads home rows locally rather than
across ranks; the dispatch/combine comm paths are the cross-rank measurements.

Prefetch bandwidth reports both ends: ingress (``pf_in_gbps``, experts pulled
in) and egress (``pf_out_gbps``, home rows served to peers).

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
import inspect
import math
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
from _verify_reference import (
    dispatch_expectation,
    plan_dedup_pairs,
    plan_expectation,
    row_fingerprint,
)

from npu_patch import Buffer, ShmemRuntime, launch_prefetch
from npu_patch._device import local_device_index
from npu_patch.buffer_c import (
    create_sym_tensor_from_ptr,
    create_vmm_physical,
    map_to_sym_ptr,
    reset_symmetric_descriptors,
    sym_tensor,
)
from npu_patch.launchers import (
    launch_combine,
    launch_dispatch,
    launch_grad_reduce,
    launch_planning,
)
from npu_patch.planning import allocate_planning_outputs

_PREFETCH_HAS_RANK_BARRIERS = (
    'rank_barriers' in inspect.signature(launch_prefetch).parameters
)


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


# --verify helpers

# Hidden-row fingerprints are bit-identical on both ends; the tolerance only
# absorbs a different summation blocking for [S, H] vs [NvS, H] reductions.
_HIDDEN_FP_RTOL = 1e-3

def _report_ok(label, ok, rank, group, dev):
    """All-reduce the pass flag (MIN) and print the verdict on rank 0."""
    t = torch.tensor([1 if ok else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    passed = int(t.item()) == 1
    if rank == 0:
        print(f'[Verify] {label}: {"PASS" if passed else "FAIL"}', flush=True)
    return passed


def _precision_stats(real, golden):
    """Return (max_abs, max_rel, n_bad); bad needs both errors >= 1e-2."""
    ae = (real.float() - golden.float()).abs()
    re = ae / (golden.float().abs() + 1e-8)
    bad = (ae >= 1e-2) & (re >= 1e-2)
    return ae.max().item(), re.max().item(), int(bad.sum().item())


def _grad_value_row(local_expert, H, Hp, dev):
    """Deterministic fp32 home-grad row (dyadic, exact accumulation)."""
    row = torch.arange(H, dtype=torch.float32, device=dev).view(H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, Hp)
    return local_expert * 17.0 + row * 0.125 + col * 0.0078125


def _slot_value_row(src_rank, b, H, Hp, dev):
    """fp32 reduce-slot row that ``src_rank`` would hold in local slot ``b``."""
    row = torch.arange(H, dtype=torch.float32, device=dev).view(H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=dev).view(1, Hp)
    return (src_rank + 1) * 101.0 + b * 11.0 + row * 0.03125 + col * 0.00390625



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
                 use_udma=False, pf_overlap=False, verify=False,
                 verify_stages=None, verify_trace=False):
        if pf_overlap and not _PREFETCH_HAS_RANK_BARRIERS:
            raise RuntimeError(
                '--pf-overlap requires launch_prefetch rank_barriers support')
        self.group = group
        self.num_sms = num_sms
        self.rank = dist.get_rank(group)
        self.R = R
        self.S = S
        self.K = K
        self.E = E
        self.H = H
        self.local_rank = local_rank
        self.hp = hp
        self.use_udma = use_udma
        self.verify = verify
        # --verify stage selection; trace prints per-rank markers per op.
        self.verify_stages = set(
            x.strip() for x in (verify_stages or
                                ('plan', 'dispatch', 'grad_reduce',
                                 'prefetch', 'combine'))
            if x.strip())
        self.verify_trace = verify_trace
        # --pf-overlap: async prefetch on a side stream, sync dispatch, join.
        self.pf_overlap = pf_overlap
        self.pf_stream = torch.npu.Stream() if pf_overlap else None
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

        # --verify grad_reduce needs extra fp32 chunks (~11 GB); allocate only
        # when that stage runs.
        self._grad_chunks = []
        self.expert_grads = None
        self.reduce_buffers = None
        self._need_grad_chunks = verify and ('grad_reduce' in
                                             self.verify_stages)
        if self._need_grad_chunks:
            assert H % 128 == 0 and hp % 128 == 0, (
                f'grad_reduce verify needs H ({H}) and hp ({hp}) multiples of '
                f'128')
            self._grad_chunks.append(
                create_vmm_physical([self.epn, H, hp], torch.float32,
                                    local_rank)[0])
            self._grad_chunks.append(
                create_vmm_physical([self.B, H, hp], torch.float32,
                                    local_rank)[0])
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
        if self._need_grad_chunks:
            self.expert_grads = sym_tensor(self._grad_chunks[0])
            self.reduce_buffers = sym_tensor(self._grad_chunks[1])

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
        valid = etc_cpu >= 0
        # ingress: experts this rank pulls in (hottest consumer).
        self.max_recv = int(valid.sum(dim=1).max().item())
        # egress: home rows served to other ranks; exclude local reads.
        owners = torch.div(etc_cpu.clamp_min(0), self.epn, rounding_mode='floor')
        dst_ranks = torch.arange(self.R).view(self.R, 1).expand_as(etc_cpu)
        remote = valid & (owners != dst_ranks)
        self.max_send = int(
            torch.bincount(owners[remote].reshape(-1),
                           minlength=self.R).max().item()
        ) if bool(remote.any()) else 0
        self.shard_view = self.ctx['hidden_buf_local']
        self.weights_view = self.ctx['weights_buf_local'].view(torch.float32)
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def planning(self):
        launch_planning(self.ctx, self._topk_flat, self.tpe,
                        self.cu_seqlens, self.plan)

    def _prefetch(self, rank_barriers=True):
        # Low-level launch_prefetch per weight tensor (gate/up/down), pulling
        # this rank's plan row into the local [B, H, hp] slots. Avoids
        # Buffer.prefetch_weight's full [E+B, H, hp] pools, which need an E+B
        # VMM the current shmem implementation does not support.
        # rank_barriers=False drops both barriers; home weights are static and
        # slots are never consumed, so the event join suffices.
        local_experts = self.plan.experts_to_copy[self.rank].contiguous()
        barrier_args = ({'rank_barriers': rank_barriers}
                        if _PREFETCH_HAS_RANK_BARRIERS else {})
        for remote_expert, slot_buf in zip(self.remote_experts,
                                           self.prefetch_buffers):
            launch_prefetch(remote_expert, slot_buf, local_experts,
                            num_sms=self.num_sms, **barrier_args)

    def prefetch(self):
        # launch_prefetch carries its own cross-rank pre/post barriers inside
        # the kernel; the launch is stream-ordered and host-async.
        self._prefetch()

    def _prefetch_async_start(self):
        """Fork prefetch on the side stream and return its event.

        The side stream waits for the current stream, then the caller joins
        the event. Mirrors prefetch_projections(async_finish=True,
        rank_barriers=False).
        """
        cur = torch.npu.current_stream()
        self.pf_stream.wait_stream(cur)
        with torch.npu.stream(self.pf_stream):
            self._prefetch(rank_barriers=False)
        return self.pf_stream.record_event()

    def dispatch_fwd(self):
        # full fwd (low-level): planning + dispatch + epilogue + prefetch.
        launch_planning(self.ctx, self._topk_flat, self.tpe,
                        self.cu_seqlens, self.plan)
        if self.pf_overlap:
            # Async prefetch on the side stream, sync dispatch, then join.
            done = self._prefetch_async_start()
            launch_dispatch(self.ctx, self.hidden, self.weights, self.plan,
                            build_dedup_map=True)
            done.wait(torch.npu.current_stream())
            return
        launch_dispatch(self.ctx, self.hidden, self.weights, self.plan,
                        build_dedup_map=True)
        self._prefetch()

    def dispatch_bwd(self):
        # bwd: saved plan (no planning, no dedup builder); the backward expert
        # GEMMs need the same weights, so prefetch again.
        if self.pf_overlap:
            done = self._prefetch_async_start()
            launch_dispatch(self.ctx, self.hidden, None, self.plan,
                            build_dedup_map=False)
            done.wait(torch.npu.current_stream())
            return
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

    # --verify checks

    def _verify_stage_boundary(self):
        """Drain this rank and fence ranks between verify stages.

        HCCL is collective-safe even if a previous stage failed.
        """
        torch.npu.synchronize()
        dist.barrier(group=self.group)

    def _trace(self, msg):
        if self.verify_trace:
            print(f'[rank {self.rank}] {msg}', flush=True)

    def verify_precision(self, hidden, weights):
        """Run the selected checks once, after ``prepare()``.

        Returns True iff all run checks pass on every rank. ``--verify-stages``
        selects stages; ``--verify-trace`` prints per-rank markers. Stage order
        is fixed (plan, dispatch, grad_reduce, prefetch, combine), independent
        of the order the stages were listed in.
        """
        ok = True
        if 'plan' in self.verify_stages:
            self._trace('verify plan: start')
            ok = self._verify_plan() and ok
            self._trace('verify plan: done')
            self._verify_stage_boundary()
        if 'dispatch' in self.verify_stages:
            self._trace('verify dispatch: start')
            ok = self._verify_dispatch() and ok
            self._trace('verify dispatch: done')
            self._verify_stage_boundary()
        if 'grad_reduce' in self.verify_stages:
            self._trace('verify grad_reduce: start')
            ok = self._verify_grad_reduce() and ok
            self._trace('verify grad_reduce: done')
            self._verify_stage_boundary()
        if 'prefetch' in self.verify_stages:
            self._trace('verify prefetch: start')
            ok = self._verify_prefetch() and ok
            self._trace('verify prefetch: done')
            self._verify_stage_boundary()
        if 'combine' in self.verify_stages:
            self._trace('verify combine: start')
            ok = self._verify_combine(hidden, weights) and ok
            self._trace('verify combine: done')
        return ok

    def _sync_ranks(self):
        """Drain the local device, then rendezvous with the peer ranks."""
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def _all_gather_stack(self, tensor):
        """Collect this rank's tensor from every rank -> host ``[R, *shape]``."""
        parts = [torch.empty_like(tensor) for _ in range(self.R)]
        dist.all_gather(parts, tensor.contiguous(), group=self.group)
        return torch.stack(parts).cpu()

    def _verify_plan(self):
        """Check the planning kernel against the torch planning reference.

        The reference model is vectorized (``_verify_reference``): it rebuilds
        the greedy MaxVio allocation, the prefetch layout and the per-route
        ``dst``/duplicate encoding from the gathered routing, then compares
        every active plan field (``dst`` / ``cu_seqlens`` / ``experts_to_copy``
        / ``dup_*``). ``zero_fill_ranges`` and ``remote_stats`` are not written
        by the Ascend planning kernel and are unused downstream, so they are
        not checked. Slot coverage, rank capacity, per-expert token
        conservation and the semantic duplicate table are checked as well, so
        a bad balance policy, prefetch selection, segment layout or dedup
        encoding all fail here.
        """
        dev = f"npu:{self.local_rank}"
        R, S, K, E = self.R, self.S, self.K, self.E
        NvS = int(self.ctx['NvS'])

        # Fresh planning: the kernel is collective, so all ranks call it.
        self.planning()
        self._sync_ranks()

        topk_all = self._all_gather_stack(self.topk.reshape(S, K))
        tpe_all = self._all_gather_stack(self.tpe)
        dst_all = self._all_gather_stack(self.plan.dst).to(torch.int64)
        cu_all = self._all_gather_stack(self.cu_seqlens)
        etc_all = self._all_gather_stack(self.plan.experts_to_copy)
        # Derive the routing counts from top-k alone: the plan model then
        # depends only on the tensor planning actually consumes, and the
        # comparison below also validates the tokens_per_expert input.
        tpe_from_topk = torch.stack([
            torch.bincount(topk_all[r].reshape(-1), minlength=E)
            for r in range(R)
        ])
        try:
            exp = plan_expectation(topk_all, tpe_from_topk, rank=self.rank,
                                   NvS=NvS)
        except Exception as exc:  # local reference model; deterministic per rank
            return _report_ok(f'plan            (model error: {exc!r})', False,
                              self.rank, self.group, dev)

        # Decode the kernel's dst to recover the plan's own allocation and slot
        # usage: this checks the balance policy/segment layout independently of
        # the exact dst comparison below.
        kernel_raw = torch.where(dst_all >= 0, dst_all, -dst_all - 1)
        kernel_dest = kernel_raw // NvS
        kernel_loff = kernel_raw - kernel_dest * NvS
        valid_dest = bool(((kernel_dest >= 0) & (kernel_dest < R)).all())
        if valid_dest:
            alloc_routes = torch.bincount(
                topk_all.reshape(-1) * R + kernel_dest.reshape(-1),
                minlength=E * R).view(E, R)
            local = kernel_dest == self.rank
            slots = torch.bincount(kernel_loff[local], minlength=NvS)
            capacity = bool((alloc_routes.sum(0) == NvS).all())
            slots_ok = slots.numel() == NvS and bool((slots == 1).all())
        else:
            # A broken dst can point outside [0, R); skip the derived checks
            # (the exact dst comparison reports the failure) instead of
            # indexing out of range.
            alloc_routes = torch.full((E, R), -1, dtype=torch.int64)
            slots = torch.zeros(NvS, dtype=torch.int64)
            capacity, slots_ok = False, False

        dedup_pairs_plan, dedup_counts_plan, dedup_errors = plan_dedup_pairs(
            self.plan.dup_groups.cpu(), self.plan.dup_loffs.cpu(),
            self.plan.dup_counts.cpu(), NvS)

        checks = {
            'routing': torch.equal(tpe_all.to(torch.int64), tpe_from_topk),
            'alloc': torch.equal(alloc_routes, exp.alloc),
            'capacity': capacity,
            'dst': torch.equal(dst_all, exp.dst),
            'cu_seqlens': torch.equal(cu_all, exp.cu_seqlens),
            'prefetch': torch.equal(etc_all,
                                   exp.experts_to_copy.expand_as(etc_all)),
            'slots': slots_ok,
            'dedup': (not dedup_errors and dedup_counts_plan == exp.dedup_counts
                      and torch.equal(dedup_pairs_plan, exp.dedup_pairs)),
        }
        ok = all(checks.values())
        detail = ' '.join(f'{name}={"ok" if passed else "FAIL"}'
                          for name, passed in checks.items())
        if dedup_errors:
            detail += f' [{"; ".join(dedup_errors[:2])}]'
        if not checks['dst']:
            detail += f' dst_bad={int((dst_all != exp.dst).sum())}'
        if not checks['alloc']:
            detail += f' alloc_bad={int((alloc_routes != exp.alloc).sum())}'
        if not checks['slots']:
            detail += (f' slot_bad={int((slots != 1).sum())}'
                       f' covered={int(slots.sum())}/{NvS}')
        if not checks['cu_seqlens']:
            detail += f' cu_bad={int((cu_all != exp.cu_seqlens).sum())}'
        if not checks['prefetch']:
            bad = int((etc_all
                       != exp.experts_to_copy.expand_as(etc_all)).sum())
            detail += f' pf_bad={bad}'
        return _report_ok(f'plan            ({detail})', ok, self.rank,
                          self.group, dev)

    def _verify_dispatch(self):
        """Check the dispatch kernel output against the plan-derived slots.

        The expected slot payload is built from the gathered ``dst``: every
        route decodes to ``(dest_rank, loff)``, so the destination rank's
        ``loff`` slot must hold that route's weight and the hidden row of its
        token (duplicates store their own weight but share the token's hidden
        row through the epilogue). Weights are compared exactly; hidden rows
        are compared through a per-row fingerprint (sum / abs-sum / ramp-dot /
        sum of squares), so no full weight-sized gather is needed. Both
        benchmark dispatch paths are exercised: ``dispatch_fwd`` (hidden +
        weights) and ``dispatch_bwd`` (hidden only, saved plan).
        """
        dev = f"npu:{self.local_rank}"
        S, K = self.S, self.K
        NvS = int(self.ctx['NvS'])

        self.planning()
        launch_dispatch(self.ctx, self.hidden, self.weights, self.plan,
                        build_dedup_map=True)
        self._sync_ranks()

        dst_all = self._all_gather_stack(self.plan.dst)
        weights_all = self._all_gather_stack(self.weights.reshape(S, K))
        hidden_fp_all = self._all_gather_stack(row_fingerprint(self.hidden))
        try:
            exp = dispatch_expectation(dst_all, weights_all, hidden_fp_all,
                                       rank=self.rank, NvS=NvS)
        except Exception as exc:  # local reference model; deterministic per rank
            return _report_ok(f'dispatch fwd    (model error: {exc!r})', False,
                              self.rank, self.group, dev)

        actual_w = self.weights_view.cpu()
        actual_fp = row_fingerprint(self.shard_view).cpu()
        rel = ((actual_fp - exp.hidden_fp).abs()
               / (exp.hidden_fp.abs() + 1.0)).max().item() if NvS else 0.0
        fwd = {
            'slots': exp.covered == NvS and exp.collisions == 0,
            'weights': torch.equal(actual_w, exp.weight),
            'hidden': rel <= _HIDDEN_FP_RTOL,
        }
        ok_fwd = all(fwd.values())
        detail = (f'slots={exp.covered}/{NvS} collisions={exp.collisions} '
                  f'weights={"ok" if fwd["weights"] else "FAIL"} '
                  f'hidden_fp_rel={rel:.3g}')
        if not fwd['weights']:
            detail += f' w_bad={int((actual_w != exp.weight).sum())}'
        ok_fwd = _report_ok(f'dispatch fwd    ({detail})', ok_fwd, self.rank,
                            self.group, dev)

        # dispatch_bwd: saved plan, hidden only (the weights buffer is stale
        # here, so only the payload is checked).
        launch_dispatch(self.ctx, self.hidden, None, self.plan,
                        build_dedup_map=False)
        self._sync_ranks()
        actual_fp_b = row_fingerprint(self.shard_view).cpu()
        rel_b = ((actual_fp_b - exp.hidden_fp).abs()
                 / (exp.hidden_fp.abs() + 1.0)).max().item() if NvS else 0.0
        ok_bwd = _report_ok(
            f'dispatch bwd    (hidden_fp_rel={rel_b:.3g})',
            rel_b <= _HIDDEN_FP_RTOL, self.rank, self.group, dev)
        return ok_fwd and ok_bwd

    def _verify_combine(self, hidden, weights):
        dev = f"npu:{self.local_rank}"
        # c_f: no expert math, so the K routes sum to K * hidden.
        self.combine_fwd()
        torch.npu.synchronize()
        golden_h = (hidden.float() * float(self.K)).to(torch.bfloat16)
        a_abs, a_rel, a_bad = _precision_stats(self.output_sh, golden_h)
        ok_h = _report_ok(
            f'combine hidden  (max_abs={a_abs:.4g} max_rel={a_rel:.4g} '
            f'bad={a_bad})', a_bad == 0, self.rank, self.group, dev)

        # c_b: route-weight gather must reproduce the original [S, K] weights.
        self.combine_bwd()
        torch.npu.synchronize()
        w_abs, w_rel, w_bad = _precision_stats(self.output_sk, weights)
        ok_w = _report_ok(
            f'combine weights (max_abs={w_abs:.4g} max_rel={w_rel:.4g} '
            f'bad={w_bad})', w_bad == 0, self.rank, self.group, dev)
        return ok_h and ok_w

    def _verify_prefetch(self):
        dev = f"npu:{self.local_rank}"
        SENT = -123.0
        # Fingerprint home rows (sum + abs-sum) and all_gather, so any rank can
        # validate a slot without moving full weights.
        fp = torch.zeros(3, self.epn, 2, dtype=torch.float32, device=dev)
        for c, chunk in enumerate(self._weight_chunks):
            home = chunk[:self.epn].float()
            fp[c, :, 0] = home.sum(dim=(1, 2))
            fp[c, :, 1] = home.abs().sum(dim=(1, 2))
        gathered = [torch.empty_like(fp) for _ in range(self.R)]
        dist.all_gather(gathered, fp, group=self.group)
        home_fp = torch.stack(gathered)  # [R, 3, epn, 2] indexed by owner rank

        for slot in self.prefetch_buffers:
            slot.fill_(SENT)
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)
        self._prefetch()
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

        plan_row = self.plan.experts_to_copy[self.rank].cpu()  # [B]
        ncell = float(self.H * self.hp)
        ok = True
        worst = 0.0
        for c in range(3):
            slots = self.prefetch_buffers[c].float()
            s_sum = slots.sum(dim=(1, 2))
            s_abs = slots.abs().sum(dim=(1, 2))
            for b in range(self.B):
                e = int(plan_row[b])
                if e < 0:
                    exp_sum, exp_abs = SENT * ncell, abs(SENT) * ncell
                else:
                    owner, le = e // self.epn, e % self.epn
                    exp_sum = home_fp[owner, c, le, 0].item()
                    exp_abs = home_fp[owner, c, le, 1].item()
                rel = max(abs(s_sum[b].item() - exp_sum) / (abs(exp_sum) + 1.0),
                          abs(s_abs[b].item() - exp_abs) / (exp_abs + 1.0))
                worst = max(worst, rel)
                if rel > 1e-4:
                    ok = False
        return _report_ok(f'prefetch slots  (worst rel fp diff={worst:.3g})',
                          ok, self.rank, self.group, dev)

    def _verify_grad_reduce(self):
        dev = f"npu:{self.local_rank}"
        H, hp, B = self.H, self.hp, self.B
        # Fill own chunks with known exact fp32 values.
        for le in range(self.epn):
            self.expert_grads[le].copy_(_grad_value_row(le, H, hp, dev))
        for b in range(B):
            self.reduce_buffers[b].copy_(_slot_value_row(self.rank, b, H, hp, dev))
        # Fence with HCCL, not the shmem barrier: the kernel uses the same
        # cann-shmem counter internally. npu.sync drains local fills; HCCL
        # then orders peers.
        torch.npu.synchronize()
        dist.barrier(group=self.group)

        plan = self.plan.experts_to_copy.contiguous()  # [R, B] int32, global ids
        self._trace(f'grad_reduce: LAUNCH (max_recv={self.max_recv} '
                    f'max_send={self.max_send})')
        launch_grad_reduce(self.expert_grads, self.reduce_buffers, plan,
                           rank=self.rank, num_sms=self.num_sms,
                           meta_buf=None, meta_stride=0, barrier_off=0,
                           grid_sync_bar=None)
        self._trace('grad_reduce: launched, syncing...')
        torch.npu.synchronize()
        self._trace('grad_reduce: sync OK')
        dist.barrier(group=self.group)

        plan_cpu = plan.cpu()
        # Expected: pristine row + all peer contributions targeting this rank.
        contrib = [[] for _ in range(self.epn)]
        for src in range(self.R):
            for b in range(B):
                e = int(plan_cpu[src, b])
                if e >= 0:
                    le = e - self.rank * self.epn
                    if 0 <= le < self.epn:
                        contrib[le].append((src, b))
        grads_ok, worst = True, 0.0
        for le in range(self.epn):
            exp = _grad_value_row(le, H, hp, dev)
            for (src, b) in contrib[le]:
                exp += _slot_value_row(src, b, H, hp, dev)
            ae = (self.expert_grads[le] - exp).abs().max().item()
            worst = max(worst, ae)
            if not torch.allclose(self.expert_grads[le], exp, rtol=0, atol=1e-3):
                grads_ok = False
        # Consumed slots zeroed; idle slots untouched.
        zero_ok, unchanged_ok = True, True
        for b in range(B):
            if int(plan_cpu[self.rank, b]) >= 0:
                if bool((self.reduce_buffers[b] != 0).any().item()):
                    zero_ok = False
            elif not torch.equal(self.reduce_buffers[b],
                                 _slot_value_row(self.rank, b, H, hp, dev)):
                unchanged_ok = False
        ok = grads_ok and zero_ok and unchanged_ok
        return _report_ok(
            f'grad_reduce     (grads={grads_ok} zero={zero_ok} '
            f'unchanged={unchanged_ok} max_abs={worst:.4g})',
            ok, self.rank, self.group, dev)

    def destroy(self):
        # Release this runner's VMM chunks via the inverse VMM ops.
        for chunk in self._weight_chunks + self._grad_chunks:
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
    ap.add_argument('--pf-overlap', action='store_true',
                    help='Overlap async prefetch (no rank barriers) with sync '
                         'dispatch; the standalone pf number is unchanged.')
    ap.add_argument('--verify', action='store_true',
                    help='Run correctness checks before timing (plan, dispatch, '
                         'combine, prefetch, grad_reduce) plus a latency sanity '
                         'check.')
    ap.add_argument('--verify-stages', default='plan,dispatch,grad_reduce,'
                    'prefetch,combine',
                    help='Comma list of checks to run: plan, dispatch, '
                         'grad_reduce, prefetch, combine '
                         '(default: %(default)s).')
    ap.add_argument('--verify-trace', action='store_true',
                    help='Print per-rank markers around each verify op.')
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
                          hp=args.hp, use_udma=use_udma,
                          pf_overlap=args.pf_overlap, verify=args.verify,
                          verify_stages=args.verify_stages.split(','),
                          verify_trace=args.verify_trace)
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

    if rank == 0 and args.pf_overlap:
        print('=== --pf-overlap: async prefetch + sync dispatch ===',
              flush=True)

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

            # Correctness self-check.
            if args.verify:
                runner.verify_precision(hidden, weights)

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
            # Report ingress (experts pulled) and egress (rows served).
            pf_bytes = 3 * H * runner.hp * 2
            pf_in_gbps = (runner.max_recv * pf_bytes / (pf * 1e-6) / 1e9) \
                if pf > 0 else 0.0
            pf_out_gbps = (runner.max_send * pf_bytes / (pf * 1e-6) / 1e9) \
                if pf > 0 else 0.0

            if args.verify and rank == 0:
                perf_ok = all(math.isfinite(x) and x > 0 for x in
                              (d_f, d_b, c_f, c_b, plan_us, pf))
                subset_ok = plan_us <= d_f * 1.05
                verdict = 'PASS' if (perf_ok and subset_ok) else 'WARN'
                print(f'[Verify] perf sanity: positive/finite={perf_ok} '
                      f'plan<=d_f={subset_ok} (plan {plan_us:.1f} us <= '
                      f'd_f {d_f:.1f} us) -> {verdict}', flush=True)

            rows.append(dict(lib='moonep', ep=R, E=E, H=H, K=K, sigma=sigma,
                             target_maxvio=target, maxvio=maxvio,
                             plan_us=plan_us, pf_us=pf,
                             max_recv=runner.max_recv, max_send=runner.max_send,
                             d_f_us=d_f, d_b_us=d_b, c_f_us=c_f, c_b_us=c_b,
                             pf_in_gbps=pf_in_gbps, pf_out_gbps=pf_out_gbps,
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
                      f'pf {pf:6.1f} us (in {pf_in_gbps:4.0f} / '
                      f'out {pf_out_gbps:4.0f} GB/s, '
                      f'recv={runner.max_recv} send={runner.max_send}, '
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
