"""MoonEP-Ascend vs torch.distributed all-to-all: dispatch / combine.

Reference: ``npu_patch/benchmarks/bench_vs.py`` (same CLI shape, same routing
generator, same event timing, same MaxVio sweep). The difference is the
opponent: instead of sweeping MaxVio against MoonEP itself, this bench puts
MoonEP's fused dispatch/combine next to a **plan-free, prefetch-free** MoE
dispatch/combine written directly on ``torch.distributed``'s
``all_to_all_single`` -- i.e. the baseline a framework writes when it has no
fused MoE communication library, only the plain collectives.

Both libraries run the SAME inputs and the SAME timing harness:
  - S=8192 tokens/rank, K topk, E experts, H hidden; EP group = whole world
    (R == world_size, so ``--nnodes N --nproc-per-node M`` gives R = N*M).
  - seed-1234 lognormal routing; rank 0 reverse-solves the lognormal sigma for
    each target MaxVio and broadcasts it (deterministic, same as bench_vs.py).
  - identical hidden states / route weights (fixed generator seeds).
  - eager NPU-event timing, warmup=20, iters=50, cross-rank mean.

The all-to-all baseline (``all2all``) -- no plan, no prefetch:

  dispatch.  Every (token, k) pair knows its expert; the expert's *home* rank is
  ``expert // epn``, so that is where the pair has to go. The pairs are permuted
  by (destination rank, destination-local expert) (one ``index_select``), moved
  with ``dist.all_to_all_single`` using uneven split sizes (each rank sends
  exactly what its routing asks for, nothing is padded to a fixed capacity) and
  the received tokens are then sorted by their local expert index so the expert
  GEMM sees the same expert-grouped layout MoonEP's dispatch epilogue produces.

  No per-entry expert id is carried on the data plane. The sender knows both the
  destination rank and the home-local expert of every pair, so it groups its
  outgoing buffer at the source; the receiver rebuilds the *same* expert grouping
  from one exchanged per-(source, destination-expert) count table (an untimed
  ``prepare``-time ``all_gather``). That is the control-plane / data-plane split
  MoonEP gets from its plan: there ``dst`` comes out of the plan and only hidden +
  route weights go on the wire (``moonep_dispatch_kernel`` takes ``dst_gm`` and
  ``route_weights_gm``) -- see the metadata note further down.

  ``d_f`` therefore moves hidden + route weights, ``d_b`` moves hidden only (the
  backward dispatch does not have to re-send the weights -- combine gathers them
  back, exactly like MoonEP's ``c_b``); both match MoonEP's payload byte for byte.

  combine.  The exact inverse: unsort back to source-rank order, all-to-all the
  expert-side activations back, unpermute to (token, k) order and reduce over the
  topk. ``c_f`` reduces unweighted (hidden only), ``c_b`` gathers the route
  weights back and reduces with them.

  The baseline pays no planning cost and owns no symmetric/shared memory, but it
  also cannot fuse the permute into the transfer (it materializes ~S*K*H*2 bytes
  of permuted activations per rank) and it owns no expert weights, so ``prefetch``
  is meaningless for it -- ``pf`` is 0 for all2all, while the MoonEP runner does
  prefetch and reports it (bench_vs.py's convention).

  Metadata note (why the baseline still needs *some* metadata, and why none of it
  has to be per-entry): the receiver cannot infer which expert an arrived
  activation belongs to -- routing is private to the sender -- so *something* has
  to tell it. But "something" need not ride next to the token: since the baseline
  is free to choose its own layout (it does not have to reproduce MoonEP's), the
  sender pre-groups by (destination, local expert) and one count table per source
  (``R * epn`` ints, exchanged once with the already-required split counts) is
  enough to rebuild the expert grouping. The only payload the receiver truly has
  to be handed is the route weight, because the arithmetic needs it: it is not
  metadata about placement, it is data. ``d_f`` sends it, ``c_b`` gathers it back,
  ``d_b``/``c_f`` do not -- exactly the same payload set as MoonEP.
  Caveat: the count exchange is untimed (``prepare``), so the baseline's ``plan``
  column is 0 while MoonEP's planning is a timed kernel; MoonEP's plan buys
  token migration and a fused epilogue, the baseline's counts buy nothing but the
  grouping (and cannot migrate, see the methodology note below).

  Methodology note: MoonEP's planner is free to *migrate* a token away from its
  expert's home rank when that home rank is overloaded (that is what keeps the
  expert load balanced as MaxVio grows -- see test_npu/planing_reference.py). The
  baseline always sends to the home rank, so at high MaxVio the two libraries do
  not move exactly the same bytes: MoonEP buys balance with extra cross-rank
  traffic, the baseline keeps the minimal path but lets the hot rank become the
  straggler. Both are timed with the same logical-payload bandwidth formula, so
  read the high-MaxVio points accordingly.

Reported ops (MoonEP = the same low-level ``launch_*`` calls on the Buffer ctx
that bench_vs.py times, prefetch included exactly as bench_vs.py includes it; the
all2all baseline owns no expert weights, so its ``pf`` is 0 by construction):
  - plan: MoonEP ``launch_planning`` (exact, separate) | all2all 0 (no planning
    kernel; the baseline's grouping counts are exchanged once in ``prepare``)
  - d_f:  planning + dispatch (dedup build) + prefetch | permute + all_to_all
          (hidden + route weights) + expert sort
  - d_b:  dispatch with the saved plan (no planning) + prefetch | same, hidden-only
  - c_f:  combine, hidden only (unweighted sum over topk)
  - c_b:  combine with the route-weights gather (weighted sum over topk)
  - pf:   ``launch_prefetch`` timed separately (its share of d_f/d_b) | 0
Bandwidth = S*K*H*2 / t (logical payload bytes, same formula for both libs); ``pf``
is reported separately and is NOT part of the payload bandwidth.
``d_f``/``d_b``/``c_f``/``c_b`` are the ops on the MoE critical path; the epilogue
arithmetic of the baseline is bf16 (the op under test is the communication).

Note: the baseline relies on ``all_to_all_single`` with *uneven* split sizes
(AlltoAllV) -- that is the point of "send every token to the rank it belongs to".
If the CANN/HCCL build behind torch_npu does not implement uneven splits, the
first all-to-all raises; run ``--libs all2all`` alone to check the collective in
isolation.

EP group = whole world (R == world_size). Single- and multi-node both work
(R = nnodes x nproc):

    single node (default; UID bootstrap, no ip_port needed; 8 cards):
        torchrun --nproc_per_node=8 npu_patch/benchmarks/bench_vs_all2all.py \
            [--out r.csv] [--plot r.png]
    multi node (ip_port/DEFAULT bootstrap; when ip_port is omitted it is
    derived from MASTER_ADDR/MASTER_PORT, port + 1):
        # node 0
        MASTER_ADDR=192.168.1.10 NODES=2 NODE_RANK=0 \
            bash npu_patch/benchmarks/run_bench.sh bench_vs_all2all.py
        # node 1
        MASTER_ADDR=192.168.1.10 NODES=2 NODE_RANK=1 \
            bash npu_patch/benchmarks/run_bench.sh bench_vs_all2all.py
    all-to-all baseline only (loads no MoonEP Buffer / shmem heap; verify the
    collective in isolation without needing Ascend shmem/VMM):
        torchrun --nproc_per_node=8 npu_patch/benchmarks/bench_vs_all2all.py \
            --libs all2all
    explicit shmem ip_port:
        torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
            --master-addr=192.168.1.10 --master-port=29500 \
            npu_patch/benchmarks/bench_vs_all2all.py --ip-port tcp://192.168.1.10:29501
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
    buffer-backed init. Identical to bench_vs.py (same test_prefetch.py
    pattern): rows ``[0:epn)`` are this rank's home experts and
    ``[epn:epn+B)`` are the prefetch slots. ``device_id`` is the node-local
    logical device id (``local_rank``: the 0..N-1 visible index from
    ASCEND_RT_VISIBLE_DEVICES), not the global rank (>= the local card count
    in multi-node) or the physical card id.
    """
    epn = (E + R - 1) // R
    return create_vmm_physical([epn + B, H, Hp], torch.bfloat16,
                               device_id)[0]


# ---------------------------------------------------------------------------
# Shared timing harness -- eager npu-event timing, cross-rank mean us
# ---------------------------------------------------------------------------

def time_op(fn, group, rank, warmup=20, iters=50, cudagraph=False):
    return time_kernel(fn, group, rank, warmup=warmup, iters=iters,
                       agg="mean", cudagraph=cudagraph)["us"]


def gbps(S, K, H, us):
    return S * K * H * 2 / (us * 1e-6) / 1e9


def _maxvio_for_sigma(sigma, S, K, E, R, dev):
    """Realized MaxVio of the lognormal routing at the given sigma, using the
    benchmark's fixed seeds (1234 shared, per-rank local) -- deterministic."""
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
# MoonEP -- the same low-level launchers bench_vs.py times, prefetch included
# ---------------------------------------------------------------------------

class MoonEPRunner:
    NAME = 'moonep'

    def __init__(self, group, R, S, K, E, H, num_sms, local_rank,
                 use_udma=False, hp=3072):
        self.group = group
        self.num_sms = num_sms
        self.rank = dist.get_rank(group)
        self.R = R
        self.local_rank = local_rank
        self.use_udma = use_udma
        self.hp = hp
        self.buffer = Buffer(S, H, K, E, R, num_sms=num_sms, group=group,
                             use_udma=use_udma, explicitly_destroy=True)
        self.ctx = self.buffer._require_ctx()
        # Pre-allocate the plan + outputs once (test_planning.py pattern): the
        # timed ops reuse these instead of allocating per call.
        self.plan, self.cu_seqlens = allocate_planning_outputs(self.ctx)
        dev = f"npu:{local_rank}"
        self.output_sh = torch.empty(S, H, dtype=torch.bfloat16, device=dev)
        self.output_sk = torch.empty(S, K, dtype=torch.float32, device=dev)

        # Prefetch (same as bench_vs.py): composite [E+B] weight layout expects
        # B == epn (native precondition). Per-rank [epn+B, H, hp] VMM chunks are
        # registered with the buffer-backed init; the logical [E, H, hp] home
        # views and the local [B, H, hp] slots are translated after
        # init_with_buffer via _prepare_shmem_views.
        self.epn = E // R
        self.B = int(self.ctx['B'])
        assert self.B == self.epn, (
            f"composite [E+B] layout expects B == epn, got B={self.B}, "
            f"epn={self.epn}")
        self._weight_chunks = []
        for _ in range(3):
            self._weight_chunks.append(
                make_symmetric_experts(local_rank, R, E, H, hp, self.B))
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

    def prepare_shmem_views(self):
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
        # structures used by the saved-plan dispatch/combine below.
        launch_planning(self.ctx, self._topk_flat, tpe, self.cu_seqlens,
                        self.plan)
        launch_dispatch(self.ctx, hidden, weights, self.plan,
                        build_dedup_map=True)
        # Byte accounting for the prefetch reporting (plan slot table, global
        # expert ids) -- same as bench_vs.py.
        etc_cpu = self.plan.experts_to_copy.cpu()   # [R, B] int32, -1 = idle
        self.max_recv = int((etc_cpu >= 0).sum(dim=1).max().item())
        torch.npu.synchronize()
        ShmemRuntime.barrier(group=self.group)

    def planning(self):
        launch_planning(self.ctx, self._topk_flat, self.tpe,
                        self.cu_seqlens, self.plan)

    def _prefetch(self):
        # Low-level launch_prefetch per weight tensor (gate/up/down), pulling
        # this rank's plan row into the local [B, H, hp] slots -- the same op
        # bench_vs.py times in d_f/d_b.
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
        # full fwd: planning + dispatch + epilogue, then prefetch (the forward
        # expert GEMMs consume the prefetched weights).
        launch_planning(self.ctx, self._topk_flat, self.tpe,
                        self.cu_seqlens, self.plan)
        launch_dispatch(self.ctx, self.hidden, self.weights, self.plan,
                        build_dedup_map=True)
        self._prefetch()

    def dispatch_bwd(self):
        # bwd: saved plan (no planning, no dedup builder), hidden only. The
        # backward expert GEMMs need the same weights, so prefetch again.
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
        # Release this runner's 3 weight VMM chunks via inverse VMM ops, then
        # the Buffer -- same teardown order as bench_vs.py.
        for chunk in self._weight_chunks:
            alloc = getattr(chunk, '_vmm_allocation', None)
            if alloc is not None:
                alloc.destroy()
        self.buffer.destroy()


# ---------------------------------------------------------------------------
# all2all -- dispatch/combine on torch.distributed all_to_all_single
# ---------------------------------------------------------------------------

class All2AllRunner:
    """Plan-free, prefetch-free MoE dispatch/combine over ``all_to_all_single``.

    Layout convention (same flattening as MoonEP): the routed entries are the
    ``S*K`` pairs ``p = t*K + k`` in row-major ``[S, K]`` order; pair ``p``'s
    expert is ``topk[t, k]`` and its home rank is ``expert // epn``, where
    ``epn = E // R`` is the number of experts each rank owns.

    Buffers are per-rank exact (not symmetric): every rank sends exactly ``S*K``
    routed entries and receives however many its routing asks for, so nothing is
    padded and the transfer size is data dependent -- which is what
    ``output_split_sizes`` / ``input_split_sizes`` are for.

    Metadata: the outgoing buffer is pre-grouped by (destination rank, destination
    -local expert), so the only placement information the receiver needs is the
    per-(source, expert) segment count table -- exchanged once, untimed, in
    ``prepare``. No expert id rides next to the activations (see the module
    docstring); the route weight does, because the combine arithmetic needs it.
    """

    NAME = 'all2all'

    def __init__(self, group, R, S, K, E, H, num_sms, local_rank):
        self.group = group
        self.rank = dist.get_rank(group)
        self.R = int(R)
        self.S, self.K, self.E, self.H = int(S), int(K), int(E), int(H)
        self.num_sms = num_sms  # torch-level baseline: no SM budget to set
        self.dev = f"npu:{local_rank}"
        self.epn = self.E // self.R
        self.SK = self.S * self.K
        pairs = torch.arange(self.SK, dtype=torch.int64, device=self.dev)
        self.pair_tok = pairs // self.K          # pair -> token
        self.n_recv = 0
        self.max_recv = 0

    # -- routing-dependent setup (untimed) ---------------------------------
    def prepare(self, topk, tpe, hidden, weights):
        self.hidden = hidden
        self.weights_flat = weights.reshape(-1).contiguous()      # [SK] fp32
        expert = topk.reshape(-1).to(torch.int64)                 # [SK] global id

        # Chunk order: pairs sorted by (destination rank, destination-local
        # expert) -- i.e. by global expert id, which is exactly what the sender
        # already knows (``expert = dest * epn + local_e``). That grouping is
        # what lets the receiver reconstruct the expert layout from counts alone.
        # ``perm[m]`` is the pair sitting at position m of this rank's outgoing
        # buffer, so positions 0..send_counts[0]-1 go to rank 0, and so on.
        perm = torch.argsort(expert)
        self.perm = perm
        self.inv_perm = torch.argsort(perm)                  # pair -> position
        self.src_tok = self.pair_tok.index_select(0, perm)   # position -> token

        # Route weights ride the same permuted order (column vector, fp32).
        self.w_send = torch.index_select(
            self.weights_flat, 0, perm).unsqueeze(1).contiguous()   # [SK, 1]

        # Control-plane table: how many of this rank's pairs go to each
        # (destination rank, destination-local expert). One entry per expert slot
        # -- never per pair. Sharing it is what removes the per-entry expert id
        # from the data plane: ``seg[d, e]`` is all the receiver needs to slice
        # each source's block into its per-expert segments.
        seg = torch.bincount(expert, minlength=self.R * self.epn).view(
            self.R, self.epn)                                # [dest, expert]
        send_counts = seg.sum(dim=1)
        gathered = [torch.empty_like(seg) for _ in range(self.R)]
        dist.all_gather(gathered, seg.contiguous(), group=self.group)
        all_seg = torch.stack(gathered)                      # [source, dest, exp]
        # Blocks the world sends to this rank, per source: [source, expert].
        recv_seg = all_seg[:, self.rank, :].contiguous()
        recv_counts = recv_seg.sum(dim=1)
        self.send_counts = send_counts
        self.recv_counts = recv_counts
        self.send_counts_list = [int(v) for v in send_counts.cpu().tolist()]
        self.recv_counts_list = [int(v) for v in recv_counts.cpu().tolist()]

        self.n_recv = int(recv_counts.sum().item())
        # Totals per rank: column q of all_counts sums what every source sends to
        # rank q (a rank's *send* total is its row -- different number).
        all_counts = all_seg.sum(dim=2)                      # [source, dest]
        self.max_recv = int(all_counts.sum(dim=0).max().item())
        assert sum(self.send_counts_list) == self.SK, (
            f"rank {self.rank}: send counts {self.send_counts_list} do not sum "
            f"to S*K={self.SK}")
        assert int(recv_seg.sum().item()) == self.n_recv
        # Token conservation: every rank sends S*K entries, so the world receives
        # exactly S*K*R of them (catches split-size / rank-map bugs).
        total = torch.tensor([self.n_recv], dtype=torch.int64, device=self.dev)
        dist.all_reduce(total, group=self.group)
        assert int(total.item()) == self.SK * self.R, (
            f"rank {self.rank}: all-to-all token conservation broken "
            f"(world recv {int(total.item())} != {self.SK * self.R})")

        H = self.H
        self.w_recv = torch.empty(self.n_recv, 1, dtype=torch.float32,
                                  device=self.dev)
        self.x_recv = torch.empty(self.n_recv, H, dtype=torch.bfloat16,
                                  device=self.dev)
        self.x_sorted = torch.empty(self.n_recv, H, dtype=torch.bfloat16,
                                    device=self.dev)
        self.w_sorted = torch.empty(self.n_recv, 1, dtype=torch.float32,
                                    device=self.dev)
        self.w_back = torch.empty(self.SK, 1, dtype=torch.float32,
                                  device=self.dev)
        self.y_back = torch.empty(self.SK, H, dtype=torch.bfloat16,
                                  device=self.dev)
        self.out_sh = torch.empty(self.S, H, dtype=torch.bfloat16,
                                  device=self.dev)
        self.order = torch.arange(self.n_recv, dtype=torch.int64,
                                  device=self.dev)

        # Expert label of every arrival position in ``x_recv``'s layout: blocks
        # arrive source-major, and inside a block expert-major (the sender
        # pre-grouped), so repeating ``e = 0..epn-1`` by the source's per-expert
        # counts lines the labels up with the buffer. Local, derived from the
        # exchanged counts -- nothing came off the wire for it.
        self.expert_of_pos = torch.repeat_interleave(
            torch.arange(self.epn, dtype=torch.int64,
                         device=self.dev).repeat(self.R),
            recv_seg.reshape(-1))

        # One untimed dispatch so the combine paths start from real data.
        self._dispatch(with_weights=True)
        torch.npu.synchronize()
        dist.barrier(group=self.group)

    # -- dispatch ----------------------------------------------------------
    def dispatch_fwd(self):
        self._dispatch(with_weights=True)

    def dispatch_bwd(self):
        self._dispatch(with_weights=False)

    def _dispatch(self, with_weights):
        # permute: pair order -> (destination, home-local expert) order.
        # index_select materializes the permuted activations, which MoonEP
        # instead fuses into its comm epilogue.
        x = torch.index_select(self.hidden, 0, self.src_tok)
        dist.all_to_all_single(
            self.x_recv, x,
            output_split_sizes=self.recv_counts_list,
            input_split_sizes=self.send_counts_list, group=self.group)
        if with_weights:
            # d_f carries hidden + route weights; d_b carries hidden only (the
            # backward dispatch does not re-send the weights -- c_b gathers them
            # back), matching MoonEP's payload exactly.
            dist.all_to_all_single(
                self.w_recv, self.w_send,
                output_split_sizes=self.recv_counts_list,
                input_split_sizes=self.send_counts_list, group=self.group)
        # expert sort: group the received tokens by their local expert index so
        # the consumer sees MoonEP's expert-grouped dispatch layout.
        order = torch.argsort(self.expert_of_pos)
        self.order = order
        self.x_sorted.copy_(torch.index_select(self.x_recv, 0, order))
        if with_weights:
            self.w_sorted.copy_(torch.index_select(self.w_recv, 0, order))

    # -- combine -----------------------------------------------------------
    def combine_fwd(self):
        self._combine(with_weights=False)

    def combine_bwd(self):
        self._combine(with_weights=True)

    def _combine(self, with_weights):
        # unsort: expert order -> the source-rank order the all-to-all expects
        # (``inv`` inverts the dispatch's expert sort).
        inv = torch.argsort(self.order)
        x = torch.index_select(self.x_sorted, 0, inv)
        dist.all_to_all_single(
            self.y_back, x,
            output_split_sizes=self.send_counts_list,
            input_split_sizes=self.recv_counts_list, group=self.group)
        if with_weights:
            # The MoE backward needs the routed weights back at the origin rank;
            # MoonEP's combine gathers them too.
            w = torch.index_select(self.w_sorted, 0, inv)
            dist.all_to_all_single(
                self.w_back, w,
                output_split_sizes=self.send_counts_list,
                input_split_sizes=self.recv_counts_list, group=self.group)
        # unpermute back to (token, k) order and reduce over the topk.
        y3 = torch.index_select(self.y_back, 0, self.inv_perm).view(
            self.S, self.K, self.H)
        if with_weights:
            w3 = torch.index_select(
                self.w_back.view(-1), 0, self.inv_perm).to(torch.bfloat16)
            self.out_sh.copy_(torch.sum(y3 * w3.view(self.S, self.K, 1), dim=1))
        else:
            self.out_sh.copy_(torch.sum(y3, dim=1))

    def destroy(self):
        # No symmetric/shared memory and no plan: the buffers are ordinary NPU
        # allocations and go away with the runner.
        self.hidden = None
        self.weights_flat = None


# ---------------------------------------------------------------------------
# CUDA-graph capturability probe (opt-in: --cudagraph)
# ---------------------------------------------------------------------------

def all_to_all_capturable(group, dev, rank, R):
    """True when a c10d ``all_to_all_single`` can be captured into an NPU graph.

    ``time_kernel`` falls back to the eager loop per op when capture fails, so
    without this probe a run could time MoonEP from a graph and the all2all
    baseline eagerly -- two different launch regimes. When the probe fails the
    whole benchmark stays eager.
    """
    if not (hasattr(torch.npu, 'CUDAGraph') and hasattr(torch.npu, 'graph')):
        return False
    ok = True
    try:
        buf = torch.zeros(R, dtype=torch.int64, device=dev)
        graph = torch.npu.CUDAGraph()
        with torch.npu.graph(graph):
            dist.all_to_all_single(buf, buf, output_split_sizes=[1] * R,
                                   input_split_sizes=[1] * R, group=group)
        torch.npu.synchronize()
    except Exception as exc:  # any capture failure means "eager"
        ok = False
        if rank == 0:
            print(f'[bench] dist.all_to_all_single is not NPU-graph '
                  f'capturable ({exc}); timing every library eagerly.',
                  flush=True)
    # Re-synchronize the group either way: a partially captured collective must
    # not leave ranks out of step for the rest of the run.
    torch.npu.synchronize()
    dist.barrier(group=group)
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

RUNNERS = {
    'moonep': MoonEPRunner,
    'all2all': All2AllRunner,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_engine_args(ap)
    ap.add_argument('--libs', nargs='+', default=['moonep', 'all2all'],
                    choices=list(RUNNERS),
                    help='Libraries to benchmark (default: both). "all2all" '
                         'alone needs neither the MoonEP Buffer nor the Ascend '
                         'shmem stack.')
    ap.add_argument('--out', default='bench_vs_all2all_results.csv',
                    help='CSV output path (rank 0). Default: %(default)s')
    ap.add_argument('--plot', default=None,
                    help='If set, save the grouped-bar figure to this path '
                         '(rank 0 only).')
    ap.add_argument('--maxvios', default='0.2,1,10,20',
                    help='Comma-separated target MaxVio values; the lognormal '
                         'sigma is reverse-solved for each (default: %(default)s)')
    ap.add_argument('--experts', type=int, default=1024)
    ap.add_argument('--hidden', type=int, default=3584)
    ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--hp', type=int, default=3072,
                    help="Expert weight inner dim H' (MoE intermediate size) for "
                         'the MoonEP prefetch op (default: %(default)s); the '
                         'all2all baseline has no weights to prefetch.')
    ap.add_argument('--num-sms', type=int, default=32,
                    help='MoonEP op SM budget (the all2all baseline has none).')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--ip-port', default=None,
                    help='shmem DEFAULT-bootstrap ip_port URL for the '
                         'buffer-backed shmem init ("tcp://<rank0_ip>:<port>" '
                         'or "tcp6://[ipv6]:<port>"). Required for multi-node; '
                         'when omitted in a multi-node launch it is derived '
                         'from MASTER_ADDR/MASTER_PORT (port + 1). Single-node '
                         'launches keep the UID bootstrap and ignore this.')
    ap.add_argument('--cudagraph', action='store_true',
                    help='Capture the timed loops into an NPU graph instead of '
                         'the eager loop. Off by default: the all2all baseline '
                         'is a c10d collective sequence and the reference '
                         '(bench_vs_deepep.py) times both libraries eagerly. '
                         'The flag is dropped (with a note) when the probe '
                         'shows the collective is not capturable, so the two '
                         'libraries are never timed in different regimes.')
    ap.set_defaults(cudagraph=False)
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
    targets = [float(x) for x in args.maxvios.split(',')]

    # Multi-node detection: under torchrun, WORLD_SIZE > LOCAL_WORLD_SIZE
    # (procs per node) means multi-node.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    multi_node = world_size > local_world_size

    # Runners: MoonEP constructs its Buffer first (it registers the VMM
    # descriptors the shmem init consumes); the all2all baseline is pure
    # torch.distributed and needs no symmetric memory at all.
    runners = {}
    if 'moonep' in args.libs:
        runners['moonep'] = MoonEPRunner(group, R, S, K, E, H, args.num_sms,
                                         local_rank, use_udma=use_udma,
                                         hp=args.hp)
    if 'all2all' in args.libs:
        runners['all2all'] = All2AllRunner(group, R, S, K, E, H,
                                           args.num_sms, local_rank)

    if 'moonep' in args.libs:
        if not ShmemRuntime.is_initialized():
            if args.ip_port is not None or multi_node:
                # Multi-node: ip_port/DEFAULT bootstrap; when ip_port is omitted
                # it is derived from MASTER_ADDR/MASTER_PORT (port + 1, see
                # ShmemRuntime.init_with_ip_port).
                ShmemRuntime.init_with_ip_port(group=group,
                                               ip_port=args.ip_port,
                                               enable_udma=use_udma)
            else:
                ShmemRuntime.init_with_buffer(group=group,
                                              enable_udma=use_udma)
        runners['moonep'].prepare_shmem_views()

    cudagraph = args.cudagraph
    if cudagraph and 'all2all' in args.libs:
        cudagraph = all_to_all_capturable(group, dev, rank, R)
        if not cudagraph and rank == 0:
            print('[bench] --cudagraph disabled for every library (mixed '
                  'timing regimes are not comparable).', flush=True)

    if rank == 0:
        print('bench_vs_all2all: MoonEP-Ascend vs torch.distributed '
              'all_to_all_single', flush=True)
        print(f'  ep={R} (world), S={S}/rank, E={E}, H={H}, K={K}, '
              f"num_sms={args.num_sms}, hp={args.hp}, multi_node={multi_node}, "
              f'libs={",".join(args.libs)}, cudagraph={cudagraph}', flush=True)
        print('  timed: d_f dispatch fwd | d_b dispatch bwd | c_f combine fwd | '
              'c_b combine bwd | pf prefetch (moonep only; the all2all baseline '
              'owns no expert weights)', flush=True)

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

            timings = {}
            for lib in args.libs:
                runner = runners[lib]
                runner.prepare(topk, tpe, hidden, weights)
                d_f = time_op(runner.dispatch_fwd, group, rank, args.warmup,
                              args.iters, cudagraph=cudagraph)
                d_b = time_op(runner.dispatch_bwd, group, rank, args.warmup,
                              args.iters, cudagraph=cudagraph)
                c_f = time_op(runner.combine_fwd, group, rank, args.warmup,
                              args.iters, cudagraph=cudagraph)
                c_b = time_op(runner.combine_bwd, group, rank, args.warmup,
                              args.iters, cudagraph=cudagraph)
                if hasattr(runner, 'planning'):
                    # MoonEP: planning is a real kernel and is measured exactly
                    # and separately (it is part of d_f, not of d_b).
                    plan_us = time_op(runner.planning, group, rank, args.warmup,
                                      args.iters, cudagraph=cudagraph)
                else:
                    # all2all: no planning kernel at all -- the destination rank
                    # comes straight from expert // epn, and the grouping counts
                    # are exchanged once (untimed) in prepare().
                    plan_us = 0.0
                if hasattr(runner, 'prefetch'):
                    # MoonEP: prefetch is included in d_f/d_b (bench_vs.py's
                    # convention) and also measured on its own for its share.
                    pf_us = time_op(runner.prefetch, group, rank, args.warmup,
                                    args.iters, cudagraph=cudagraph)
                else:
                    # all2all: no expert weights exist, so there is nothing to
                    # prefetch -- 0 by construction, not a hidden win.
                    pf_us = 0.0
                timings[lib] = dict(plan_us=plan_us, pf_us=pf_us, d_f=d_f,
                                    d_b=d_b, c_f=c_f, c_b=c_b)
                rows.append(dict(lib=lib, ep=R, E=E, H=H, K=K, sigma=sigma,
                                 target_maxvio=target, maxvio=maxvio,
                                 plan_us=plan_us, pf_us=pf_us,
                                 d_f_us=d_f, d_b_us=d_b, c_f_us=c_f, c_b_us=c_b,
                                 d_f_gbps=gbps(S, K, H, d_f),
                                 d_b_gbps=gbps(S, K, H, d_b),
                                 c_f_gbps=gbps(S, K, H, c_f),
                                 c_b_gbps=gbps(S, K, H, c_b)))
                if rank == 0:
                    extra = ''
                    if lib == 'all2all':
                        extra = (f' | recv {runner.n_recv} entries '
                                 f'({runner.n_recv / (S * K):.2f}x sent, '
                                 f'max rank {runner.max_recv / (S * K):.2f}x)')
                    elif pf_us > 0:
                        pf_gbps = (3 * runner.max_recv * H * args.hp * 2
                                   / (pf_us * 1e-6) / 1e9)
                        extra = (f' | pf {pf_us:6.1f} us ({pf_gbps:4.0f} GB/s)')
                    print(f'{lib:>8} ep={R} E={E} H={H} K={K} '
                          f'sigma={sigma:.5f}: maxvio={maxvio:6.2f} | '
                          f'plan {plan_us:6.1f} us ({plan_us / d_f * 100:4.1f}%)'
                          f' | d_f {d_f:7.1f} us ({gbps(S, K, H, d_f):5.0f} '
                          f'GB/s) | d_b {d_b:7.1f} us ({gbps(S, K, H, d_b):5.0f}'
                          f' GB/s) | c_f {c_f:7.1f} us ({gbps(S, K, H, c_f):5.0f}'
                          f' GB/s) | c_b {c_b:7.1f} us ({gbps(S, K, H, c_b):5.0f}'
                          f' GB/s){extra}', flush=True)
            if rank == 0 and 'moonep' in timings and 'all2all' in timings:
                m, a = timings['moonep'], timings['all2all']
                ratios = ' '.join(f'{op} x{a[op] / m[op]:.2f}'
                                  for op in ('d_f', 'd_b', 'c_f', 'c_b'))
                print(f'         all2all / moonep latency: {ratios}', flush=True)
            del hidden, weights
    finally:
        # Buffer-backed teardown order (same as bench_vs.py): finalize (drop
        # symmetric mappings), inverse VMM ops, then clear the descriptor
        # registry. Kept in finally so a failing target still frees
        # shmem/VMM resources.
        if 'moonep' in runners:
            if ShmemRuntime.is_initialized():
                ShmemRuntime.finalize()
            runners['moonep'].destroy()
            reset_symmetric_descriptors()

    if rank == 0 and args.out and rows:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.out}')
    if rank == 0 and args.plot and rows:
        plot_rows(rows, args.plot)
        print(f'wrote {args.plot}')

    dist.barrier(group=group)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Plotting (inline): grouped bars, MoonEP (stacked segments) vs all2all
# ---------------------------------------------------------------------------

_COLORS = {'comm': '#2563eb', 'planning': '#22c55e', 'prefetch': '#a855f7',
           'all2all': '#f59e0b'}


def plot_rows(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    libs = [lib for lib in ('moonep', 'all2all')
            if any(r['lib'] == lib for r in rows)]
    xs = sorted({r['maxvio'] for r in rows})
    xpos = np.arange(len(xs))
    labels = [f'{x:g}' for x in xs]
    C = _COLORS
    color = {'moonep': C['comm'], 'all2all': C['all2all']}
    width = 0.8 / len(libs)

    def get(lib, op, x):
        return next(r[f'{op}_us'] for r in rows
                    if r['lib'] == lib and r['maxvio'] == x)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)

    def offset(i):
        return (i - (len(libs) - 1) / 2) * width

    def stacked(ax, comm_fn, segs, title, total_fn):
        for i, lib in enumerate(libs):
            comm = [comm_fn(lib, x) for x in xs]
            ax.bar(xpos + offset(i), comm, width, color=color[lib], label=lib)
            bottoms = comm[:]
            for sname, fn in segs:
                vals = [fn(lib, x) for x in xs]
                ax.bar(xpos + offset(i), vals, width, bottom=bottoms,
                       color=C[sname])
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            for x0, t in zip(xpos + offset(i), [total_fn(lib, x) for x in xs]):
                ax.text(x0, t + 12, f'{t:.0f}', ha='center', va='bottom',
                        fontsize=7.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel('latency (us)')

    stacked(axes[0, 0],
            lambda lib, x: (get(lib, 'd_f', x) - get(lib, 'plan', x)
                            - get(lib, 'pf', x)),
            [('planning', lambda lib, x: get(lib, 'plan', x)),
             ('prefetch', lambda lib, x: get(lib, 'pf', x))],
            'dispatch fwd', lambda lib, x: get(lib, 'd_f', x))
    stacked(axes[0, 1],
            lambda lib, x: get(lib, 'd_b', x) - get(lib, 'pf', x),
            [('prefetch', lambda lib, x: get(lib, 'pf', x))],
            'dispatch bwd', lambda lib, x: get(lib, 'd_b', x))

    for ax, op, title in ((axes[1, 0], 'c_f', 'combine fwd'),
                          (axes[1, 1], 'c_b', 'combine bwd')):
        for i, lib in enumerate(libs):
            ys = [get(lib, op, x) for x in xs]
            ax.bar(xpos + offset(i), ys, width, color=color[lib], label=lib)
            for x0, y in zip(xpos + offset(i), ys):
                ax.text(x0, y, f'{y:.0f}', ha='center', va='bottom',
                        fontsize=7.5)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylabel('latency (us)')

    for ax in axes[-1]:
        ax.set_xlabel('MaxVio (0 = balanced, k = hottest expert at (k+1)x load)')
        ax.legend(frameon=False, fontsize=9)
    E, H, K = rows[0]['E'], rows[0]['H'], rows[0]['K']
    ep = rows[0]['ep']
    fig.suptitle(f'MoonEP-Ascend vs torch.distributed all_to_all '
                 f'\u2014 ep={ep}, E={E}, H={H}, K={K}, S=8192/rank',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == '__main__':
    main()
