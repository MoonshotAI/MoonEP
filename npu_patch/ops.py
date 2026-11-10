"""
KernelOp subclasses for ``npu_patch``.

These mirror the ``ascend_deepep`` op-object pattern (op constructed at Buffer
init, ``compute`` invoked on demand) while exposing MoonEP's contracts. Each op
launches its Ascend implementation through ``launchers``. The former
dispatch-epilogue / combine-prologue steps are folded into ``DispatchOp`` and
``CombineOp`` respectively, so no dedicated ops remain for them.
"""

import torch

from .kernel_op import KernelOp
from .launchers import (
    launch_combine_op,
    launch_dispatch_op,
    launch_dispatch_profile_op,
    launch_grad_reduce,
    launch_planning_op,
    launch_prefetch,
)


def _collect_inputs(op_name, tensors=(), scalars=()):
    """Validate and summarize a kernel-launch input set.

    ``tensors``: iterable of ``(name, tensor, expected_shape_or_None, expected_dtype)``;
    a ``None`` tensor is reported as ``name=None`` (optional input). ``scalars``:
    iterable of ``(name, value)``. Raises ``ValueError`` on a shape/dtype mismatch
    so the input plumbing is validated before the AscendC kernel exists; returns a
    human-readable summary that the stub embeds in the ``NotImplementedError`` it
    raises.
    """
    parts = []
    for name, t, shape, dtype in tensors:
        if t is None:
            parts.append(f"{name}=None")
            continue
        actual = tuple(t.shape)
        if shape is not None and actual != tuple(shape):
            raise ValueError(
                f"{op_name}: {name} expected shape {list(shape)}, got {list(actual)}")
        if t.dtype != dtype:
            raise ValueError(
                f"{op_name}: {name} expected dtype {dtype}, got {t.dtype}")
        parts.append(f"{name}={list(actual)}:{t.dtype}")
    for name, value in scalars:
        parts.append(f"{name}={value}")
    return ", ".join(parts)


class PlanningOp(KernelOp):
    """Build the same communication plan independently on every rank.

    ``tokens_per_expert`` is retained for MoonEP API compatibility. The kernel
    computes local expert counts from ``topk_flat`` and all-gathers them before
    balancing, so the compatibility tensor is validated but not consumed.
    ``zero_fill_ranges`` and ``remote_stats`` are likewise retained in the plan
    object but are not written by this implementation.
    """

    def compute(self, ctx, topk_flat, tokens_per_expert, cu_seqlens, plan,
                shmem_instance_id: int = 0):
        rank = int(ctx["rank"])
        R = int(ctx["R"])
        S = int(ctx["S"])
        K = int(ctx["K"])
        E = int(ctx["E"])
        B = int(ctx["B"])
        N = S * K
        NvS = int(ctx["NvS"])
        _collect_inputs(
            "PlanningOp",
            tensors=(
                ("topk_flat(topk_idx)", topk_flat, [N], torch.int32),
                ("tokens_per_expert", tokens_per_expert, [E], torch.int32),
                ("cu_seqlens", cu_seqlens, [E + B], torch.int32),
                ("plan.dst", plan.dst, [N], torch.int32),
                ("plan.experts_to_copy", plan.experts_to_copy, [R, B], torch.int32),
                ("plan.zero_fill_ranges", plan.zero_fill_ranges, [E + B, 2], torch.int32),
                ("plan.remote_stats", plan.remote_stats, [2], torch.int32),
                ("plan.dup_groups", plan.dup_groups, [NvS, 3], torch.int32),
                ("plan.dup_loffs", plan.dup_loffs, [NvS], torch.int32),
                ("plan.dup_counts", plan.dup_counts, [2], torch.int32),
                ("ctx.planning_workspace", ctx["planning_workspace"], None, torch.int32),
            ),
            scalars=(("shmem_instance_id", shmem_instance_id),),
        )

        workspace = self.sym_workspace()
        if workspace is None:
            raise RuntimeError("PlanningOp requires a symmetric workspace")
        launch_planning_op(
            topk_flat,
            plan.dst,
            cu_seqlens,
            plan.experts_to_copy,
            plan.dup_groups,
            plan.dup_loffs,
            plan.dup_counts,
            ctx["planning_workspace"],
            int(workspace.addr),
            int(workspace.length),
            S,
            K,
            E,
            NvS,
            B,
            int(self.num_sms),
            rank,
            R,
        )


class DispatchOp(KernelOp):
    """Corresponds to ``moonep.launch_dispatch``.

    compute(ctx, hidden_sh, route_weights_sk, plan, build_dedup_map=True,
            shmem_instance_id=0):
      - hidden_sh [S, H] bf16, route_weights_sk [S, K] fp32 or None
      - scatters payload rows to remote ranks' symmetric hidden chunks
        (dest = aclshmem_ptr(ctx['hidden_buf'].data_ptr(), drank) + loff*H*2),
        writes weights into ctx['weights_buf'] (remote rank's symmetric chunk).
      - build_dedup_map True only right after fresh planning.
      - shmem_instance_id: shmem runtime instance to operate on (default 0; the
        shmem Python API currently supports a single instance).
    v1 degrade: host put segment scatter per plan.dst ranges.
    """

    def compute(self, ctx, hidden_sh, route_weights_sk, plan, build_dedup_map=True,
                shmem_instance_id: int = 0):
        weights_buf = (
            ctx["weights_buf"].view(torch.float32)
            if route_weights_sk is not None else None
        )
        launch_dispatch_op(
            hidden_sh,
            route_weights_sk,
            plan.dst,
            plan.dup_groups,
            plan.dup_loffs,
            plan.dup_counts,
            ctx["hidden_buf"],
            weights_buf,
            ctx["dispatch_sync_buffer"],
            int(ctx["S"]),
            int(ctx["H"]),
            int(ctx["K"]),
            int(ctx["R"]),
            int(ctx["rank"]),
        )

    def compute_profile(self, ctx, hidden_sh, route_weights_sk, plan,
                        profile_cycles, build_dedup_map=True,
                        shmem_instance_id: int = 0):
        weights_buf = (
            ctx["weights_buf"].view(torch.float32)
            if route_weights_sk is not None else None
        )
        launch_dispatch_profile_op(
            hidden_sh,
            route_weights_sk,
            plan.dst,
            plan.dup_groups,
            plan.dup_loffs,
            plan.dup_counts,
            ctx["hidden_buf"],
            weights_buf,
            ctx["dispatch_sync_buffer"],
            profile_cycles,
            int(ctx["S"]),
            int(ctx["H"]),
            int(ctx["K"]),
            int(ctx["R"]),
            int(ctx["rank"]),
        )


class CombineOp(KernelOp):
    """Corresponds to ``moonep.launch_combine`` → Ascend ``moonEP_combine``.

    compute(ctx, output_sh, plan, output_sk=None, shmem_instance_id=0):
      - output_sh [S, H] bf16 (caller-allocated)
      - plan.dst / dup_* drive routing + prologue
      - reads remote rows from symmetric ``ctx['hidden_buf']`` via shmem_ptr
      - optional weight gather from ``ctx['weights_buf']`` into user-facing
        output_sk [S, K] fp32
      - launcher selects native/UDMA from ASCEND_MOONEP_COMBINE_USE_UDMA
    """

    def compute(self, ctx, output_sh, plan, output_sk=None,
                shmem_instance_id: int = 0):
        weight_buf = (
            ctx["weights_buf"].view(torch.float32)
            if output_sk is not None else None
        )
        launch_combine_op(
            ctx["hidden_buf"],
            weight_buf,
            ctx["combine_sync_buffer"],
            plan.dst,
            plan.dup_groups,
            plan.dup_loffs,
            plan.dup_counts,
            output_sh,
            output_sk,
            int(ctx["S"]),
            int(ctx["H"]),
            int(ctx["K"]),
            int(ctx["R"]),
            int(ctx["rank"]),
        )

class PrefetchOp(KernelOp):
    """Corresponds to ``moonep.launch_prefetch``.

    compute(ctx, weight_home, weight_slots, experts_to_copy, shmem_instance_id=0):
      - weight_home [E, H, H'] bf16 (source expert weights, rows on home ranks)
      - weight_slots [B, H, H'] bf16 (this rank's prefetch slots)
      - experts_to_copy [B] int32: global expert id to pull into slot b (-1 empty)
      - shmem_instance_id: shmem runtime instance to operate on (default 0; the
        shmem Python API currently supports a single instance).
    Cross-rank: pull home rows via host get (v1) or device RMA.
    """

    def compute(self, ctx, weight_home, weight_slots, experts_to_copy,
                shmem_instance_id: int = 0):
        # Inputs: remote_expert = full_weight[:E], weight_slots[E:E+B], experts_to_copy
        # Output: full_weight[E:E+B]
        E = int(ctx["E"]); B = int(ctx["B"])
        H = int(weight_home.shape[1])
        H_prime = int(weight_home.shape[2])
        _collect_inputs(
            "PrefetchOp",
            tensors=(
                ("weight_home(remote_expert)", weight_home, [E, H, H_prime], torch.bfloat16),
                ("weight_slots", weight_slots, [B, H, H_prime], torch.bfloat16),
                ("experts_to_copy", experts_to_copy, [B], torch.int32),
            ),
            scalars=(
                ("E", E), ("B", B), ("H", H), ("H_prime", H_prime),
                ("shmem_instance_id", shmem_instance_id),
            ),
        )
        return launch_prefetch(
            weight_home,
            weight_slots,
            experts_to_copy,
            self.num_sms,
        )


class GradReduceOp(KernelOp):
    """Corresponds to ``moonep.launch_grad_reduce``.

    compute(ctx, full_grad_home, reduce_buffer, experts_to_copy, rank, num_sms,
            shmem_instance_id=0):
      - full_grad_home [epn, H, H'] fp32 (this rank's physical home grad
        rows), where epn=E/R.  The logical ``full_grad`` projection remains
        ``[E+B,H,H']``; its owner source VA names only this rank's epn-row
        caller-owned backing.
      - reduce_buffer: the logical [R, B, H, H'] dist view (all R ranks'
        reduce buffers mapped as one, as validated by Buffer.reduce_grad). On
        Ascend there is no cross-rank contiguous VA: each rank owns its own
        [B, H, H'] shmem symmetric chunk, and peer chunk r is addressed as
        aclshmem_ptr(local_base, r) + b*H*H'*4 (device RMA) or materialized via
        sym_get_into (v1 host path).
      - accumulates remote duplicate grads back into home rows, then zeroes the
        consumed slots for the next microbatch.
      - shmem_instance_id: shmem runtime instance to operate on (default 0; the
        shmem Python API currently supports a single instance).
    Cross-rank: home rank gets each peer's reduce-buffer rows (host get / RMA).
    """

    def compute(self, ctx, full_grad_home, reduce_buffer, experts_to_copy,
                rank, num_sms, shmem_instance_id: int = 0):
        # full_grad_home is this rank's physical epn-row owner chunk;
        # reduce_buffer is this rank's B-row symmetric contribution chunk.
        R = int(ctx["R"]); E = int(ctx["E"]); B = int(ctx["B"])
        if R <= 0 or E % R:
            raise RuntimeError(f"invalid expert layout for grad reduce: E={E}, R={R}")
        epn = E // R
        H = int(full_grad_home.shape[1])
        H_prime = int(full_grad_home.shape[2])
        _collect_inputs(
            "GradReduceOp",
            tensors=(
                (
                    "full_grad_home(local epn rows)",
                    full_grad_home,
                    [epn, H, H_prime],
                    torch.float32,
                ),
                (
                    "reduce_buffer",
                    reduce_buffer,
                    [B, H, H_prime],
                    torch.float32,
                ),
                # The API passes the full [R, B] plan; the kernel uses every
                # row for reduction and this rank's row for local clearing.
                ("experts_to_copy", experts_to_copy, [R, B], torch.int32),
            ),
            scalars=(
                ("rank", rank),
                ("num_sms", num_sms),
                ("E", E),
                ("B", B),
                ("H", H),
                ("H_prime", H_prime),
                ("shmem_instance_id", shmem_instance_id),
            ),
        )
        launch_grad_reduce(
            full_grad_home,
            reduce_buffer,
            experts_to_copy,
            rank=int(rank),
            num_sms=int(num_sms),
            meta_buf=None,
            meta_stride=0,
            barrier_off=0,
            grid_sync_bar=None,
        )
