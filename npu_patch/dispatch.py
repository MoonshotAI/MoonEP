"""Plan-driven MoonEP dispatch launch helpers for Ascend.

The public boundary mirrors ``moonep.dispatch.launch_dispatch``: callers pass
``ctx`` plus input tensors and a plan, while symmetric output tensors are owned
by the Buffer and looked up from ``ctx``.  The Ascend kernel fuses the remote
scatter and local duplicate expansion into one launch.
"""

from __future__ import annotations

import torch

from .planning import MoonEPCommPlan


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.dtype != dtype or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous {dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must be shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _check_dispatch_inputs(
    ctx: dict,
    hidden_sh: torch.Tensor,
    route_weights_sk: torch.Tensor | None,
    plan: MoonEPCommPlan,
) -> None:
    if not isinstance(plan, MoonEPCommPlan):
        raise TypeError("plan must be a MoonEPCommPlan")

    required_ctx_keys = (
        "rank", "R", "E", "B", "S", "H", "K", "NvS", "kernels",
        "hidden_buf", "hidden_buf_local", "weights_buf", "weights_buf_local",
        "dispatch_sync_buffer",
    )
    missing = [key for key in required_ctx_keys if key not in ctx]
    if missing:
        raise KeyError(f"ctx is missing required keys: {missing}")

    rank = int(ctx["rank"])
    R = int(ctx["R"])
    S = int(ctx["S"])
    H = int(ctx["H"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    if NvS != S * K:
        raise ValueError(f"Ascend dispatch requires NvS=S*K, got {NvS} != {S*K}")
    if rank < 0 or rank >= R:
        raise ValueError(f"ctx rank must be in [0, {R}), got {rank}")
    if H % 8 != 0:
        raise ValueError("H must be a multiple of 8")
    if H * hidden_sh.element_size() > 128 * 1024:
        raise ValueError("one hidden row exceeds the 128 KiB dispatch UB capacity")

    _check_tensor(hidden_sh, "hidden_sh", (S, H), torch.bfloat16, hidden_sh.device)
    if route_weights_sk is not None:
        _check_tensor(
            route_weights_sk,
            "route_weights_sk",
            (S, K),
            torch.float32,
            hidden_sh.device,
        )

    E = int(ctx["E"])
    B = int(ctx["B"])
    if (
        plan.N != S * K
        or plan.R != R
        or plan.E != E
        or plan.B != B
        or plan.K != K
        or plan.NvS != NvS
    ):
        raise ValueError(
            "plan scalar metadata must match ctx: "
            f"plan(N={plan.N}, R={plan.R}, E={plan.E}, B={plan.B}, "
            f"K={plan.K}, NvS={plan.NvS}) vs "
            f"ctx(N={S*K}, R={R}, E={E}, B={B}, K={K}, NvS={NvS})"
        )
    _check_tensor(plan.dst, "plan.dst", (S * K,), torch.int32, hidden_sh.device)
    _check_tensor(
        plan.dup_groups,
        "plan.dup_groups",
        (NvS, 3),
        torch.int32,
        hidden_sh.device,
    )
    _check_tensor(
        plan.dup_loffs,
        "plan.dup_loffs",
        (NvS,),
        torch.int32,
        hidden_sh.device,
    )
    _check_tensor(
        plan.dup_counts,
        "plan.dup_counts",
        (2,),
        torch.int32,
        hidden_sh.device,
    )
    _check_tensor(
        ctx["hidden_buf"],
        "ctx.hidden_buf",
        (NvS, H),
        torch.bfloat16,
        hidden_sh.device,
    )
    if route_weights_sk is not None:
        weights_buf = ctx["weights_buf"].view(torch.float32)
        _check_tensor(
            weights_buf,
            "ctx.weights_buf(fp32 view)",
            (NvS,),
            torch.float32,
            hidden_sh.device,
        )


def launch_dispatch(
    ctx: dict,
    hidden_sh: torch.Tensor,
    route_weights_sk: torch.Tensor | None,
    plan: MoonEPCommPlan,
) -> None:
    """Launch fused scatter plus duplicate expansion into Buffer-owned SHMEM.

    Planning is intentionally outside this function.  ``plan`` must already
    contain canonical ``dst`` and the three duplicate tables.
    """
    _check_dispatch_inputs(ctx, hidden_sh, route_weights_sk, plan)
    dispatch_op = ctx["kernels"].get("dispatch")
    if dispatch_op is None:
        raise KeyError("ctx['kernels']['dispatch'] is required")
    dispatch_op.compute(ctx, hidden_sh, route_weights_sk, plan)


def materialize_dispatch_outputs(
    ctx: dict,
    *,
    with_weights: bool,
    zero_copy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return shared output aliases or copy them into ordinary NPU tensors."""
    hidden_local = ctx["hidden_buf_local"]
    weights_local = ctx["weights_buf_local"].view(torch.float32)
    if zero_copy:
        return hidden_local, weights_local if with_weights else None
    return (
        hidden_local.clone(),
        weights_local.clone() if with_weights else None,
    )
