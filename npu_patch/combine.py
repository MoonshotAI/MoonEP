"""Plan-driven MoonEP combine launch helpers for Ascend.

The public boundary mirrors ``dispatch.launch_dispatch``: callers pass ``ctx``
plus output tensors and a plan, while symmetric input shards are owned by the
Buffer and looked up from ``ctx``.  The Ascend kernel fuses prologue accumulate
and K-sum into one launch.
"""

from __future__ import annotations

import torch

from .launchers import _env_use_udma
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


def _check_combine_inputs(
    ctx: dict,
    output_sh: torch.Tensor,
    plan: MoonEPCommPlan,
    output_sk: torch.Tensor | None,
) -> None:
    if not isinstance(plan, MoonEPCommPlan):
        raise TypeError("plan must be a MoonEPCommPlan")

    use_udma = _env_use_udma("combine")
    required_ctx_keys = [
        "rank", "R", "E", "B", "S", "H", "K", "NvS", "kernels",
    ]
    if use_udma:
        required_ctx_keys.extend((
            "hidden_buf_local", "weights_buf_local", "combine_sync_buffer",
        ))
    else:
        required_ctx_keys.extend(("hidden_buf", "weights_buf"))
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
        raise ValueError(f"Ascend combine requires NvS=S*K, got {NvS} != {S*K}")
    if rank < 0 or rank >= R:
        raise ValueError(f"ctx rank must be in [0, {R}), got {rank}")
    if K > 16:
        raise ValueError("K must be <= 16")
    if use_udma:
        if H % 8 != 0:
            raise ValueError("UDMA combine requires H to be a multiple of 8")
        # Four bf16 rows plus fp32 accumulator/temp use 16 bytes per hidden
        # element in the UDMA kernel's 128 KiB full-row double buffer.
        if H * 16 > 128 * 1024:
            raise ValueError(
                "UDMA combine requires H<=8192 so its full-row double "
                "buffer fits in 128 KiB of UB"
            )
    else:
        if H % 16 != 0:
            raise ValueError("H must be a multiple of 16")
        if H * 10 + 256 > 192 * 1024:
            raise ValueError(
                "H too large for single-pass combine UB (max ~19680)"
            )

    _check_tensor(output_sh, "output_sh", (S, H), torch.bfloat16, output_sh.device)
    if output_sk is not None:
        _check_tensor(
            output_sk,
            "output_sk",
            (S, K),
            torch.float32,
            output_sh.device,
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
    _check_tensor(plan.dst, "plan.dst", (S * K,), torch.int32, output_sh.device)
    _check_tensor(
        plan.dup_groups,
        "plan.dup_groups",
        (NvS, 3),
        torch.int32,
        output_sh.device,
    )
    _check_tensor(
        plan.dup_loffs,
        "plan.dup_loffs",
        (NvS,),
        torch.int32,
        output_sh.device,
    )
    _check_tensor(
        plan.dup_counts,
        "plan.dup_counts",
        (2,),
        torch.int32,
        output_sh.device,
    )
    hidden_key = "hidden_buf_local" if use_udma else "hidden_buf"
    _check_tensor(
        ctx[hidden_key],
        f"ctx.{hidden_key}",
        (NvS, H),
        torch.bfloat16,
        output_sh.device,
    )
    if output_sk is not None:
        weights_key = "weights_buf_local" if use_udma else "weights_buf"
        weights_buf = ctx[weights_key].view(torch.float32)
        _check_tensor(
            weights_buf,
            f"ctx.{weights_key}(fp32 view)",
            (NvS,),
            torch.float32,
            output_sh.device,
        )

    if use_udma:
        sync_buffer = ctx["combine_sync_buffer"]
        if sync_buffer.dtype != torch.uint8 or not sync_buffer.is_contiguous():
            raise ValueError(
                "ctx.combine_sync_buffer must be contiguous torch.uint8"
            )
        if sync_buffer.device != output_sh.device:
            raise ValueError(
                "ctx.combine_sync_buffer must be on "
                f"{output_sh.device}, got {sync_buffer.device}"
            )


def launch_combine(
    ctx: dict,
    output_sh: torch.Tensor,
    plan: MoonEPCommPlan,
    output_sk: torch.Tensor | None = None,
) -> None:
    """Launch fused prologue + K-sum into caller-owned ``output_sh`` / ``output_sk``.

    Planning is intentionally outside this function.  ``plan`` must already
    contain canonical ``dst`` and the three duplicate tables.

    ``output_sk`` is the user-facing ``[S, K]`` gather buffer when present.
    AlignUp(K, 8) padding for the native C++ ABI lives in
    ``launch_combine_op`` (same Buffer call path as dispatch →
    ``kernels['*'].compute``).
    """
    _check_combine_inputs(ctx, output_sh, plan, output_sk)
    combine_op = ctx["kernels"].get("combine")
    if combine_op is None:
        raise KeyError("ctx['kernels']['combine'] is required")
    combine_op.compute(
        ctx,
        output_sh,
        plan,
        output_sk=output_sk,
    )
