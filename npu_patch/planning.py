"""
Local, CUDA-free mirror of the ``moonep/planning.py`` data structures.

``MoonEPCommPlan`` and ``allocate_planning_outputs`` are ported verbatim except
they depend only on torch (device comes from ``ctx``), so this module imports
cleanly without CUDA/CuTe. The planning *kernel* itself is provided by the
``PlanningOp`` (AscendC kernel migration).
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    dst: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    N: int
    R: int
    E: int
    B: int
    NvS: int
    K: int

    # Dedup structures materialized by the dispatch builder and consumed by the
    # dispatch epilogue / combine prologue. dup_counts = [n_groups, n_dup_loffs];
    # only the compact prefix of dup_groups/dup_loffs is valid.
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor

    def __post_init__(self) -> None:
        N = int(self.N)
        R = int(self.R)
        E = int(self.E)
        B = int(self.B)
        NvS = int(self.NvS)
        assert self.dst.dtype == torch.int32 and self.dst.is_contiguous()
        assert self.dst.numel() == N
        assert self.experts_to_copy.dtype == torch.int32 and self.experts_to_copy.is_contiguous()
        assert tuple(self.experts_to_copy.shape) == (R, B)
        assert self.remote_stats.dtype == torch.int32 and self.remote_stats.is_contiguous()
        assert tuple(self.remote_stats.shape) == (2,)
        assert self.zero_fill_ranges.dtype == torch.int32 and self.zero_fill_ranges.is_contiguous()
        assert tuple(self.zero_fill_ranges.shape) == (E + B, 2)
        assert self.dup_groups.dtype == torch.int32 and self.dup_groups.is_contiguous()
        assert tuple(self.dup_groups.shape) == (NvS, 3)
        assert self.dup_loffs.dtype == torch.int32 and self.dup_loffs.is_contiguous()
        assert tuple(self.dup_loffs.shape) == (NvS,)
        assert self.dup_counts.dtype == torch.int32 and self.dup_counts.is_contiguous()
        assert tuple(self.dup_counts.shape) == (2,)

    def clone(self) -> "MoonEPCommPlan":
        """Deep-clone every plan tensor (mirrors ``moonep.planning``; the tests
        snapshot plans with ``plan.clone()``)."""
        return type(self)(
            dst=self.dst.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dup_groups=self.dup_groups.clone(),
            dup_loffs=self.dup_loffs.clone(),
            dup_counts=self.dup_counts.clone(),
            N=self.N,
            R=self.R,
            E=self.E,
            B=self.B,
            NvS=self.NvS,
            K=self.K,
        )


def _round4(n: int) -> int:
    return (n + 3) & ~3


def allocate_planning_outputs(ctx: dict):
    """Allocate a ``(MoonEPCommPlan, cu_seqlens)`` pair on ctx's NPU device.

    The plan-owned dedup tensors are allocated here; the Ascend Planning kernel
    materializes their compact prefixes together with ``dup_counts``.
    """
    E = ctx['E']
    B = ctx.get('B', 0)
    S = ctx['S']
    K = ctx['K']
    N = S * K
    NvS = ctx['NvS']
    dev = ctx['weights_buf'].device

    # Over-allocate output tensors to a multiple of 4 and then slice, avoiding
    # an out-of-bounds full-vector write at the tail (returned shapes unchanged).
    dst = torch.empty(_round4(N), dtype=torch.int32, device=dev)[:N]
    cu_seqlens = torch.empty(_round4(E + B), dtype=torch.int32, device=dev)[:E + B]
    experts_to_copy = torch.empty(_round4(ctx['R'] * B), dtype=torch.int32, device=dev)[
        :ctx['R'] * B
    ].view(ctx['R'], B)
    zero_fill_ranges = torch.empty(
        _round4((E + B) * 2), dtype=torch.int32, device=dev
    )[:(E + B) * 2].view(E + B, 2)
    remote_stats = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]
    dup_groups = torch.empty(
        _round4(NvS * 3), dtype=torch.int32, device=dev
    )[:NvS * 3].view(NvS, 3)
    dup_loffs = torch.empty(_round4(NvS), dtype=torch.int32, device=dev)[:NvS]
    dup_counts = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]

    plan = MoonEPCommPlan(
        dst=dst,
        experts_to_copy=experts_to_copy,
        zero_fill_ranges=zero_fill_ranges,
        remote_stats=remote_stats,
        dup_groups=dup_groups,
        dup_loffs=dup_loffs,
        dup_counts=dup_counts,
        N=N,
        R=ctx['R'],
        E=E,
        B=B,
        NvS=NvS,
        K=K,
    )
    return plan, cu_seqlens
