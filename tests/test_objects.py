"""CPU-only validation tests for MoonEP object capability contracts."""

from dataclasses import replace

import pytest
import torch

from moonep import Buffer, MoonEPDispatchTarget, MoonEPSourceStorage
from moonep.planning import MoonEPCommPlan


def make_plan() -> MoonEPCommPlan:
    N, R, E, B, NvS, K = 2, 2, 2, 1, 4, 1
    return MoonEPCommPlan(
        dst=torch.zeros(N, dtype=torch.int32),
        experts_to_copy=torch.zeros((R, B), dtype=torch.int32),
        zero_fill_ranges=torch.zeros((E + B, 2), dtype=torch.int32),
        remote_stats=torch.zeros(2, dtype=torch.int32),
        dup_groups=torch.zeros((NvS, 3), dtype=torch.int32),
        dup_loffs=torch.zeros(NvS, dtype=torch.int32),
        dup_counts=torch.zeros(2, dtype=torch.int32),
        N=N,
        R=R,
        E=E,
        B=B,
        NvS=NvS,
        K=K,
    )


def test_source_storage_layout_rejects_capability_and_shape() -> None:
    storage = MoonEPSourceStorage(
        mapped_hidden=torch.empty((4, 8), dtype=torch.bfloat16),
        local_hidden=torch.empty((2, 8), dtype=torch.bfloat16),
        mapped_route_weights=torch.empty(8, dtype=torch.float32),
        local_route_weights=torch.empty((2, 1), dtype=torch.float32),
        buffer_token=1,
        R=2,
        S=2,
        H=8,
        K=1,
        source_rows_padded=2,
        source_topk_padded=4,
    )
    storage.validate_layout()

    with pytest.raises(ValueError, match="capability"):
        replace(storage, capability="unknown").validate_layout()
    with pytest.raises(ValueError, match="mapped_hidden must have shape"):
        replace(
            storage,
            mapped_hidden=torch.empty((3, 8), dtype=torch.bfloat16),
        ).validate_layout()


def test_dispatch_target_state_checks_survive_python_optimization() -> None:
    plan = make_plan()
    mapped_hidden = torch.empty((8, 8), dtype=torch.bfloat16)
    local_hidden = mapped_hidden[:4]
    target = MoonEPDispatchTarget(
        mapped_hidden=mapped_hidden,
        local_hidden=local_hidden,
        plan=plan,
        generation=1,
        buffer_token=123,
        R=2,
        S=2,
        H=8,
        K=1,
        NvS=4,
        NvS_padded=4,
    )
    buffer = object.__new__(Buffer)
    buffer._dispatch_generation = 1
    buffer._published_dispatch_generation = None
    ctx = {
        "hidden_buf": mapped_hidden,
        "hidden_buf_local": local_hidden,
        "E": 2,
        "B": 1,
    }

    target = replace(target, buffer_token=id(buffer))
    buffer._validate_dispatch_target(ctx, target, plan)

    buffer._dispatch_generation = 2
    with pytest.raises(RuntimeError, match="stale"):
        buffer._validate_dispatch_target(ctx, target, plan)

    buffer._dispatch_generation = 1
    buffer._published_dispatch_generation = 1
    with pytest.raises(RuntimeError, match="already been published"):
        buffer._validate_dispatch_target(ctx, target, plan)


def test_dispatch_target_rejects_mutated_plan_layout() -> None:
    plan = make_plan()
    mapped_hidden = torch.empty((8, 8), dtype=torch.bfloat16)
    target = MoonEPDispatchTarget(
        mapped_hidden=mapped_hidden,
        local_hidden=mapped_hidden[:4],
        plan=plan,
        generation=1,
        buffer_token=1,
        R=2,
        S=2,
        H=8,
        K=1,
        NvS=4,
        NvS_padded=4,
    )
    object.__setattr__(
        plan,
        "zero_fill_ranges",
        torch.zeros((2, 2), dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="plan.zero_fill_ranges"):
        target.validate_layout()
