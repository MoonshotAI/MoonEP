import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from tests.generate_topk_routing import generate_topk_routing


DEFAULT_TOKEN_PADDING = 128

_ACTIVE_BUFFERS = []


def local_device_index() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class KernelCase:
    name: str
    S: int
    K: int
    epn: int
    H: int
    num_sms: int
    B: int | None = None
    token_padding: int = DEFAULT_TOKEN_PADDING
    routing: str = "balanced"
    bias_ratio: float = 0.0
    seed: int = 42
    min_R: int = 1
    max_R: int | None = None

    def E(self, R):
        return R * self.epn


def case_params(cases):
    return [pytest.param(case, id=case.name) for case in cases]


def init_case(case, R):
    from moonep import Buffer

    skip_if_unsupported_world_size(case, R)
    buffer = Buffer(
        case.S,
        case.H,
        case.K,
        case.E(R),
        R,
        B=case.B,
        num_sms=case.num_sms,
        token_padding=case.token_padding,
    )
    ctx = buffer._require_ctx()
    ctx["_buffer"] = buffer
    _ACTIVE_BUFFERS.append(buffer)
    return ctx


def destroy_active_buffers():
    while _ACTIVE_BUFFERS:
        buffer = _ACTIVE_BUFFERS.pop()
        if not buffer.destroyed:
            buffer.destroy()


def skip_if_unsupported_world_size(case, R):
    if R < case.min_R:
        pytest.skip(f"case {case.name} requires R >= {case.min_R}, got R={R}")
    if case.max_R is not None and R > case.max_R:
        pytest.skip(f"case {case.name} requires R <= {case.max_R}, got R={R}")


def make_topk(case, rank, R):
    skip_if_unsupported_world_size(case, R)

    dev = "cuda"
    E = case.E(R)
    if case.K > E and case.routing in {"balanced", "biased"}:
        pytest.skip(f"case {case.name} requires K <= E, got K={case.K}, E={E}")

    if case.routing in {"balanced", "biased"}:
        bias = case.bias_ratio if case.routing == "biased" else 0.0
        return generate_topk_routing(
            case.S, case.K, E, R, bias, dev, case.seed, rank=rank
        )

    s = torch.arange(case.S, device=dev)[:, None]
    k = torch.arange(case.K, device=dev)[None, :]
    epn = case.epn

    if case.routing == "all_local":
        topk = rank * epn + ((s + k) % epn)
    elif case.routing == "all_remote":
        remote_rank = (rank + 1) % R
        topk = remote_rank * epn + ((s + k) % epn)
    elif case.routing == "single_expert":
        topk = torch.zeros((case.S, case.K), dtype=torch.long, device=dev)
    elif case.routing == "duplicate_topk":
        expert = ((rank + 1) % R) * epn
        topk = torch.full((case.S, case.K), expert, dtype=torch.long, device=dev)
    else:
        raise ValueError(f"unknown routing pattern: {case.routing}")

    topk = topk.to(torch.int32).contiguous()
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return topk, tpe


def gather_tensor(t, R):
    gathered = [torch.zeros_like(t) for _ in range(R)]
    dist.all_gather(gathered, t)
    return torch.stack(gathered)


def assert_all_ranks(ok, rank, R, label, detail=""):
    ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    all_ok = gather_tensor(ok_tensor, R).cpu()
    if int(all_ok.sum().item()) != R:
        if not ok and detail:
            raise AssertionError(f"{label} failed on rank {rank}: {detail}")
        raise AssertionError(f"{label} failed on another rank")


def assert_tensor_equal_all_ranks(name, actual, expected, rank, R, max_print=5):
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    ok = torch.equal(actual_cpu, expected_cpu)
    detail = ""
    if not ok:
        diff_mask = actual_cpu != expected_cpu
        n_diff = int(diff_mask.sum().item())
        diff_pos = diff_mask.nonzero()[:max_print]
        lines = [f"{n_diff}/{actual_cpu.numel()} elements differ"]
        for pos in diff_pos:
            idx = tuple(int(v.item()) for v in pos)
            lines.append(
                f"{name}{idx}: actual={actual_cpu[idx].item()} "
                f"expected={expected_cpu[idx].item()}"
            )
        detail = "; ".join(lines)
    assert_all_ranks(ok, rank, R, name, detail)


DEDUP_PLAN_FIELDS = (
    "dup_groups",
    "dup_loffs",
    "dup_counts",
)


def clone_dedup_plan_fields(plan):
    return {field: getattr(plan, field).clone() for field in DEDUP_PLAN_FIELDS}


def dedup_plan_fields_equal(plan, snapshot):
    return all(torch.equal(getattr(plan, field), snapshot[field])
               for field in DEDUP_PLAN_FIELDS)


def _dedup_group_map(plan, *, max_print=5):
    """Build {primary_loff: sorted duplicate loff tuple} plus internal
    consistency errors. The builder allocates the compact prefixes with
    per-warp atomicAdds, so ``dup_groups`` / ``dup_loffs`` ordering is not
    stable — only the group set is compared."""
    groups = plan.dup_groups.cpu()
    dup_loffs = plan.dup_loffs.cpu()
    counts = plan.dup_counts.cpu()
    NvS = dup_loffs.numel()
    errors = []
    mapping = {}

    group_count = int(counts[0].item())
    dup_count_total = int(counts[1].item())
    if group_count < 0 or group_count > NvS:
        errors.append(f"dup group count {group_count} out of range [0, {NvS}]")
        group_count = max(0, min(group_count, NvS))
    if dup_count_total < 0 or dup_count_total > NvS:
        errors.append(f"dup loff count {dup_count_total} out of range [0, {NvS}]")
        dup_count_total = max(0, min(dup_count_total, NvS))

    seen_dups = []
    for group_idx in range(group_count):
        primary, dup_start, dup_count = (
            int(v) for v in groups[group_idx].tolist()
        )
        if not (0 <= primary < NvS):
            errors.append(f"dup group {group_idx} primary {primary} out of range")
            continue
        if not (0 <= dup_start <= dup_count_total):
            errors.append(f"dup group {group_idx} dup_start {dup_start} out of range")
            continue
        if dup_count <= 0 or dup_start + dup_count > dup_count_total:
            errors.append(
                f"dup group {group_idx} invalid dup range "
                f"start={dup_start} count={dup_count} total={dup_count_total}"
            )
            continue
        dups = []
        for offset_t in dup_loffs[dup_start:dup_start + dup_count].tolist():
            offset = int(offset_t)
            if not (0 <= offset < NvS):
                errors.append(
                    f"dup group {group_idx} contains out-of-range dup loff {offset}"
                )
                continue
            dups.append(offset)
            seen_dups.append(offset)
        if primary in mapping:
            errors.append(f"primary loff {primary} appears in multiple dup groups")
        if primary in dups:
            errors.append(f"dup group {group_idx} lists its own primary {primary}")
        mapping[primary] = tuple(sorted(dups))

    seen_dup_set = set(seen_dups)
    if len(seen_dups) != len(seen_dup_set):
        errors.append("dup_loffs contains repeated duplicate rows")
    if len(seen_dups) != dup_count_total:
        errors.append(
            f"dup loff count mismatch: header={dup_count_total}, used={len(seen_dups)}"
        )
    overlap = seen_dup_set & set(mapping.keys())
    if overlap:
        errors.append(
            f"rows appear both as primary and duplicate: {sorted(overlap)[:max_print]}"
        )

    return mapping, errors


def dedup_plan_semantic_errors(name, actual, expected, max_print=5):
    errors = []
    for field in DEDUP_PLAN_FIELDS:
        actual_t = getattr(actual, field)
        expected_t = getattr(expected, field)
        if actual_t.dtype != expected_t.dtype:
            errors.append(
                f"{field} dtype actual={actual_t.dtype} expected={expected_t.dtype}"
            )
        if tuple(actual_t.shape) != tuple(expected_t.shape):
            errors.append(
                f"{field} shape actual={tuple(actual_t.shape)} "
                f"expected={tuple(expected_t.shape)}"
            )
        if errors:
            return errors

    actual_counts = actual.dup_counts.cpu()
    expected_counts = expected.dup_counts.cpu()
    for idx, label in ((0, "dup group count"), (1, "dup loff count")):
        a = int(actual_counts[idx].item())
        e = int(expected_counts[idx].item())
        if a != e:
            errors.append(f"{name} {label} actual={a} expected={e}")

    actual_map, group_errors = _dedup_group_map(actual, max_print=max_print)
    expected_map, expected_group_errors = _dedup_group_map(
        expected, max_print=max_print
    )
    errors.extend(f"{name} actual {err}" for err in group_errors[:max_print])
    errors.extend(f"{name} expected {err}" for err in expected_group_errors[:max_print])
    if actual_map != expected_map:
        actual_keys = set(actual_map)
        expected_keys = set(expected_map)
        missing = sorted(expected_keys - actual_keys)[:max_print]
        extra = sorted(actual_keys - expected_keys)[:max_print]
        mismatched = [
            key for key in sorted(actual_keys & expected_keys)
            if actual_map[key] != expected_map[key]
        ][:max_print]
        errors.append(
            f"{name} duplicate group map differs: "
            f"missing={missing} extra={extra} mismatched={mismatched}"
        )

    return errors


def assert_dedup_plan_semantic_equal_all_ranks(
    name, actual, expected, rank, R, max_print=5
):
    errors = dedup_plan_semantic_errors(name, actual, expected, max_print=max_print)
    assert_all_ranks(
        not errors,
        rank,
        R,
        name,
        "; ".join(errors[:max_print]),
    )


def assert_close_all_ranks(name, actual, expected, rank, R, atol=0.05):
    diff = (actual.float() - expected.float()).abs()
    max_err = float(diff.max().item()) if diff.numel() else 0.0
    assert_all_ranks(max_err < atol, rank, R, name, f"max_err={max_err}")


def bf16_ulp(x):
    ax = x.abs().float()
    exp = torch.floor(torch.log2(ax.clamp(min=2**-126)))
    return (2.0 ** (exp - 7)).clamp(min=2**-133)


def assert_ulp_all_ranks(name, actual, expected, rank, R, max_ulps=1):
    diff = (actual.float() - expected.float()).abs()
    ulp = bf16_ulp(expected)
    ulp_err = diff / ulp.clamp(min=1e-30)
    max_ulp_err = float(ulp_err.max().item()) if ulp_err.numel() else 0.0
    assert_all_ranks(
        max_ulp_err <= max_ulps + 0.5,
        rank,
        R,
        name,
        f"max_ulp_err={max_ulp_err:.2f}",
    )


def planning_invariant_errors(case, ctx, dst, cu_seqlens, experts_to_copy):
    errors = []
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    NvS = int(ctx["NvS"])
    N = case.S * case.K

    planning_out_elems = (
        3 * E * R
        + R * (E + B)
        + 2 * R * (E + B)
        + R * B
        + 2 * R
    )
    n4 = _align_up(N, 4)
    expected_topk0_off = _align_up(int(ctx["PLAN_OFF"]) + planning_out_elems, 4)
    expected_order_off = expected_topk0_off + n4
    expected_order0_off = expected_order_off + n4
    expected_barrier_off = expected_order0_off + n4
    expected_src_info_off = expected_barrier_off + 3
    layout_checks = (
        ("TOPK0_OFF", expected_topk0_off),
        ("ORDER_OFF", expected_order_off),
        ("ORDER0_OFF", expected_order0_off),
        ("BARRIER_OFF", expected_barrier_off),
        ("SRC_INFO_OFF", expected_src_info_off),
    )
    for key, expected in layout_checks:
        actual = int(ctx[key])
        if actual != expected:
            errors.append(f"{key}={actual}, expected {expected}")
    if int(ctx["meta_chunk_padded"]) < expected_src_info_off + NvS:
        errors.append(
            f"meta_chunk_padded={int(ctx['meta_chunk_padded'])} is smaller than "
            f"src_info end={expected_src_info_off + NvS}"
        )

    if dst.dtype != torch.int32 or tuple(dst.shape) != (N,):
        errors.append(f"dst must be int32 [{N}], got {dst.dtype} {tuple(dst.shape)}")
    else:
        dst_cpu = dst.cpu()
        raw_dst = torch.where(dst_cpu < 0, -dst_cpu - 1, dst_cpu)
        dest_rank = torch.div(raw_dst, NvS, rounding_mode="floor")
        local_off = raw_dst % NvS
        if not torch.all((dest_rank >= 0) & (dest_rank < R)):
            errors.append("dst contains an out-of-range destination rank")
        if not torch.all((local_off >= 0) & (local_off < NvS)):
            errors.append("dst contains an out-of-range local offset")

    cu_cpu = cu_seqlens.cpu()
    if cu_seqlens.dtype != torch.int32 or tuple(cu_seqlens.shape) != (E + B,):
        errors.append(
            f"cu_seqlens must be int32 [{E + B}], "
            f"got {cu_seqlens.dtype} {tuple(cu_seqlens.shape)}"
        )
    else:
        prev = 0
        for gid, cur_t in enumerate(cu_cpu.tolist()):
            cur = int(cur_t)
            seg_len = cur - prev
            if seg_len < 0:
                errors.append(f"cu_seqlens decreases at group {gid}")
                break
            if seg_len and seg_len % case.token_padding != 0:
                errors.append(
                    f"group {gid} segment length {seg_len} is not divisible "
                    f"by token_padding={case.token_padding}"
                )
                break
            prev = cur
        if int(cu_cpu[-1].item()) > NvS:
            errors.append(f"cu_seqlens total {int(cu_cpu[-1].item())} exceeds NvS={NvS}")

    copy_cpu = experts_to_copy.cpu()
    if experts_to_copy.dtype != torch.int32 or tuple(experts_to_copy.shape) != (R, B):
        errors.append(
            f"experts_to_copy must be int32 [{R}, {B}], "
            f"got {experts_to_copy.dtype} {tuple(experts_to_copy.shape)}"
        )
    elif not torch.all((copy_cpu == -1) | ((copy_cpu >= 0) & (copy_cpu < E))):
        errors.append("experts_to_copy contains invalid expert ids")

    return errors


def dedup_plan_invariant_errors(case, ctx, plan):
    errors = []
    NvS = int(ctx["NvS"])
    expected_shapes = {
        "dup_groups": (NvS, 3),
        "dup_loffs": (NvS,),
        "dup_counts": (2,),
    }

    for field, shape in expected_shapes.items():
        t = getattr(plan, field)
        if t.dtype != torch.int32 or not t.is_contiguous() or tuple(t.shape) != shape:
            errors.append(
                f"{field} must be contiguous int32 {shape}, "
                f"got {t.dtype} {tuple(t.shape)}"
            )

    if errors:
        return errors

    counts = plan.dup_counts.cpu()
    for idx, label in ((0, "dup group count"), (1, "dup loff count")):
        count = int(counts[idx].item())
        if count < 0 or count > NvS:
            errors.append(f"{label} {count} is out of range [0, {NvS}]")

    _mapping, group_errors = _dedup_group_map(plan)
    errors.extend(group_errors)
    return errors
