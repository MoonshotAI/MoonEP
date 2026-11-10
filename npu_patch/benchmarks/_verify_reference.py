"""Reference models for the ``bench_vs.py --verify`` plan / dispatch stages.

Pure ``torch`` (CPU) so the models carry no NPU dependency and can be checked
against ``test_npu/planing_reference.py`` offline.  The planning model mirrors
the master torch reference (greedy MaxVio balancing -> prefetch layout ->
per-route ``dst`` -> duplicate encoding), but is vectorized: the Python-loop
reference needs ~1 minute per MaxVio point at benchmark scale
(R=8, S=8192, K=16, E=1024), while these models run in ~1 s.

``plan_expectation`` returns the full expected plan for every rank;
``dispatch_expectation`` maps the gathered ``dst``/weights/hidden fingerprints
to the slot layout of one destination rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Bound the [chunk, R] destination-search workspace of the route model: a
# 65536-route chunk is 4 MiB of int64 boundaries for R=8.
_ROUTE_CHUNK = 65536


# ---------------------------------------------------------------------------
# planning model
# ---------------------------------------------------------------------------


def local_occurrence(experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Per-route occurrence index of its expert within ``experts``.

    Stable bucketing per expert: the i-th route (in input order) that selects
    expert ``e`` gets index ``i``.  This is the per-rank ordinal the planning
    reference adds to the cross-rank expert prefix.
    """
    order = torch.argsort(experts, stable=True)
    sorted_experts = experts[order]
    counts = torch.bincount(experts, minlength=num_experts)
    starts = counts.cumsum(0) - counts
    local = (torch.arange(experts.numel(), dtype=torch.int64)
             - starts[sorted_experts])
    out = torch.empty_like(local)
    out[order] = local
    return out


def policy_alloc(
    tpe_all: torch.Tensor, R: int, E: int, NvS: int
) -> torch.Tensor:
    """Greedy MaxVio balancing policy: expected ``[E, R]`` token allocation.

    ``tpe_all`` is ``[R, E]`` per-source-rank expert counts; every destination
    rank has capacity ``NvS``.  Home overload is drained into the ranks with
    the most free capacity (exactly ``MoonEP/planing_reference``).
    """
    epn = E // R
    tpe_all = tpe_all.to(torch.int64)
    expert_count = tpe_all.sum(0)
    group_tokens = expert_count.view(R, epn).sum(1)
    balance = group_tokens - NvS

    # z[h, d]: tokens home group h sends to destination rank d.
    z = torch.zeros(R, R, dtype=torch.int64)
    while True:
        h = int(balance.argmax())
        u = int(balance.argmin())
        if int(balance[h]) <= 0:
            break
        move = -int(balance[u])
        z[h, u] = move
        balance[h] -= move
        balance[u] = 0

    alloc = torch.zeros(E, R, dtype=torch.int64)
    for e in range(E):
        alloc[e, e // epn] = expert_count[e]

    for h in range(R):
        expert_base = h * epn
        remaining = expert_count[expert_base:expert_base + epn].clone()
        quotas = z[h].clone()
        while True:
            d = int(quotas.argmax())
            quota = int(quotas[d])
            if quota <= 0:
                break
            le = int(remaining.argmax())
            rem = int(remaining[le])
            if rem <= 0:
                break
            take = min(quota, rem)
            e = expert_base + le
            alloc[e, d] += take
            alloc[e, h] -= take
            remaining[le] -= take
            quotas[d] -= take

    return alloc


def prefetch_experts(
    alloc: torch.Tensor, R: int, E: int, B: int
) -> torch.Tensor:
    """Expected ``[R, B]`` prefetch slot table (``-1`` = idle slot)."""
    epn = E // R
    out = torch.full((R, B), -1, dtype=torch.int64)
    counts = alloc.tolist()
    for d in range(R):
        local_start = d * epn
        local_end = local_start + epn
        remote = [
            (counts[e][d], e)
            for e in range(E)
            if counts[e][d] > 0 and not (local_start <= e < local_end)
        ]
        # Most tokens first; ties by larger expert id (reference order).
        remote.sort(reverse=True)
        for b, (_, e) in enumerate(remote[:B]):
            out[d, b] = e
    return out


def plan_layout(
    alloc: torch.Tensor,
    experts_to_copy: torch.Tensor,
    R: int,
    E: int,
    B: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expected ``cu_seqlens`` ``[R, E+B]`` and ``expert_off`` ``[R, E]``.

    Groups ``0..E-1`` hold local experts that are not prefetched; groups
    ``E..E+B-1`` hold the prefetched remote experts.  A zero-count group
    occupies no space.
    """
    cu = torch.zeros(R, E + B, dtype=torch.int64)
    expert_off = torch.zeros(R, E, dtype=torch.int64)
    epn = E // R
    for d in range(R):
        start = 0
        prefetched = {
            int(v) for v in experts_to_copy[d].tolist() if v >= 0
        }
        local_start = d * epn
        local_end = local_start + epn
        for g in range(E + B):
            if g < E:
                expert = g
                # Local experts are only laid out here when not prefetched;
                # prefetched ones live in their slot group.
                remote = not (local_start <= expert < local_end)
                cnt = (0 if (remote and expert in prefetched)
                       else int(alloc[expert, d]))
            else:
                expert = int(experts_to_copy[d, g - E])
                cnt = int(alloc[expert, d]) if expert >= 0 else 0
            if cnt > 0:
                expert_off[d, expert] = start
            start += cnt
            cu[d, g] = start
    return cu, expert_off


def _first_route_per_destination(dest_sk: torch.Tensor) -> torch.Tensor:
    """``[S, K]`` bool: first top-k route of each token to a destination rank.

    Stable: the first entry in top-k order is canonical; later routes to the
    same destination rank are duplicates.
    """
    order = torch.argsort(dest_sk, dim=1, stable=True)
    sorted_dest = dest_sk.gather(1, order)
    first_sorted = torch.ones_like(sorted_dest, dtype=torch.bool)
    first_sorted[:, 1:] = sorted_dest[:, 1:] != sorted_dest[:, :-1]
    first = torch.empty_like(first_sorted)
    first.scatter_(1, order, first_sorted)
    return first


def _route_destinations(
    topk_all: torch.Tensor,
    tpe_all: torch.Tensor,
    alloc: torch.Tensor,
    expert_off: torch.Tensor,
    NvS: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expected per-route destination/loff/canonical flags for every rank.

    Returns ``(dest, loff, canonical, raw)``, each ``[R, N]`` with
    ``N = S * K``: destination rank, offset inside the destination layout,
    first-route-to-destination flag (canonical vs duplicate) and the
    un-encoded ``dest * NvS + loff``.
    """
    R, S, K = topk_all.shape
    N = S * K
    E = alloc.shape[0]
    alloc_cumsum = alloc.cumsum(1)
    tpe_cumsum = tpe_all.to(torch.int64).cumsum(0)
    prev_count = torch.zeros(R, E, dtype=torch.int64)
    prev_count[1:] = tpe_cumsum[:-1]

    dest = torch.empty(R, N, dtype=torch.int64)
    loff = torch.empty(R, N, dtype=torch.int64)
    canonical = torch.empty(R, N, dtype=torch.bool)
    flat = topk_all.reshape(R, N).to(torch.int64)

    for r in range(R):
        experts = flat[r]
        ordinal = local_occurrence(experts, E)
        # Global (cross-rank) ordinal of this route's expert.
        g = prev_count[r][experts] + ordinal

        # dest = first rank whose cumulative allocation covers g.
        d = torch.empty(N, dtype=torch.int64)
        for base in range(0, N, _ROUTE_CHUNK):
            end = min(base + _ROUTE_CHUNK, N)
            boundaries = alloc_cumsum[experts[base:end]]
            d[base:end] = (boundaries <= g[base:end, None]).sum(1)
        dest[r] = d

        prev_alloc = torch.where(
            d > 0,
            alloc_cumsum[experts, (d - 1).clamp_min(0)],
            torch.zeros_like(d),
        )
        loff[r] = expert_off[d, experts] + g - prev_alloc
        canonical[r] = _first_route_per_destination(d.view(S, K)).reshape(-1)

    raw = dest * NvS + loff
    return dest, loff, canonical, raw


def encode_dst(raw: torch.Tensor, canonical: torch.Tensor) -> torch.Tensor:
    """Apply the duplicate encoding: later routes store ``-raw - 1``."""
    return torch.where(canonical, raw, -raw - 1)


def dedup_pairs(
    dest: torch.Tensor,
    loff: torch.Tensor,
    canonical: torch.Tensor,
    rank: int,
    S: int,
    NvS: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Expected dedup table for one destination rank, as semantic pairs.

    Returns ``(pairs, (n_groups, n_dups))`` where ``pairs`` is a lexicographically
    sorted ``[n_dups, 2]`` int64 tensor of ``(primary_loff, duplicate_loff)``.
    Group order is irrelevant (the kernel assigns it with atomics), so only the
    ``primary -> {dups}`` mapping is compared.
    """
    R, N = dest.shape
    K = N // S
    src = torch.arange(R, dtype=torch.int64).view(R, 1).expand(R, N)
    token = torch.arange(N, dtype=torch.int64).view(1, N) // K
    # Group id: (src rank, token, destination rank).
    key = (src * S + token) * R + dest

    local = dest == rank
    prim_mask = local & canonical
    dup_mask = local & ~canonical

    prim_keys = key[prim_mask]
    prim_loffs = loff[prim_mask]
    primary = torch.zeros(int(key.max().item()) + 1, dtype=torch.int64)
    primary[prim_keys] = prim_loffs

    dup_keys = key[dup_mask]
    dup_loffs = loff[dup_mask]
    n_dups = int(dup_keys.numel())
    if n_dups == 0:
        empty = torch.zeros(0, 2, dtype=torch.int64)
        return empty, (0, 0)

    uniq, inverse = torch.unique(dup_keys, return_inverse=True)
    pair_primary = primary[uniq][inverse]
    pairs = torch.stack((pair_primary, dup_loffs), dim=1)
    pairs = pairs[torch.argsort(pairs[:, 0] * NvS + pairs[:, 1])]
    return pairs, (int(uniq.numel()), n_dups)


def plan_dedup_pairs(
    dup_groups: torch.Tensor,
    dup_loffs: torch.Tensor,
    dup_counts: torch.Tensor,
    NvS: int,
) -> tuple[torch.Tensor, tuple[int, int], list[str]]:
    """Normalize a kernel-plan dedup table to the same semantic pair form.

    Also validates the table's structure (compact prefixes, group ranges,
    duplicate offsets in range), returning human-readable errors.
    """
    errors: list[str] = []
    n_groups = int(dup_counts[0])
    n_dups = int(dup_counts[1])
    if not 0 <= n_groups <= NvS:
        errors.append(f"dup group count {n_groups} out of [0, {NvS}]")
    if not 0 <= n_dups <= NvS:
        errors.append(f"dup loff count {n_dups} out of [0, {NvS}]")
    if errors:
        return torch.zeros(0, 2, dtype=torch.int64), (n_groups, n_dups), errors

    groups = dup_groups[:n_groups].to(torch.int64)
    prim = groups[:, 0]
    starts = groups[:, 1]
    counts = groups[:, 2]
    bad = (counts <= 0) | (starts < 0) | (starts + counts > n_dups)
    if bool(bad.any()):
        errors.append(
            f"{int(bad.sum())} malformed dup groups (count<=0 or range outside "
            f"n_dups={n_dups})"
        )
        keep = ~bad
        prim, starts, counts = prim[keep], starts[keep], counts[keep]
    if bool(((prim < 0) | (prim >= NvS)).any()):
        errors.append("dup group primary loff out of range")

    total = int(counts.sum())
    if total:
        offsets = torch.arange(total, dtype=torch.int64)
        group_offsets = torch.repeat_interleave(counts.cumsum(0) - counts, counts)
        within = offsets - group_offsets
        dup_idx = torch.repeat_interleave(starts, counts) + within
        if int(dup_idx.max()) >= n_dups or int(dup_idx.min()) < 0:
            errors.append("dup group references loffs outside the compact prefix")
            dup_idx = dup_idx.clamp(0, max(n_dups - 1, 0))
        dup_values = dup_loffs[dup_idx].to(torch.int64)
        pairs = torch.stack((torch.repeat_interleave(prim, counts), dup_values), 1)
        if int(torch.unique(dup_values).numel()) != total:
            errors.append("dup_loffs lists the same duplicate row twice")
        pairs = pairs[torch.argsort(pairs[:, 0] * NvS + pairs[:, 1])]
    else:
        pairs = torch.zeros(0, 2, dtype=torch.int64)

    return pairs, (n_groups, n_dups), errors


@dataclass(frozen=True)
class PlanExpectation:
    """Expected plan tensors derived from the gathered routing."""

    alloc: torch.Tensor            # [E, R]
    experts_to_copy: torch.Tensor  # [R, B]
    cu_seqlens: torch.Tensor       # [R, E+B]
    expert_off: torch.Tensor       # [R, E]
    dest: torch.Tensor             # [R, N] destination rank per route
    loff: torch.Tensor             # [R, N]
    canonical: torch.Tensor        # [R, N] bool
    dst: torch.Tensor              # [R, N] expected encoded dst
    dedup_pairs: torch.Tensor      # [n_dups, 2] for this rank's destination
    dedup_counts: tuple[int, int]
    tpe_global: torch.Tensor       # [E]


def plan_expectation(
    topk_all: torch.Tensor,
    tpe_all: torch.Tensor,
    *,
    rank: int,
    NvS: int,
) -> PlanExpectation:
    """Build the expected plan for every rank from the full routing."""
    topk_all = topk_all.to(torch.int64)
    tpe_all = tpe_all.to(torch.int64)
    R, S, K = topk_all.shape
    E = int(tpe_all.shape[1])
    B = E // R

    alloc = policy_alloc(tpe_all, R, E, NvS)
    experts_to_copy = prefetch_experts(alloc, R, E, B)
    cu_seqlens, expert_off = plan_layout(alloc, experts_to_copy, R, E, B)
    dest, loff, canonical, raw = _route_destinations(
        topk_all, tpe_all, alloc, expert_off, NvS
    )
    dst = encode_dst(raw, canonical)
    pairs, counts = dedup_pairs(dest, loff, canonical, rank, S, NvS)
    return PlanExpectation(
        alloc=alloc,
        experts_to_copy=experts_to_copy,
        cu_seqlens=cu_seqlens,
        expert_off=expert_off,
        dest=dest,
        loff=loff,
        canonical=canonical,
        dst=dst,
        dedup_pairs=pairs,
        dedup_counts=counts,
        tpe_global=tpe_all.sum(0),
    )


# ---------------------------------------------------------------------------
# dispatch model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchExpectation:
    """Expected dispatch slots for one destination rank."""

    weight: torch.Tensor        # [NvS] fp32 expected route weight per slot
    hidden_fp: torch.Tensor     # [NvS, F] expected hidden-row fingerprint
    covered: int                # routes that decode to this rank
    collisions: int             # slots written by more than one route


def dispatch_expectation(
    dst_all: torch.Tensor,
    weights_all: torch.Tensor,
    hidden_fp_all: torch.Tensor,
    *,
    rank: int,
    NvS: int,
) -> DispatchExpectation:
    """Map gathered routing/weights/fingerprints onto one rank's slots.

    ``dst_all`` is ``[R, N]`` int32 (canonical >= 0, duplicate ``-raw - 1``),
    ``weights_all`` ``[R, S, K]`` fp32 and ``hidden_fp_all`` ``[R, S, F]``
    fp32 row fingerprints.  Duplicate routes carry their own weight but the
    hidden row of their token, so both are keyed by the decoded slot.
    """
    R, N = dst_all.shape
    dst = dst_all.to(torch.int64)
    raw = torch.where(dst >= 0, dst, -dst - 1)
    dest = raw // NvS
    loff = raw - dest * NvS

    local = dest == rank
    flat_w = weights_all.reshape(R, N).to(torch.float32)
    src = torch.arange(R, dtype=torch.int64).view(R, 1).expand(R, N)
    route = torch.arange(N, dtype=torch.int64).view(1, N).expand(R, N)

    sel_src = src[local]
    sel_route = route[local]
    sel_loff = loff[local]

    weight = torch.zeros(NvS, dtype=torch.float32)
    weight[sel_loff] = flat_w[sel_src, sel_route]

    token = sel_route // weights_all.shape[2]
    out_fp = torch.zeros(NvS, hidden_fp_all.shape[2], dtype=torch.float32)
    out_fp[sel_loff] = hidden_fp_all[sel_src, token].to(torch.float32)

    counts = torch.bincount(sel_loff, minlength=NvS)
    return DispatchExpectation(
        weight=weight,
        hidden_fp=out_fp,
        covered=int(local.sum()),
        collisions=int(((counts - 1).clamp_min(0)).sum()),
    )


def row_fingerprint(
    rows: torch.Tensor, chunk_rows: int = 8192
) -> torch.Tensor:
    """Per-row fingerprint ``[sum, abs-sum, ramp-dot, sum of squares]``.

    Rows are identical bit patterns on both ends of the dispatch (bf16 copy),
    so the fingerprint is deterministic; it is compared with a small relative
    tolerance because the source and destination tensors have different shapes
    (``[S, H]`` vs ``[NvS, H]``) which can change the reduction blocking.
    """
    num_rows, hidden = rows.shape
    device = rows.device
    ramp = torch.arange(hidden, dtype=torch.float32, device=device)
    ramp = (ramp - (hidden - 1) * 0.5) / max(hidden, 1)
    out = torch.empty(num_rows, 4, dtype=torch.float32, device=device)
    for base in range(0, num_rows, chunk_rows):
        end = min(base + chunk_rows, num_rows)
        block = rows[base:end].float()
        out[base:end, 0] = block.sum(1)
        out[base:end, 1] = block.abs().sum(1)
        out[base:end, 2] = (block * ramp).sum(1)
        out[base:end, 3] = (block * block).sum(1)
    return out