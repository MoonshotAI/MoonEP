"""Torch reference for the MoonEP planning kernel."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ReferenceDedupPlan:
    """Reference dup structures for the dispatch builder.

    The production builder allocates the compact prefixes with per-warp
    atomicAdds, so its ``dup_groups`` / ``dup_loffs`` ordering is not stable
    run-to-run; this reference uses a deterministic order and tests must
    compare group *sets* (see ``kernel_test_utils.dedup_plan_semantic_errors``),
    never element order.
    """

    dup_groups: torch.Tensor   # [NvS, 3] (primary_loff, dup_start, dup_n)
    dup_loffs: torch.Tensor    # [NvS]
    dup_counts: torch.Tensor   # [2] (n_groups, n_dup_loffs)


def launch_planning_torch_reference(
    ctx,
    topk_experts,
    tokens_per_expert,
    *,
    return_alloc=False,
):
    rank = ctx["rank"]
    R = ctx["R"]
    E = ctx["E"]
    B = int(ctx["B"])
    S = ctx["S"]
    K = ctx["K"]
    N = S * K
    epn = E // R
    CAP = ctx["NvS_capacity"]
    NvS = ctx.get("NvS", CAP) # TODO: a bit odd; this lets scans omit NvS
    token_padding = ctx.get("token_padding", 1)
    group = ctx.get("group")

    flat_topk_experts = topk_experts[rank].reshape(-1) if topk_experts.dim() == 3 else topk_experts.reshape(-1)
    if tokens_per_expert.dim() == 2:
        tpe = tokens_per_expert
    else:
        tpe = tokens_per_expert.reshape(1, E)
        if R > 1:
            gathered = [torch.zeros_like(tokens_per_expert) for _ in range(R)]
            dist.all_gather(gathered, tokens_per_expert, group=group)
            tpe = torch.stack(gathered)
    flat_topk_experts = flat_topk_experts.cpu()
    tpe = tpe.cpu()

    # tpe[src_rank, expert] is the token count of each expert on each source
    # rank. tpe_cumsum prefixes over source ranks; it later converts the
    # per-rank expert token index into a global one. The global sequence is
    # concatenated in source-rank order.
    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[R - 1]

    group_tokens = torch.zeros(R, dtype=expert_count.dtype)
    for h in range(R):
        group_tokens[h] = expert_count[h * epn:(h + 1) * epn].sum()

    # CAP is the real token capacity of each destination rank. Every source
    # rank has S*K routed entries, so sum(balance) = 0; positive means the
    # home group is overloaded, negative means the destination rank has room.
    balance = group_tokens - CAP

    # alloc[e, d] is how many global tokens of expert e land on dest rank d.
    # Note: kernel-side ctx['alloc'] uses the transposed [R, E] layout
    # (alloc[d*E+e]) so step4 reads coalesced by dest rank; here we compute
    # internally in [E, R] (clearest for migration/balancing) and transpose
    # on return_alloc to match the kernel buffer.
    alloc = torch.zeros(E, R, dtype=torch.int32)
    for e in range(E):
        alloc[e, e // epn] = expert_count[e]

    # z[h, d] is the token count home group h migrates to remote dest rank d.
    z = torch.zeros(R, R, dtype=torch.int32)

    # Choose remote receivers and their receive quotas.
    while True:
        # Pick the most overloaded home group and the roomiest dest rank; the
        # policy fills receivers fully, so chosen ones become balance[u] = 0.
        h = balance.argmax()
        u = balance.argmin()
        if balance[h] <= 0:
            break

        move = -balance[u]
        z[h, u] = move
        balance[h] -= move
        balance[u] = 0

    # From the receive quotas, compute exactly which tokens go to which
    # remote rank.
    for h in range(R):
        expert_start = h * epn
        remaining = expert_count[expert_start:expert_start + epn].clone()
        quotas = z[h].clone()

        while True:
            # Each round, re-pick the largest remote quota and the local
            # expert with the most remaining tokens.
            d = quotas.argmax()
            quota = quotas[d]
            if quota <= 0:
                break

            local_e = remaining.argmax()
            e = expert_start + local_e
            rem = remaining[local_e]

            take = torch.minimum(rem, quota)
            alloc[e, d] += take
            alloc[e, h] -= take
            remaining[local_e] -= take
            quotas[d] -= take

    if not torch.equal(alloc.sum(dim=1), expert_count):
        raise AssertionError("torch planning reference: per-expert token conservation failed")
    if bool((alloc.sum(dim=0) > CAP).any().item()):
        raise AssertionError("torch planning reference: rank capacity exceeded")

    if return_alloc:
        # Transpose to the kernel's [R, E] layout (flat as alloc[d*E + e]).
        return alloc.t().contiguous()

    # Planning lookup tables: build the tables Part 2 and dispatch/prefetch
    # will use.
    alloc_cumsum = alloc.cumsum(dim=1)

    expert_off = torch.zeros(R, E, dtype=torch.int32)
    cu_seqlens = torch.zeros(R, E + B, dtype=torch.int32)
    experts_to_copy = torch.full((R, B), -1, dtype=torch.int32)
    remote_stats_all = torch.zeros(R, 2, dtype=torch.int32)
    # zero_fill_ranges[d, g] = (pad_start_loff, n_pad_rows): the segment-
    # padding rows the dispatch zero warp clears. If n_pad_rows == 0, the
    # kernel leaves pad_start_loff as 0 because dispatch never reads it.
    # cnt == 0 groups occupy no space and contribute 0 pad rows; the
    # [last_padded_end, NvS) tail is intentionally NOT covered (master
    # behavior — consumers read via cu_seqlens, the tail is undefined).
    zero_fill_by_rank = torch.zeros(R, E + B, 2, dtype=torch.int32)

    for d in range(R):
        local_start = d * epn
        local_end = local_start + epn

        remote_experts = [
            e for e in range(E)
            if alloc[e, d].item() > 0 and not (local_start <= e < local_end)
        ]

        # Pick the B remote expert segments with the most tokens for VM
        # prefetch.
        remote_experts.sort(key=lambda e: (alloc[e, d].item(), e), reverse=True)
        remote_stats_all[d, 0] = len(remote_experts)
        for b, e in enumerate(remote_experts[:B]):
            experts_to_copy[d, b] = e
            remote_stats_all[e // epn, 1] += 1

        experts_to_prefetch = set(remote_experts[:B])
        start = 0

        # The physical token order follows the VM group id:
        #   0..E-1 are the global expert groups;
        #   E..E+B-1 are the prefetch slots of the selected remote experts.
        for g in range(E + B):
            cnt = 0
            expert_id = -1

            if g < E:
                if g not in experts_to_prefetch:
                    cnt = alloc[g, d].item()
                    expert_id = g
            else:
                b = g - E
                if experts_to_copy[d, b].item() >= 0:
                    expert_id = experts_to_copy[d, b].item()
                    cnt = alloc[expert_id, d].item()

            end = start + cnt
            if cnt > 0:
                padded = ((cnt + token_padding - 1) // token_padding) * token_padding
            else:
                padded = 0
            aligned_end = start + padded

            cu_seqlens[d, g] = aligned_end

            if cnt > 0:
                expert_off[d, expert_id] = start
                n_pad = padded - cnt
                if n_pad > 0:
                    zero_fill_by_rank[d, g, 0] = end
                    zero_fill_by_rank[d, g, 1] = n_pad

            start = aligned_end

        if cu_seqlens[d].max().item() > NvS:
            raise AssertionError("torch planning reference: padded layout exceeds NvS")

    # Part 2: local entry to dispatch offset.
    local_cnt = torch.zeros(N, dtype=torch.int32)
    counter = torch.zeros(E, dtype=torch.int32)
    for i in range(N):
        e = flat_topk_experts[i].item()
        local_cnt[i] = counter[e]
        counter[e] += 1

    global_rank = torch.zeros(N, dtype=torch.int32)
    for i in range(N):
        e = flat_topk_experts[i].item()
        prev = 0 if rank == 0 else tpe_cumsum[rank - 1, e].item()
        global_rank[i] = prev + local_cnt[i]

    dst = torch.zeros(N, dtype=torch.int32)
    for i in range(N):
        e = flat_topk_experts[i].item()
        g = global_rank[i].item()

        # Find the first dest rank whose cumulative allocation covers g, i.e.
        # the first alloc_cumsum[e, d] > g.
        dest = int(torch.searchsorted(alloc_cumsum[e], g, right=True).item())
        if dest >= R:
            raise AssertionError("torch planning reference: no destination rank found")

        prev_alloc = 0 if dest == 0 else alloc_cumsum[e, dest - 1].item()
        seg_pos = g - prev_alloc
        base_off = expert_off[dest, e].item()
        # Encoded global dispatch-buffer offset: high bits are the dest rank,
        # low bits are the offset within that rank's VM physical layout.
        dst[i] = dest * NvS + base_off + seg_pos

    # Part 3: dedup dst and produce the plan-owned dedup structures
    # (dup_groups / dup_loffs / dup_counts) the fresh dispatch builder
    # expects. The production planning kernel only emits canonical dst and
    # src_info; dedup structures materialize in fresh dispatch. The reference
    # returns equivalent structures here (deterministic order) for the
    # dispatch builder's semantic comparison.
    #
    # For dst, within each token's topK, only the slot offset at the first
    # occurrence of each rank is kept; the rest encode as -raw_dst - 1. The
    # reason: dispatch/combine weight scatter/gather needs the raw dst value
    # to write/read the meta_tensor slot, as each token-topk entry has its
    # own weight, so the raw dst must be recoverable to keep weight transfer
    # correct; the -1 ensures an encoded duplicate entry is always negative.
    dup_groups_by_rank = torch.zeros((R, NvS, 3), dtype=torch.int32)
    dup_loffs_by_rank = torch.zeros((R, NvS), dtype=torch.int32)
    dup_counts_by_rank = torch.zeros((R, 2), dtype=torch.int32)

    dst_by_rank = dst.unsqueeze(dim=0).contiguous()
    if R > 1:
        dev = tokens_per_expert.device
        gathered = [torch.zeros_like(dst, device=dev) for _ in range(R)]
        dist.all_gather(gathered, dst.to(device=dev), group=group)
        dst_by_rank = torch.stack(gathered).cpu()

    for src_rank in range(R):
        for s_tok in range(S):
            base = s_tok * K
            dst_vals = dst_by_rank[src_rank, base:base + K]
            dests = torch.div(dst_vals, NvS, rounding_mode="floor")
            loffs = dst_vals % NvS

            groups = {}
            indices = {}
            for k in range(K):
                dest = int(dests[k].item())
                groups.setdefault(dest, []).append(int(loffs[k].item()))
                indices.setdefault(dest, []).append(k)

            for dest, group_loffs in groups.items():
                dup_count = len(group_loffs) - 1
                primary_loff = group_loffs[0]
                if dup_count > 0:
                    group_idx = int(dup_counts_by_rank[dest, 0].item())
                    dup_start = int(dup_counts_by_rank[dest, 1].item())
                    dup_groups_by_rank[dest, group_idx, 0] = primary_loff
                    dup_groups_by_rank[dest, group_idx, 1] = dup_start
                    dup_groups_by_rank[dest, group_idx, 2] = dup_count
                    dup_loffs_by_rank[dest, dup_start:dup_start + dup_count] = (
                        dup_loffs_by_rank.new_tensor(group_loffs[1:])
                    )
                    dup_counts_by_rank[dest, 0] += 1
                    dup_counts_by_rank[dest, 1] += dup_count

                for i in range(1, len(group_loffs)):
                    if src_rank == rank:
                        idx = base + indices[dest][i]
                        dst[idx] = -dst[idx] - 1

    remote_stats = remote_stats_all[rank]

    dev = tokens_per_expert.device
    ref_dedup_plan = ReferenceDedupPlan(
        dup_groups=dup_groups_by_rank[rank].to(dev),
        dup_loffs=dup_loffs_by_rank[rank].to(dev),
        dup_counts=dup_counts_by_rank[rank].to(dev),
    )
    return (
        dst.to(dev),
        cu_seqlens[rank].to(dev),
        experts_to_copy.to(dev),
        remote_stats.to(dev),
        zero_fill_by_rank[rank].to(dev),
        ref_dedup_plan,
    )
