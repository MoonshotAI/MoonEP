"""TopK routing generation shared by communication benchmarks and tests."""

import torch


def generate_topk_routing(S, K, E, R, bias_ratio, dev, seed, rank=0, all_ranks=False):
    """Generate topk_experts and tokens_per_expert with benchmark semantics.

    bias_ratio is the sigma of the underlying lognormal expert-logit
    distribution:
      0   -> round-robin balanced baseline
      0.5 -> mild skew
      1   -> typical dropless-MoE skew
      2   -> heavy skew
      5   -> near-degenerate routing

    `seed` seeds rank-shared state (the expert-logit distribution and the
    round-robin expert permutation); `rank` seeds the per-token draws. This
    is bit-for-bit identical to the biased-routing generator used in
    training when run on the same cuda device: shared generator <- base
    seed, inner generator <- ep rank, so hot experts line up across ranks
    while per-token draws stay independent.

    Return contract (matches the ascend ``Buffer.dispatch`` input and the
    torch planning reference in ``planing_reference.py``):
      all_ranks=False (default) -> this rank's routing only:
          topk [S, K] int32  (topk_experts_sk), tpe [E] int32 (tokens_per_expert)
      all_ranks=True -> routing for every rank in one call, no cross-rank
          all_gather needed to build golden/reference:
          topk [R, S, K] int32, tpe [R, E] int32
    """
    assert K <= E, f"K ({K}) must be <= E ({E})"
    assert bias_ratio >= 0.0, f"bias_ratio must be >= 0, got {bias_ratio}"
    ranks = range(R) if all_ranks else (rank,)
    topk_list, tpe_list = [], []
    for r in ranks:
        topk_r, tpe_r = _generate_rank_routing(S, K, E, R, bias_ratio, dev, seed, r)
        topk_list.append(topk_r)
        tpe_list.append(tpe_r)
    if all_ranks:
        return torch.stack(topk_list), torch.stack(tpe_list)
    return topk_list[0], tpe_list[0]


def _generate_rank_routing(S, K, E, R, bias_ratio, dev, seed, rank):
    """Generate one rank's topk [S, K] and tokens_per_expert [E]."""
    g_shared = torch.Generator(device=dev).manual_seed(seed)
    g_local = torch.Generator(device=dev).manual_seed(rank)
    if bias_ratio == 0.0:
        epn = E // R
        toks = torch.arange(S, device=dev)
        ks = torch.arange(K, device=dev)
        target_rank = (toks[:, None] + ks[None, :]) % R
        target_local = ((toks[:, None] // R) + ks[None, :]) % epn
        perm = torch.randperm(epn, device=dev, generator=g_local)
        topk = (target_rank * epn + perm[target_local]).to(torch.int32)
    else:
        logits = torch.exp(torch.normal(
            mean=0.0, std=bias_ratio, size=(E,), device=dev, generator=g_shared
        ))
        probs = logits[None, :].expand(S, E)
        topk = torch.multinomial(
            probs, K, replacement=False, generator=g_local
        ).to(torch.int32)

    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return topk, tpe


def main():
    """Call examples for ``generate_topk_routing`` (run: python test_npu/generate_topk_routing.py)."""
    S, K, E, R = 64, 4, 32, 4
    seed, dev = 1234, "cpu"

    # 1) This rank's routing only -> feeds ascend Buffer.dispatch as
    #    topk_experts_sk [S, K] / tokens_per_expert [E].
    topk, tpe = generate_topk_routing(S, K, E, R, 1.0, dev, seed, rank=0)
    assert topk.shape == (S, K) and tpe.shape == (E,)
    assert int(tpe.sum()) == S * K
    assert bool(((0 <= topk) & (topk < E)).all())
    print(f"this rank: topk {tuple(topk.shape)} {topk.dtype}, "
          f"tpe {tuple(tpe.shape)} {tpe.dtype}")

    # 2) All ranks at once -> build golden / torch reference without all_gather.
    topk_all, tpe_all = generate_topk_routing(S, K, E, R, 1.0, dev, seed, all_ranks=True)
    assert topk_all.shape == (R, S, K) and tpe_all.shape == (R, E)
    assert int(tpe_all.sum()) == S * K * R
    assert torch.equal(topk_all[0], topk) and torch.equal(tpe_all[0], tpe)
    print(f"all ranks: topk {tuple(topk_all.shape)} {topk_all.dtype}, "
          f"tpe {tuple(tpe_all.shape)} {tpe_all.dtype}")

    # 3) Balanced baseline (bias_ratio=0) is deterministic round-robin.
    topk_b, tpe_b = generate_topk_routing(S, K, E, R, 0.0, dev, seed, all_ranks=True)
    assert int(tpe_b.sum()) == S * K * R
    print(f"balanced : topk {tuple(topk_b.shape)}, tpe {tuple(tpe_b.shape)}")

    print("generate_topk_routing OK")


if __name__ == "__main__":
    main()
