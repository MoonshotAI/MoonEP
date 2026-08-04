# MoonEP

MoonEP is an Expert Parallelism communication library that keeps token loads perfectly balanced across ranks via dynamic redundant experts.

**Notation**: `S` = input tokens per rank, `K` = routed top-k per token.

1. **Perfect balance**: every rank receives exactly `S × K` tokens, no matter how skewed the routing is. A small number of redundant experts is planned online from the current router outputs and prefetched before expert computation; their gradients are reduced back to their home ranks in the backward pass.
2. **Online planning**: a near-optimal GPU planning kernel with negligible overhead
3. **Zero copy and static shapes**: fused permute/unpermute — tokens are sent directly to their expert-grouped positions on remote ranks and buffer views are returned to the computation. Only a fixed `S × K` buffer is needed, and statically known shapes eliminate per-layer MoE host synchronization.

## Performance

Both benchmarks run on H20 with EP=8, sweeping the router imbalance:

$$\text{maxvio} = \max_e \left( \frac{T_e}{\bar{T}} \right) - 1$$

where $T_e$ is the number of tokens routed to expert $e$, and $\bar{T}$ is the expected tokens per expert under perfect balance (maxvio = 0 means perfectly balanced).

**Communication vs DeepEP v2** ([benchmarks/bench_vs_deepep.py](benchmarks/bench_vs_deepep.py)):

<img src="figure/comm_vs_deepep.png" alt="MoonEP vs DeepEP v2 communication" width="800">

- **Zero copy makes raw communication faster**: tokens are written directly to their final expert-grouped positions on remote ranks — no permute in, no permute out — and views of the communication buffer are handed straight to the computation, eliminating the comm-buffer → user-buffer copy that dominates the epilogue. MoonEP's comm time is consistently below DeepEP v2 at every imbalance level.
- **Perfect balance makes it immune to imbalance**: MoonEP's comm time stays almost flat as maxvio grows, while DeepEP v2 — whose latency is set by the hottest rank — degrades steadily.
- **The comparison counts MoonEP's extra kernels**: MoonEP adds planning and weight-prefetch kernels that DeepEP does not need, and they are already stacked in the bars above. Even with the whole critical path included, total dispatch time is on par with DeepEP v2's dispatch alone and pulls ahead under imbalance, while combine is significantly faster at every level.

**End-to-end training**:

<img src="figure/e2e_vs_deepep.png" alt="MoonEP vs DeepEP e2e training" width="800">

- **DeepEP degrades with imbalance**: the hottest ranks receive more tokens, so iteration time climbs steadily as maxvio grows; meanwhile the ever-changing activation shapes fragment GPU memory, until training OOMs at high imbalance.
- **MoonEP is unaffected**: every rank always computes exactly `S × K` tokens per layer, so iteration time stays flat at every imbalance level; fully static memory shapes mean no fragmentation, and training never OOMs.

## Supported Devices

- NVIDIA GPU
- Zhenwu PPU (under review, coming soon)

## Usage

### Integration

**Notation**: `S` = input tokens per rank, `K` = routed top-k per token, `E` = total routed experts in the EP group, `R` = number of EP ranks (EP comm size), `B` = weight prefetch slots per rank, `NvS` = dispatched token slots per rank (`S × K` real tokens plus per-VM-group padding), `H` = hidden size, `H'` = expert FFN intermediate size.

MoonEP's contract with a training or inference framework is **one contiguous symmetric-memory weight tensor per expert projection, plus a planner-produced `cu_seqlens`**. The VM group GEMM consumes a single `[E+B, H, H']` weight tensor; `cu_seqlens[E+B]` (returned by `dispatch`) selects which expert rows are active for the current step.

#### Weight buffer

<img src="figure/weight_buffer.png" alt="MoonEP weight buffer layout" width="1000">

For each expert projection (gate/up/down), every layer holds **one contiguous VMM range** `[E+B, H, H']`, identically laid out on every rank. Contiguity is a hard requirement: the group GEMM addresses experts purely by row index.

- **Rows `[0, E)`: all ranks' local experts** — `E/R` rows per rank. Each chunk physically *is* the home rank's parameter memory, mapped everywhere via symmetric memory.
- **Rows `[E, E+B)`: local prefetch slots**, filled by `buffer.prefetch_weight`; the planner points duplicated experts' token segments at these slots via `cu_seqlens`. Their physical memory comes from a process-global pool shared by all layers, so the extra cost is `B` expert weights per projection in total, not per layer.

**How to set B.**

- **Training**: must use **`B = E/R`** — the planner duplicates experts from at most one remote home group per rank (≤ `E/R` experts), so every expert the group GEMM touches is local.
- **Inference** (prefetch only, no gradients): `B < E/R` is allowed, **`B = 3–4` is recommended**. If a rank ever needs more distinct remote experts than `B`, the group GEMM reads the overflow weights straight from the home rank through the symmetric mapping (memory semantics, addressed by `cu_seqlens`) — slightly slower, with no impact on correctness.

#### Gradient buffers (training only)

<img src="figure/grad_buffer.png" alt="MoonEP grad buffer and grad reduce" width="1000">

Training mirrors the weight layout in fp32: one contiguous `[E+B, H, H']` **grad buffer** per projection.

- **Rows `[0, E)`**: the owner ranks' parameter grads.
- **Rows `[E, E+B)`**: prefetch-slot grads, backed by a **separate reduce buffer, not by the parameter grads** — duplicated experts' grads are temporary and must stay invisible to the framework's own grad reduce. The physical memory comes from a process-global pool shared across layers, like the prefetch pool.
- **Reduce buffer**: every rank maps all `R` reduce buffers as one `[R, B, H, H']` view. `reduce_grad` lets each rank read the slots holding its own experts' grads from every rank's reduce buffer (remote reads over NVLink), accumulate them into its local parameter grad, then zero its own consumed slots for the next microbatch.

### API walkthrough

```python
from moonep import Buffer

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8,
                num_sms=32, token_padding=128)
```

- `num_sms=None` defaults to 32. `B` defaults to `E // num_ep_ranks`; an explicit value like `B=4` may also be passed.
- `dispatch` / `combine` / `prefetch_weight` / `reduce_grad` all accept `async_finish=True` to run on the comm stream and return a CUDA event.

#### dispatch fwd

```python
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh,          # [S, H] bf16
    route_weights_sk,   # [S, K] fp32
    topk_experts_sk,    # [S, K] int32
    tokens_per_expert,  # [E] int32, local count
)
# hidden_nvsh:       [NvS, H] bf16 — dispatched tokens in physical VM group order
# route_weights_nvs: [NvS] fp32
# cu_seqlens:        [E+B] int32 — padded token end offset per VM group row
# plan:              MoonEPCommPlan — save it for prefetch/combine and both backward passes

buffer.prefetch_weight(
    plan=plan,
    full_gate_weight=full_gate_weight,    # [E+B, H, H'] bf16
    full_up_weight=full_up_weight,        # [E+B, H, H'] bf16
    full_down_weight=full_down_weight,    # [E+B, H, H'] bf16
)
# full_*_weight: rows [0, E) are source expert weights, rows [E, E+B) are prefetch slots
```

#### dispatch bwd

Backward of dispatch: sum each token's K dispatched grad copies back to token-major — a combine — and reduce duplicated experts' weight grads back to their home ranks.

```python
grad_hidden_sh, _, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=grad_hidden_nvsh,    # [NvS, H] bf16
)
# grad_hidden_sh: [S, H] bf16

buffer.reduce_grad(
    plan=plan,
    full_gate_grad=full_gate_grad,          # [E+B, H, H'] fp32
    full_up_grad=full_up_grad,              # [E+B, H, H'] fp32
    full_down_grad=full_down_grad,          # [E+B, H, H'] fp32
    gate_reduce_buffer=gate_reduce_buffer,  # [R, B, H, H'] fp32
    up_reduce_buffer=up_reduce_buffer,      # [R, B, H, H'] fp32
    down_reduce_buffer=down_reduce_buffer,  # [R, B, H, H'] fp32
)
# full_*_grad: same [E+B] layout as the weights; rows [E, E+B) are backed by the reduce buffer
# *_reduce_buffer: all R ranks' reduce buffers mapped as one [R, B, H, H'] view
```

#### combine fwd

```python
output_sh, gathered_route_weights_sk, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=expert_output_nvsh,       # [NvS, H] bf16
    route_weights_nvs=route_weights_nvs,  # [NvS] fp32, optional
)
# output_sh:                 [S, H] bf16 — combined token-major output
# gathered_route_weights_sk: [S, K] fp32 or None — routing weights gathered back to token-major
```

#### combine bwd

Backward of combine: scatter the output grad back to VM group order by re-dispatching with the saved plan — planning is skipped and no prefetch is needed.

```python
grad_expert_output_nvsh, _, _, _ = buffer.dispatch(
    grad_output_sh,    # [S, H] bf16
    plan=plan,
)
# grad_expert_output_nvsh: [NvS, H] bf16
```

#### zero_copy

By default `dispatch` returns fresh tensors and `combine` first copies its inputs into the NVL shard. With `zero_copy=True` on both sides, `dispatch` returns views of the communication buffer and the expert FFN reads/writes them in place — no boundary copy at all:

```python
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
    zero_copy=True,
)
# hidden_nvsh / route_weights_nvs are views of the NVL buffer;
# the expert FFN must write its output in place on hidden_nvsh
output_sh, gathered_route_weights_sk, _ = buffer.combine(
    plan=plan,
    hidden_nvsh=hidden_nvsh,
    route_weights_nvs=route_weights_nvs,
    zero_copy=True,  # asserts the inputs are exactly the views from dispatch
)
```

- The views alias buffer state that the next `dispatch` / `combine` overwrites — do not hold them across communication calls (autograd must not save them for backward; that case requires `zero_copy=False`).

```python
# explicitly release VMM/NVLink resources held by the Buffer before destroying the process group
buffer.destroy()
```

## Build & Test

```bash
pip install -e .

# run tests (requires multiple GPUs + NVLink)
torchrun --nproc_per_node=8 -m pytest tests/test_planning.py
torchrun --nproc_per_node=8 -m pytest tests/test_dispatch.py
torchrun --nproc_per_node=8 -m pytest tests/test_combine.py
torchrun --nproc_per_node=8 -m pytest tests/test_e2e.py
torchrun --nproc_per_node=8 -m pytest tests/test_grad_reduce.py
torchrun --nproc_per_node=8 -m pytest tests/test_prefetch.py
```

## Acknowledgments

This library is inspired by the following works:

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [Echo](https://arxiv.org/abs/2603.07685)
- [UltraEP](https://github.com/Dots-Infra/UltraEP)
- AcclEP (Alibaba's EP communication library)

## Citation

```bibtex
@misc{moonep2026,
      title={MoonEP: A Perfectly Balanced Expert Parallelism Library via Dynamic Redundant Experts},
      author={Yutian Chen, Cong Li, Yucheng Wang, Ming Wei},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/MoonEP}},
}
```
