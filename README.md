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

**Notation**: `S` = input tokens per rank, `K` = routed top-k per token, `E` = total routed experts in the EP group, `R` = number of EP ranks (EP comm size), `epn = E/R` = local experts and prefetch/reduce slots per rank, `NvS` = dispatched token slots per rank (`S × K` real tokens plus per-VM-group padding), `H` = hidden size, `H'` = expert FFN intermediate size.

MoonEP's communication API receives each projection as a local expert tensor `[epn, H, H']` plus an all-rank prefetch-buffer view `[R, epn, H, H']`. The integrating framework exposes this rank's local experts followed by its local prefetch slice as one contiguous `[2*epn, H, H']` compute view for the VM group GEMM; the planner-produced `cu_seqlens[2*epn]` selects the active rows.

#### Weight buffer

<img src="figure/weight_buffer.png" alt="MoonEP weight buffer layout" width="1000">

For each expert projection (gate/up/down), the framework builds two related views:

- **Communication view `[R, epn, H, H']`**: all ranks' prefetch pools mapped directly for `buffer.prefetch_weight`.
- **Compute view `[2*epn, H, H']`**: rows `[0, epn)` alias this rank's local parameter weights; rows `[epn, 2*epn)` alias this rank's prefetch slice. The VM group GEMM and `cu_seqlens` use this compact row order without mapping the other ranks' parameter weights.

Each rank's prefetch pool is process-global and shared by all layers, so the extra physical cost is `epn` expert weights per projection in total, not per layer.

Each rank has `epn` prefetch slots. The planner moves experts from at most one remote home group to each destination rank, so these slots cover every remote expert segment.

#### Gradient buffers (training only)

<img src="figure/grad_buffer.png" alt="MoonEP grad buffer and grad reduce" width="1000">

Training mirrors the compact weight layout in fp32:

- **Local grad `[epn, H, H']`**: this rank's parameter grads.
- **Compute grad view `[2*epn, H, H']`**: the local grad followed by this rank's reduce-buffer slice. The tail contains temporary prefetch-slot grads and stays separate from the framework's own parameter-grad reduction.
- **Reduce buffer**: every rank maps all `R` reduce buffers as one `[R, epn, H, H']` view. `reduce_grad` lets each rank read the slots holding its own experts' grads from every rank's reduce buffer (remote reads over NVLink), accumulate them into its local parameter grad, then zero its own consumed slots for the next microbatch.

### API walkthrough

```python
from moonep import Buffer

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8,
                num_sms=32, token_padding=128)
```

- `num_sms=None` defaults to 32. The current implementation derives `epn = E // num_ep_ranks` internally.
- `dispatch` / `combine` / `prefetch_weight` / `reduce_grad` all accept `async_finish=True` to run on the comm stream and return a CUDA event.
- `combine` defaults to `inter_rank_sync=True`, which runs an explicit rank sync before staging. Pass `inter_rank_sync=False` to skip this pre-staging sync; the combine kernel still performs its own entry cross-rank barrier.

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
# cu_seqlens:        [2*epn] int32 — padded token end offset per VM group row
# plan:              MoonEPCommPlan — save it for prefetch/combine and both backward passes

buffer.prefetch_weight(
    plan=plan,
    local_gate_weight=local_gate_weight,        # [epn, H, H'] bf16
    local_up_weight=local_up_weight,            # [epn, H, H'] bf16
    local_down_weight=local_down_weight,        # [epn, H, H'] bf16
    gate_prefetch_buffer=gate_prefetch_buffer,  # [R, epn, H, H'] bf16
    up_prefetch_buffer=up_prefetch_buffer,      # [R, epn, H, H'] bf16
    down_prefetch_buffer=down_prefetch_buffer,  # [R, epn, H, H'] bf16
)
# The framework's GEMM view aliases local_*_weight followed by this rank's
# *_prefetch_buffer slice as [2*epn, H, H'].
```

#### FP8 activation dispatch

Like [DeepEP](https://github.com/deepseek-ai/DeepEP), `dispatch` accepts a
`(data, scales)` tuple for DeepSeek tile-wise FP8 activations:

- `data`: contiguous `[S, H]` `torch.float8_e4m3fn` (E4M3).
- `scales`: `[S, H // 128]` `torch.float32`, one **dequantization** scale per
  **1 × 128** activation tile; `H` must be divisible by 128. Row-major and
  column-major input scales are accepted. Contiguous, 16-byte aligned scales
  with `H` divisible by 512 (e.g. 3584, 7168) are the fast path: scale rows
  are loaded by TMA together with their payload rows. Other layouts fall back
  to per-element scale loads.
- Reconstruction: `data.float().view(S, H // 128, 128) * scales.unsqueeze(-1)`.

```python
# Quantize before communication; this is a PyTorch reference implementation.
# A training/inference framework can supply its own fused quantizer instead.
tiles = hidden_sh.float().view(S, H // 128, 128)
amax = tiles.abs().amax(dim=-1).clamp_min(1e-4)
scales = (amax / 448.0).contiguous()
data = (tiles * (448.0 / amax).unsqueeze(-1)).to(torch.float8_e4m3fn).view(S, H)

(recv_data, recv_scales), route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
    (data, scales), route_weights_sk, topk_experts_sk, tokens_per_expert,
    zero_copy=True,
)
# recv_data:   [NvS, H]        torch.float8_e4m3fn, contiguous
# recv_scales: [NvS, H // 128] torch.float32, contiguous
# Both follow the same expert-grouped layout; padding data and scales are zero.

# Expert computation consumes recv_data/recv_scales and produces BF16 output.
# Finish reading both FP8 inputs before writing buffer.hidden_nvsh_buffer_view;
# that BF16 output view aliases their storage.
output_sh, _, _ = buffer.combine(plan=plan, hidden_nvsh=expert_output_nvsh)
```

FP8 data and scales are transferred together by the dispatch kernel and
expanded together for duplicate tokens. `plan` reuse, `async_finish`, and
`router_weights_zero_copy` retain their existing contracts. All ranks must
use the same activation format for a communication call. Combine continues
to accept BF16 expert outputs or gradients.

`zero_copy=True` returns views for **both** FP8 data and scales. They share
the existing BF16 communication allocation, which the next dispatch or
combine overwrites. Use `zero_copy=False` when either output must survive
another communication call. The persistent communication allocation does
not grow; it retains BF16 capacity for combine and backward dispatch.

Each 128-element activation tile carries 128 bytes of FP8 data plus 4 bytes
of scale, compared with 256 bytes in BF16: **48.44% less activation traffic**
(or **1.94×** compression). This counts scales and excludes route weights;
latency gains depend on workload and hardware.

[benchmarks/bench_dispatch_fp8.py](benchmarks/bench_dispatch_fp8.py) compares
BF16 and FP8 on identical inputs and routing, including scale traffic,
planning in `fresh` mode, duplicate expansion, and output copies where enabled.
Inputs are prequantized, so quantization, GEMM, prefetch and combine are outside
the timed region. CUDA graph timing reports the median of samples using the
slowest rank per replay, with alternating BF16/FP8 measurement order.

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 benchmarks/bench_dispatch_fp8.py \
    --tokens 256 2048 8192 --hidden 3584 7168 --bias 0 1 5 --out fp8_dispatch.csv
```

The CSV records the loaded MoonEP checkout's HEAD commit, a SHA-256 of its
actual Python sources and compiled extension (including local edits), a
benchmark-script SHA-256, and timing settings.
`speedup_vs_run_bf16` means **BF16 latency / measured-format latency in the same
run**; it is empty for an FP8-only run.

For an upstream comparison, use the latest official `master` at a recorded
commit. Copy this benchmark, `tests/quantization_reference.py`, and
`tests/generate_topk_routing.py` into that checkout, build it, and run the same
command with `--dtypes bf16` and a separate output path. Match rows by workload
and settings, then report **upstream BF16 / branch FP8** for the feature gain
and **branch BF16 / upstream BF16 - 1** for BF16 latency regression. The
benchmark measures the loaded checkout; it does not switch branches.

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
    local_gate_grad=local_gate_grad,          # [epn, H, H'] fp32
    local_up_grad=local_up_grad,              # [epn, H, H'] fp32
    local_down_grad=local_down_grad,          # [epn, H, H'] fp32
    gate_reduce_buffer=gate_reduce_buffer,  # [R, epn, H, H'] fp32
    up_reduce_buffer=up_reduce_buffer,      # [R, epn, H, H'] fp32
    down_reduce_buffer=down_reduce_buffer,  # [R, epn, H, H'] fp32
)
# The framework's GEMM grad view aliases local_*_grad followed by this rank's
# *_reduce_buffer slice as [2*epn, H, H'].
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
torchrun --nproc_per_node=8 -m pytest tests/test_dispatch_fp8.py
torchrun --nproc_per_node=8 -m pytest tests/test_combine.py
torchrun --nproc_per_node=8 -m pytest tests/test_e2e.py
torchrun --nproc_per_node=8 -m pytest tests/test_grad_reduce.py
torchrun --nproc_per_node=8 -m pytest tests/test_prefetch.py
```

FP8 tests can also run on two NVLink-connected GPUs for a quicker check:

```bash
torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_dispatch_fp8.py
```

`torchrun` starts one process per GPU. Every process runs the same cases in
the same order so the distributed collectives can cooperate. Plain `pytest`
skips these tests without a distributed process group. Use eight GPUs to
validate the full EP=8 topology, including the existing BF16 paths:

```bash
torchrun --standalone --nproc_per_node=8 -m pytest -q \
    tests/test_dispatch_fp8.py tests/test_dispatch.py tests/test_combine.py
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
