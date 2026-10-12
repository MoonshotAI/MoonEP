"""
Shared helpers for the MoonEP-Ascend operator benchmarks.

Mirrors ``MoonEP/benchmarks`` (CUDA) but targets the Ascend port
(``npu_patch``): torch_npu devices, HCCL process group, cann-shmem
symmetric memory, and the public ``Buffer`` API (``dispatch`` /
``prefetch_weight`` / ``combine`` / ``reduce_grad``).

Kernels are implemented; every flow runs a live Ascend kernel. ``measure_flow``
times a Buffer flow with npu events and returns a cross-rank mean latency;
``time_kernel`` is the same for a raw ``launch_*`` kernel (no Buffer flow
seam). Both return ``{status: "ok", us}``.
"""

import argparse
import math
import os
import sys

import torch
import torch_npu
import torch.distributed as dist

# Make ``npu_patch`` (repo root) and the routing generator (test/) importable
# regardless of the cwd torchrun is launched from.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TEST = os.path.join(_ROOT, "test_npu")
for _p in (_ROOT, _TEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_topk_routing import generate_topk_routing

from npu_patch._device import local_device_index

# Marker the op stubs embed in the NotImplementedError they raise at the seam.
KERNEL_NOT_IMPLEMENTED = "kernel not implemented"

# Production 8-card EP shape (user-specified): S=4096, K=16, E=896, H=7168,
# H'=3072 (R=8, B=E//R=112). Default sweeps stay native-identical for direct
# comparison with MoonEP; these constants drive the ``--prod`` shortcuts.
PROD_S = 4096
PROD_K = 16
PROD_E = 896
PROD_H = 7168
PROD_HP = 3072
PROD_B = 112


def setup():
    """torchrun-style distributed init over HCCL (Ascend)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = local_device_index(local_rank)
    torch.npu.set_device(device)
    dist.init_process_group(
        backend="hccl", device_id=torch.device(f"npu:{device}")
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank


def fmt4(x: float) -> str:
    """Format float to exactly 4 significant digits (reference ``fmt4``)."""
    if x != x:  # NaN (stub flows)
        return "   n/a"
    if x == 0:
        return "0.000"
    d = math.floor(math.log10(abs(x))) + 1  # digits before the decimal point
    return f"{x:.{max(0, 4 - d)}f}"


def time_kernel(fn, group, rank, *, warmup: int = 5, iters: int = 20,
                agg: str = "mean", cudagraph: bool = True):
    """Time a raw kernel launch (no Buffer-flow seam). Return ``{status, us}``.

    ``warmup`` + ``iters`` launches timed with npu events on the current stream;
    cross-rank ``agg`` = ``mean`` (all ranks equal work) or ``max`` (worst rank,
    for rank-initiated prefetch / grad-reduce).

    ``cudagraph=True`` (default) captures ``iters`` launches into a
    ``torch.npu.CUDAGraph`` and times the replay, removing per-launch host
    overhead (mirrors ``MoonEP/benchmarks/bench_comm.py::time_gpu_op``). If the
    flow is not capturable (e.g. a host-side cross-rank sync inside), it falls
    back to the eager loop with a rank-0 note.
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    dist.barrier(group=group)

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    captured = False
    if cudagraph and hasattr(torch.npu, 'CUDAGraph') and hasattr(torch.npu, 'graph'):
        try:
            g = torch.npu.CUDAGraph()
            with torch.npu.graph(g):
                for _ in range(iters):
                    fn()
            torch.npu.synchronize()
            dist.barrier(group=group)
            start.record()
            g.replay()
            end.record()
            captured = True
        except Exception:
            if rank == 0:
                print('  (note: flow not CUDA-graph capturable; eager timing)',
                      flush=True)
            torch.npu.synchronize()
            dist.barrier(group=group)
    if not captured:
        start.record()
        for _ in range(iters):
            fn()
        end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3  # ms -> us

    dev = torch.device(f"npu:{torch.npu.current_device()}")
    ws = dist.get_world_size(group=group)
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(ws)]
    dist.all_gather(outs, t, group=group)
    cat = torch.cat(outs)
    us = cat.max().item() if agg == "max" else cat.mean().item()
    dist.barrier(group=group)
    return {"status": "ok", "us": us}


def measure_flow(
    label: str,
    expected_op: str,
    fn,
    group,
    rank: int,
    *,
    allow_stub: bool = True,
    warmup: int = 5,
    iters: int = 20,
    agg: str = "mean",
    cudagraph: bool = True,
):
    """Run a Buffer flow; return ``{status, op, us}``.

    One probe call classifies the flow (kept stub-aware for safety):
      - ``NotImplementedError`` naming ``expected_op.compute`` and carrying the
        stub marker -> pre-launch plumbing reached the op seam. With
        ``allow_stub`` the flow is reported ``stub`` (``us`` = NaN); without it
        the exception propagates (fail).
      - any other error is a plumbing failure -> AssertionError (fail).
      - the flow completes (live kernel) -> ``time_kernel`` cross-rank
        ``agg`` = ``mean`` (all ranks equal work) or ``max``.
    """
    try:
        fn()
    except NotImplementedError as exc:
        msg = str(exc)
        assert f"{expected_op}.compute:" in msg, (
            f"rank {rank}: {label} reached a DIFFERENT op than {expected_op}:\n{msg}"
        )
        assert KERNEL_NOT_IMPLEMENTED in msg, (
            f"rank {rank}: {label} NotImplementedError outside the op seam:\n{msg}"
        )
        torch.npu.synchronize()
        dist.barrier(group=group)
        if not allow_stub:
            raise
        return {"status": "stub", "op": expected_op, "us": float("nan")}
    except Exception as exc:  # surface any plumbing failure
        raise AssertionError(
            f"rank {rank}: {label} errored before/outside the op seam: {exc!r}"
        ) from exc

    us = time_kernel(fn, group, rank, warmup=warmup, iters=iters, agg=agg,
                     cudagraph=cudagraph)["us"]
    return {"status": "ok", "op": expected_op, "us": us}


def time_text(res: dict) -> str:
    """Render one flow result: ``<us>`` or ``stub``."""
    if res["status"] == "stub":
        return "  stub"
    return fmt4(res["us"])


# ---------------------------------------------------------------------------
# Transport engine: ubmem (native MTE/UB path) vs urma (UDMA/URMA path)
# ---------------------------------------------------------------------------
# Every benchmark can target either transport of the same operator set:
#   ubmem (default) - native MTE/UB path, available on every SoC.
#   urma            - UDMA/URMA path, A5 (Ascend950/SIMT) only.
# Buffer-backed benches pass ``use_udma=`` to ``Buffer``; benches driving the
# raw ``launch_*`` entry points (no Buffer; launchers read the engine from the
# environment) call ``apply_engine``, which publishes the
# ``ASCEND_MOONEP_*_USE_UDMA`` policy **through the library's own capability
# gate** (``npu_patch.api._is_ascend950``) - requesting urma on A2/A3 degrades
# to ubmem with a note instead of bypassing ``udma_supported``. run_test.sh
# enforces the same rule (``--use_urma`` rejected on non-A5).
ENGINES = ("ubmem", "urma")
_UDMA_OPS = ("planning", "dispatch", "combine", "prefetch", "grad_reduce")


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--engine`` switch shared by every benchmark."""
    parser.add_argument(
        "--engine", choices=ENGINES, default="ubmem",
        help="Transport engine: ubmem = MTE/UB native path (default), "
             "urma = UDMA/URMA path (A5/Ascend950 only).")


def _urma_supported() -> bool:
    """UDMA capability probe reusing the library's own check (api._is_ascend950)."""
    from npu_patch.api import _is_ascend950

    device = int(torch.npu.current_device())
    return bool(_is_ascend950(device, torch.npu.get_device_properties(device)))


def apply_engine(engine: str) -> bool:
    """Publish the engine choice as the ``ASCEND_MOONEP_*_USE_UDMA`` policy.

    Call it after ``setup()``: the udma probe reads the current device. Returns
    the **effective** flag (True = urma), which the callers thread into
    ``Buffer`` / ``ShmemRuntime.init_*`` so the operator choice and the symmetric
    heap agree.
    """
    want = engine == "urma"
    if want:
        try:
            supported = _urma_supported()
        except Exception as exc:  # probe failure must not pick urma
            supported = False
            print(f"[bench] udma capability probe failed ({exc}).", flush=True)
        if not supported:
            print("[bench] --engine urma requested, but this device is not "
                  "Ascend950 (no SIMT/UDMA): falling back to ubmem.", flush=True)
            want = False
    value = "1" if want else "0"
    os.environ["ASCEND_MOONEP_USE_UDMA"] = value
    for name in _UDMA_OPS:
        os.environ[f"ASCEND_MOONEP_{name.upper()}_USE_UDMA"] = value
    return want


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by all benchmarks."""
    add_engine_args(parser)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per flow.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations per flow.")
    parser.add_argument("--num-sms", type=int, default=None,
                        help="SM count for the op launches (default: device max).")
    parser.add_argument("--ep-size", type=int, default=None,
                        help="EP group size (default: world size = launched ranks). "
                             "Must be <= world size; only ranks [0, ep) run the bench.")
    parser.add_argument("--no-allow-stub", dest="allow_stub", action="store_false",
                        help="Treat op-seam NotImplementedError as a failure instead "
                             "of reporting the flow as stub (use once kernels land).")
    parser.set_defaults(allow_stub=True)
    parser.add_argument("--no-cudagraph", dest="cudagraph", action="store_false",
                        help="Use the eager timing loop instead of CUDA-graph capture "
                             "(CUDA-graph removes per-launch host overhead).")
    parser.set_defaults(cudagraph=True)
    parser.add_argument("--out", default=None, help="Output CSV path (rank 0 only).")
