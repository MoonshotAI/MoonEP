# SPDX-License-Identifier: Apache-2.0
"""Shared kernel-test helpers (Ascend port of ``MoonEP/tests/kernel_test_utils.py``).

Everything lives in a single module so distributed tests share one set of
collective assertions (``assert_all_ranks`` / ``assert_tensor_equal_all_ranks`` /
``assert_close_all_ranks`` / ``assert_ulp_all_ranks``), the ``KernelCase``
configuration dataclass, and the Ascend Buffer runtime lifecycle
(``ShmemRuntime`` init sized from the case, buffer tracked for teardown).

Differences from the CUDA original:
  - device is ``"npu"`` and the process group backend is ``"hccl"``.
  - ``Buffer`` construction no longer requires shmem: it registers CANN VMM
    descriptors, then ``init_case`` calls ``init_shmem`` to bootstrap the
    buffer-backed symmetric heap. ``init_shmem`` auto-selects the bootstrap:
      * ``ShmemRuntime.init_with_buffer`` (UID, single node) -- single node (``nnode=1`` or
        unspecified, i.e. ``WORLD_SIZE == LOCAL_WORLD_SIZE``, or the script did not pass
        ``--nproc-per-node``) or no master addr/port given;
      * ``ShmemRuntime.init_with_ip_port`` (needs dist / config-store) -- multi node with the
        master addr/port given (or the caller explicitly passes ``ip_port`` / ``--bootstrap ipport``).
    ``destroy_active_buffers`` (and the unified ``finalize_shmem_and_release``) release in a fixed
    order: first ``ShmemRuntime.finalize()`` (drops the symmetric mapping), then destroy Buffer / VMM
    allocations (the reverse op), then clear the descriptor registry so the next case can re-init.
  - the CUDA-VMM layout invariants (``PLAN_OFF``/``TOPK0_OFF``/``BARRIER_OFF``,
    token-padding divisibility) do not exist on Ascend and are dropped; the
    ``dst`` / ``cu_seqlens`` / ``experts_to_copy`` range checks and the dedup
    plan invariants are kept.
"""

import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

import torch_npu

from generate_topk_routing import generate_topk_routing

DEFAULT_TOKEN_PADDING = 128


def shmem_ip_port(ip_port=None, args=None) -> str:
    """Derive the shmem DEFAULT-bootstrap ip_port for ``init_with_ip_port``.

    Only consulted when a caller explicitly wants the ip_port/config-store
    bootstrap (multi-node). Precedence:

      * explicit ``ip_port`` (e.g. a test that already computed one)
      * ``args.shmem_ip_port`` / args-derived ``master_addr:master_port+1``
        (script launches that pass ``--master-addr``/``--master-port``)
      * torchrun env ``MASTER_ADDR``/``MASTER_PORT`` (+1)  (multi-node torchrun)
      * loopback default ``tcp://127.0.0.1:29501``

    ``ip_port`` is only derived here when ``init_shmem`` has decided to use the
    ip_port bootstrap (multi node with master specified); the single-node path always uses
    ``ShmemRuntime.init_with_buffer`` and never calls this function.
    """
    if ip_port:
        return ip_port
    if args is not None:
        explicit = getattr(args, "shmem_ip_port", None)
        if explicit:
            return explicit
        addr = getattr(args, "master_addr", None)
        port = getattr(args, "master_port", None)
        if addr and port:
            from npu_patch.shmem_runtime import default_shmem_ip_port
            return default_shmem_ip_port(addr, port)
    from npu_patch.shmem_runtime import default_shmem_ip_port
    addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29500"))
    return default_shmem_ip_port(addr, port)


def _master_specified(args=None) -> bool:
    """True means master addr/port were given explicitly (args or torchrun env vars)."""
    if args is not None:
        addr = getattr(args, "master_addr", None)
        port = getattr(args, "master_port", None)
        if addr and port:
            return True
    return bool(os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"))


def _should_use_ip_port_bootstrap(ip_port=None, args=None) -> bool:
    """Decide ``ShmemRuntime`` bootstrap: ip_port vs buffer (UID).

    Rules (single node or unspecified master -> buffer, otherwise -> ip_port):

      * explicit ``ip_port`` (e.g. ``--shmem-ip-port``) -> ip_port;
      * script (``args``):
          - ``args.bootstrap == 'ipport'`` -> ip_port;
          - ``args.bootstrap == 'uid'``    -> buffer (UID);
          - otherwise decide by node count: ``--nproc-per-node`` passed and ``R > nproc``
            (multi node) and master specified -> ip_port; otherwise (single node) -> buffer;
      * pytest / torchrun (``args is None``): multi node
        (``WORLD_SIZE > LOCAL_WORLD_SIZE``) and master specified -> ip_port;
        otherwise -> buffer.
    """
    if ip_port:
        return True
    if args is not None:
        bootstrap = getattr(args, "bootstrap", None)
        if bootstrap == "ipport":
            return True
        if bootstrap == "uid":
            return False
        nproc = getattr(args, "nproc_per_node", None)
        R = getattr(args, "R", None)
        multi_node = (nproc is not None and R is not None and int(R) > int(nproc))
        if multi_node:
            return _master_specified(args)
        return False
    world = os.environ.get("WORLD_SIZE")
    local = os.environ.get("LOCAL_WORLD_SIZE")
    multi_node = bool(world and local and int(world) > int(local))
    if not multi_node:
        return False
    return _master_specified(args=None)


def init_shmem(group=None, enable_udma=None, qp_count=0, ip_port=None, args=None, **kwargs):
    """Bootstrap the buffer-backed shmem heap, auto-selecting the bootstrap.

    Must run AFTER every Buffer / VMM chunk is constructed and BEFORE the first
    dispatch/combine/prefetch/reduce_grad. Idempotent (skips if already
    initialized).

    Bootstrap selection (see ``_should_use_ip_port_bootstrap``):

      * single node (``nnode==1`` or unspecified) or no master addr/port given
        -> ``ShmemRuntime.init_with_buffer`` (UID, broadcast over dist, no config-store needed);
      * multi node with master addr/port given (or explicitly passing ``ip_port`` /
        ``--bootstrap ipport``)
        -> ``ShmemRuntime.init_with_ip_port`` (config-store / DEFAULT bootstrap).

    The buffer-backed symmetric heap is built from the VMM descriptors registered
    by ``Buffer`` / ``create_vmm_physical`` (passed explicitly so the call never
    silently uses an empty descriptor set).
    """
    from npu_patch import ShmemRuntime
    from npu_patch.buffer_c import symmetric_descriptors
    if ShmemRuntime.is_initialized():
        return
    descriptors = symmetric_descriptors()
    if _should_use_ip_port_bootstrap(ip_port=ip_port, args=args):
        if ip_port is None:
            ip_port = shmem_ip_port(ip_port=None, args=args)
        ShmemRuntime.init_with_ip_port(
            group=group,
            ip_port=ip_port,
            descriptors=descriptors,
            enable_udma=enable_udma,
            qp_count=qp_count,
            **kwargs,
        )
    else:
        ShmemRuntime.init_with_buffer(
            group=group,
            descriptors=descriptors,
            enable_udma=enable_udma,
            qp_count=qp_count,
            **kwargs,
        )


_ACTIVE_BUFFERS = []


def finalize_shmem_and_release(source_tensors=(), buffers=(), reset_registry=True):
    """Unified buffer-backed shmem teardown with a single, fixed release order.

    Order (must not be reversed):

      1. ``ShmemRuntime.finalize()`` -- tear down the symmetric heap / symmetric mapping first. If the
         backing VMM physical memory were freed or reused first, shmem could still hold mappings to
         those addresses, causing UAF or a wrong mapping;
      2. destroy the VMM allocations (``_vmm_allocation.destroy()``), then the ``Buffer`` -- the
         caller owns this CANN physical memory and may only release it after the symmetric mapping is dead;
      3. ``reset_symmetric_descriptors()`` -- clear the process-level descriptor registry so
         the next case can re-init with a new Buffer (shmem init is once per process).

    ``source_tensors`` are the VMM chunks obtained from ``create_vmm_physical`` (tracked manually by
    script-style tests); ``buffers`` are ``Buffer`` objects. Several chunks/projection tensors may share
    one VMM allocation, so this is idempotent (finalize is idempotent, each alloc is destroyed once, an
    already destroyed Buffer is skipped).
    """
    from npu_patch import ShmemRuntime
    from npu_patch.buffer_c import reset_symmetric_descriptors

    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()
    destroyed_allocs = set()
    for t in source_tensors:
        alloc = getattr(t, "_vmm_allocation", None)
        if alloc is None or id(alloc) in destroyed_allocs:
            continue
        destroyed_allocs.add(id(alloc))
        try:
            alloc.destroy()
        except Exception:  # best-effort teardown
            pass
    for buf in buffers:
        if not buf.destroyed:
            buf.destroy()
    if reset_registry:
        reset_symmetric_descriptors()


def local_device_index() -> int:
    from npu_patch._device import local_device_index as _resolve

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    return _resolve(local_rank)


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


def init_case(case, R, use_udma: bool = True, ip_port=None, args=None, group=None):
    """Construct the case's Buffer and bootstrap the buffer-backed shmem heap.

    ``group`` (optional) scopes both to an EP **subgroup** instead of the whole
    world: the Buffer takes ``rank = dist.get_rank(group)`` and the shmem domain
    is bootstrapped over that subgroup only (the UID is broadcast within it, see
    ``ShmemRuntime.init_with_buffer``).  ``R`` must then equal
    ``dist.get_world_size(group)``, i.e. the *domain* EP size.  Every rank of a
    given subgroup must pass the same subgroup object; see
    ``test_npu/buffer_test/test_buffer_grouped_e2e.py`` for the two-domain layout.

    Note the device must still be selected from the *global* ``LOCAL_RANK``
    (``local_device_index()``) -- never from the subgroup rank, which would make
    two subgroups collide on the same cards.
    """
    from npu_patch import Buffer, ShmemRuntime

    skip_if_unsupported_world_size(case, R)
    E = case.E(R)
    # buffer-backed only: Buffer construction does not require shmem to be initialized (the
    # communication chunks are CANN VMM, registered into the global descriptor registry by
    # create_vmm_physical during construction); after construction run one
    # buffer-backed shmem init (init_shmem picks init_with_buffer / init_with_ip_port by single/multi node).
    buffer = Buffer(
        case.S,
        case.H,
        case.K,
        E,
        R,
        B=case.B,
        num_sms=case.num_sms,
        token_padding=case.token_padding,
        use_udma=use_udma,
        group=group,
    )
    if not ShmemRuntime.is_initialized():
        init_shmem(group=group, enable_udma=use_udma, ip_port=ip_port, args=args)
    ctx = buffer._require_ctx()
    ctx["_buffer"] = buffer
    _ACTIVE_BUFFERS.append(buffer)
    return ctx


def destroy_active_buffers():
    if not _ACTIVE_BUFFERS:
        # Nothing tracked: the test is a low-level/C++-path case that never
        # touched the Python Buffer runtime — stay out of its process.
        return
    # Finalize(then destroy) uses the unified release order: first finalize (tears down the symmetric
    # mapping), then destroy the Buffer (the reverse VMM op), so a re-init's get_buffer_ptr does not miss
    # the newly registered chunks.
    finalize_shmem_and_release(buffers=list(_ACTIVE_BUFFERS), reset_registry=True)
    _ACTIVE_BUFFERS.clear()


def skip_if_unsupported_world_size(case, R):
    if R < case.min_R:
        pytest.skip(f"case {case.name} requires R >= {case.min_R}, got R={R}")
    if case.max_R is not None and R > case.max_R:
        pytest.skip(f"case {case.name} requires R <= {case.max_R}, got R={R}")


def make_topk(case, rank, R):
    skip_if_unsupported_world_size(case, R)

    dev = "npu"
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
    ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device="npu")
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
