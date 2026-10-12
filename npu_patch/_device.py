# SPDX-License-Identifier: Apache-2.0
"""Central NPU device-index resolution (rank -> device).

Ascend semantics (CANN ``ASCEND_RT_VISIBLE_DEVICES``): the cards listed in the
mask are renumbered for the process to the *logical* indices ``0..N-1`` -- with
``ASCEND_RT_VISIBLE_DEVICES=2,3,4,5`` the tensor device ``npu:0`` addresses the
**physical** card 2, ``npu:1`` card 3, and so on.  Two id spaces therefore
coexist:

  - ``local_device_index`` (logical, renumbered 0..N-1): what
    ``torch.npu.set_device``, the ``npu:i`` tensor/stream device string, the
    ``dist`` ``device_id`` argument **and the VMM allocation APIs** expect
    (``aclrtMemGetAllocationGranularity`` / ``aclrtMallocPhysical`` /
    ``aclrtMapMem`` -- ``create_vmm_physical`` passes it as ``device``).
  - ``local_physical_id`` (the mask entry == npu-smi card id): what
    ``aclrtMemSetAccess``'s ``desc.location.id`` expects, and what logs / HCCL
    topology use.  ``create_vmm_physical`` / ``create_full_projection_tensor``
    pass it as the access id.

Measured on Ascend 950 with ``ASCEND_RT_VISIBLE_DEVICES=4,5``:
``aclrtGetDevice()`` returns ``0``; the allocation APIs accept ``0``/``1`` and
reject ``4``/``5`` (107001); ``aclrtMemSetAccess`` is the opposite -- ``0``/``1``
fail with 507899 while ``4``/``5`` succeed.  Both numbers are needed; mixing
them up is the classic failure here.  This module is the single source of truth
for rank -> device mapping; tests and benchmarks must not re-derive it.
"""

import os

import torch

__all__ = [
    "visible_npu_ids",
    "visible_npu_count",
    "local_device_index",
    "local_physical_id",
]


def visible_npu_ids():
    """Physical NPU ids visible to this process, or ``None`` if no mask is set.

    Parses ``ASCEND_RT_VISIBLE_DEVICES``, supporting comma-separated ids
    (``4,5``) and/or ``a-b`` intervals (``4-7`` == ``4,5,6,7``).  Returns
    ``None`` when the variable is absent/empty so callers fall back to the full
    device list; malformed tokens are ignored.
    """
    mask = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not mask:
        return None
    ids = []
    for token in mask.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, _, hi = token.partition("-")
            lo_s, hi_s = lo.strip(), hi.strip()
            if not (lo_s.isdigit() and hi_s.isdigit() and int(lo_s) <= int(hi_s)):
                continue
            ids.extend(range(int(lo_s), int(hi_s) + 1))
            continue
        if token.isdigit():
            ids.append(int(token))
    return ids or None


def visible_npu_count() -> int:
    """Number of NPU devices visible to this process.

    Uses the ``ASCEND_RT_VISIBLE_DEVICES`` mask when set, else
    ``torch.npu.device_count()``.
    """
    visible = visible_npu_ids()
    if visible is not None:
        return len(visible)
    if not torch.npu.is_available():
        return 0
    return int(torch.npu.device_count())


def _require_rank_in_range(local_rank: int, n_dev: int) -> int:
    if n_dev <= 0:
        raise RuntimeError("no Ascend NPU devices are visible to this process")
    if not 0 <= local_rank < n_dev:
        raise RuntimeError(
            f"local_rank={local_rank} out of range: only {n_dev} NPU device(s) "
            f"are visible (ASCEND_RT_VISIBLE_DEVICES="
            f"{os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')!r}). "
            f"Launch with nproc-per-node == the number of visible devices."
        )
    return local_rank


def local_device_index(local_rank: int) -> int:
    """Logical NPU device index for a torchrun-style local rank.

    The visible cards are renumbered ``0..N-1``, so the rank's logical device
    index is its position in the visible set == ``local_rank``.  Raises a clear
    error instead of an opaque "device not available" when more ranks than
    visible devices are requested (e.g. default ``NPROC=8`` with 4 visible
    cards).
    """
    local_rank = int(local_rank)
    return _require_rank_in_range(local_rank, visible_npu_count())


def local_physical_id(local_rank: int) -> int:
    """Physical card id behind ``local_rank`` (mask entry, npu-smi numbering).

    Needs two different consumers: ``aclrtMemSetAccess`` takes this id (see the
    module docstring), while the allocation APIs take the process-visible index
    (``local_device_index``).  Also handy for logs / HCCL topology / cross-node
    identification.  Returns ``local_rank`` unchanged when no mask is set (the
    two spaces coincide there).
    """
    local_rank = int(local_rank)
    visible = visible_npu_ids()
    if visible is None:
        return local_rank
    return visible[_require_rank_in_range(local_rank, len(visible))]
