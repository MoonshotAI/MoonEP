"""Object-centric execution views exposed by MoonEP."""

from dataclasses import dataclass

import torch

from .planning import MoonEPCommPlan


def _validate_tensor(
    name: str,
    tensor: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    return tensor


def _validate_plan_layout(
    plan: object,
    *,
    N: int,
    R: int,
    NvS: int,
    K: int,
    device: torch.device,
) -> MoonEPCommPlan:
    if not isinstance(plan, MoonEPCommPlan):
        raise TypeError("plan must be a MoonEPCommPlan")
    for name, expected in (("N", N), ("R", R), ("NvS", NvS), ("K", K)):
        actual = int(getattr(plan, name))
        if actual != expected:
            raise ValueError(
                f"plan.{name} must be {expected}, got {actual}"
            )
    E = int(plan.E)
    B = int(plan.B)
    if E <= 0 or B <= 0:
        raise ValueError(f"plan E and B must be positive, got E={E}, B={B}")
    if E % R != 0:
        raise ValueError(f"plan.E={E} must be divisible by R={R}")
    for name, tensor, shape in (
        ("plan.dst", plan.dst, (N,)),
        ("plan.experts_to_copy", plan.experts_to_copy, (R, B)),
        ("plan.zero_fill_ranges", plan.zero_fill_ranges, (E + B, 2)),
        ("plan.remote_stats", plan.remote_stats, (2,)),
        ("plan.dup_groups", plan.dup_groups, (NvS, 3)),
        ("plan.dup_loffs", plan.dup_loffs, (NvS,)),
        ("plan.dup_counts", plan.dup_counts, (2,)),
    ):
        _validate_tensor(
            name,
            tensor,
            dtype=torch.int32,
            shape=shape,
            device=device,
        )
    return plan


@dataclass(frozen=True, slots=True)
class MoonEPSourceStorage:
    """One physical source-activation shard per rank, globally mapped.

    Producers should write only ``local_hidden`` and routers should write
    ``local_route_weights``. The mapped tensors are exposed for kernels
    declaring an explicit peer-VMM load capability; generic tensor operations
    are not assumed safe on their remote rows.
    """

    mapped_hidden: torch.Tensor
    local_hidden: torch.Tensor
    mapped_route_weights: torch.Tensor
    local_route_weights: torch.Tensor
    buffer_token: int
    R: int
    S: int
    H: int
    K: int
    source_rows_padded: int
    source_topk_padded: int
    capability: str = "owner_sharded_source_v1"

    def validate_layout(self) -> None:
        if self.capability != "owner_sharded_source_v1":
            raise ValueError(
                "source storage requires capability owner_sharded_source_v1"
            )
        mapped_hidden = _validate_tensor(
            "mapped_hidden",
            self.mapped_hidden,
            dtype=torch.bfloat16,
            shape=(self.R * self.source_rows_padded, self.H),
        )
        _validate_tensor(
            "local_hidden",
            self.local_hidden,
            dtype=torch.bfloat16,
            shape=(self.S, self.H),
            device=mapped_hidden.device,
        )
        _validate_tensor(
            "mapped_route_weights",
            self.mapped_route_weights,
            dtype=torch.float32,
            shape=(self.R * self.source_topk_padded,),
            device=mapped_hidden.device,
        )
        _validate_tensor(
            "local_route_weights",
            self.local_route_weights,
            dtype=torch.float32,
            shape=(self.S, self.K),
            device=mapped_hidden.device,
        )


@dataclass(frozen=True, slots=True)
class MoonEPSourceView:
    """Planned peer-indexed view over an owner-sharded source object.

    ``slot_to_source[loff]`` maps each valid local expert-layout slot to
    ``source_rank * NvS + source_topk_offset``. A compatible consumer decodes:

    ``source_rank = encoded // NvS``
    ``source_token = (encoded % NvS) // K``
    ``source_row = source_rank * source_rows_padded + source_token``

    Negative entries are padding. The view is valid only until the same Buffer
    starts another dispatch, source preparation, direct-out preparation, or
    combine operation. This is the one-physical-copy activation view; no routed
    hidden rows have been materialized by its preparation.

    ``slot_to_primary[loff]`` names the representative local slot for the same
    source token on this consumer rank. It equals ``loff`` for representatives,
    points at the representative for same-rank top-k duplicates, and is
    negative for padding. Materializers may therefore pull one peer row per
    consumer rank and expand duplicates from local memory.

    ``slot_to_next`` links those slots in top-k order. A representative
    materializer can load each source vector once, then store the register
    value directly to every linked final slot without an intermediate local
    reread.

    The plan and metadata tensors are opaque runtime state and must not be
    modified by callers.
    """

    storage: MoonEPSourceStorage
    slot_to_source: torch.Tensor
    slot_to_primary: torch.Tensor
    slot_to_next: torch.Tensor
    plan: MoonEPCommPlan
    generation: int
    NvS: int
    K: int
    capability: str = "peer_indexed_source_v1"

    def validate_layout(self) -> None:
        self.storage.validate_layout()
        if self.capability != "peer_indexed_source_v1":
            raise ValueError(
                "source view requires capability peer_indexed_source_v1"
            )
        device = self.storage.mapped_hidden.device
        _validate_tensor(
            "slot_to_source",
            self.slot_to_source,
            dtype=torch.int32,
            shape=(self.NvS,),
            device=device,
        )
        _validate_tensor(
            "slot_to_primary",
            self.slot_to_primary,
            dtype=torch.int32,
            shape=(self.NvS,),
            device=device,
        )
        _validate_tensor(
            "slot_to_next",
            self.slot_to_next,
            dtype=torch.int32,
            shape=(self.NvS,),
            device=device,
        )
        _validate_plan_layout(
            self.plan,
            N=self.storage.S * self.K,
            R=self.storage.R,
            NvS=self.NvS,
            K=self.K,
            device=device,
        )


@dataclass(frozen=True, slots=True)
class MoonEPDispatchTarget:
    """Prepared owner-sharded activation object for producer direct-out.

    ``mapped_hidden`` is the full VMM mapping shaped
    ``[R * NvS_padded, H]``. A compatible producer kernel reads ``plan.dst``
    and writes its natural output row directly to every non-negative encoded
    destination:

    ``mapped_hidden[(dst // NvS) * NvS_padded + (dst % NvS)]``.

    Negative ``dst`` entries are same-rank duplicates and must not be written
    by the producer; ``Buffer.publish_dispatch`` expands them locally after
    publishing the representative rows.

    The target is intentionally capability-scoped. Generic tensor consumers
    are not assumed safe on peer VMM mappings. Publication is one-shot, and
    the target expires when the same Buffer starts another dispatch, prepare,
    or combine operation. The plan tensors are opaque runtime state and must
    not be modified by callers.
    """

    mapped_hidden: torch.Tensor
    local_hidden: torch.Tensor
    plan: MoonEPCommPlan
    generation: int
    buffer_token: int
    R: int
    S: int
    H: int
    K: int
    NvS: int
    NvS_padded: int
    capability: str = "remote_final_slot_store_v1"

    def validate_layout(self) -> None:
        if self.capability != "remote_final_slot_store_v1":
            raise ValueError(
                "dispatch target requires capability remote_final_slot_store_v1"
            )
        mapped_hidden = _validate_tensor(
            "mapped_hidden",
            self.mapped_hidden,
            dtype=torch.bfloat16,
            shape=(self.R * self.NvS_padded, self.H),
        )
        _validate_tensor(
            "local_hidden",
            self.local_hidden,
            dtype=torch.bfloat16,
            shape=(self.NvS, self.H),
            device=mapped_hidden.device,
        )
        _validate_plan_layout(
            self.plan,
            N=self.S * self.K,
            R=self.R,
            NvS=self.NvS,
            K=self.K,
            device=mapped_hidden.device,
        )


@dataclass(frozen=True, slots=True)
class MoonEPPreparedPublication:
    """Metadata publication scheduled ahead of producer completion.

    ``metadata_done`` records completion of route-weight scatter, padding
    initialization, dedup construction, and that kernel's own barrier. Those
    operations may overlap a compatible direct-out producer because they do
    not touch representative hidden payload rows. The producer's writes are
    not yet published: ``Buffer.publish_dispatch`` must consume this exact
    object and run a second visibility barrier before duplicate expansion.
    """

    target: MoonEPDispatchTarget
    plan: MoonEPCommPlan
    route_weights_sk: torch.Tensor | None
    metadata_done: torch.cuda.Event
    generation: int
    buffer_token: int
    capability: str = "overlapped_dispatch_publication_v1"

    @property
    def with_route_weights(self) -> bool:
        return self.route_weights_sk is not None

    def validate_layout(self) -> None:
        if self.capability != "overlapped_dispatch_publication_v1":
            raise ValueError(
                "prepared publication requires capability "
                "overlapped_dispatch_publication_v1"
            )
        self.target.validate_layout()
        if self.target.plan is not self.plan:
            raise ValueError("prepared publication must carry its target plan")
        if self.target.generation != self.generation:
            raise ValueError("prepared publication generation does not match")
        if self.target.buffer_token != self.buffer_token:
            raise ValueError("prepared publication Buffer does not match")
        if not isinstance(self.metadata_done, torch.cuda.Event):
            raise TypeError("metadata_done must be a torch.cuda.Event")
        if self.route_weights_sk is not None:
            _validate_tensor(
                "route_weights_sk",
                self.route_weights_sk,
                dtype=torch.float32,
                shape=(self.target.S, self.target.K),
                device=self.target.mapped_hidden.device,
            )
