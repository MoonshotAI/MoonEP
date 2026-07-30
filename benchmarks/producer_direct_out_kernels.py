"""Benchmark-only CUDA Tile producer kernels for direct-out publication."""

import math

import cuda.tile as ct
import torch


ConstInt = ct.Constant[int]


@ct.kernel
def producer_gemm_local_kernel(
    producer_input,
    producer_weight,
    output,
    tile_m: ConstInt,
    tile_n: ConstInt,
    tile_k: ConstInt,
):
    """Run a producer GEMM with an ordinary token-major local output."""
    bid = ct.bid(0)
    num_n_tiles = ct.cdiv(output.shape[1], tile_n)
    bid_m = bid // num_n_tiles
    bid_n = bid % num_n_tiles

    accumulator = ct.zeros((tile_m, tile_n), dtype=ct.float32)
    for k in range(0, ct.cdiv(producer_input.shape[1], tile_k)):
        a = ct.load(
            producer_input,
            (bid_m, k),
            shape=(tile_m, tile_k),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b = ct.load(
            producer_weight,
            (k, bid_n),
            shape=(tile_k, tile_n),
            padding_mode=ct.PaddingMode.ZERO,
        )
        accumulator = ct.mma(a, b, accumulator)

    ct.store(
        output,
        (bid_m, bid_n),
        ct.astype(accumulator, output.dtype),
    )


@ct.kernel
def producer_gemm_direct_out_kernel(
    producer_input,
    producer_weight,
    destinations,
    mapped_output,
    nvs: int,
    nvs_padded: int,
    topk: ConstInt,
    tile_m: ConstInt,
    tile_n: ConstInt,
    tile_k: ConstInt,
):
    """Run a producer GEMM that stores representative final-owner rows."""
    bid = ct.bid(0)
    num_n_tiles = ct.cdiv(mapped_output.shape[1], tile_n)
    bid_m = bid // num_n_tiles
    bid_n = bid % num_n_tiles

    accumulator = ct.zeros((tile_m, tile_n), dtype=ct.float32)
    for k in range(0, ct.cdiv(producer_input.shape[1], tile_k)):
        a = ct.load(
            producer_input,
            (bid_m, k),
            shape=(tile_m, tile_k),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b = ct.load(
            producer_weight,
            (k, bid_n),
            shape=(tile_k, tile_n),
            padding_mode=ct.PaddingMode.ZERO,
        )
        accumulator = ct.mma(a, b, accumulator)

    token_rows = bid_m * tile_m + ct.arange(tile_m, dtype=ct.int32)
    output_cols = bid_n * tile_n + ct.arange(tile_n, dtype=ct.int32)
    result = ct.astype(accumulator, mapped_output.dtype)
    for kidx in range(topk):
        token_topk = token_rows * topk + kidx
        encoded = ct.gather(destinations, token_topk)
        valid = encoded >= 0
        safe_encoded = ct.where(valid, encoded, 0)
        destination_rank = safe_encoded // nvs
        destination_loff = safe_encoded - destination_rank * nvs
        physical_row = destination_rank * nvs_padded + destination_loff
        ct.scatter(
            mapped_output,
            (physical_row[:, None], output_cols[None, :]),
            result,
            mask=valid[:, None],
        )


def launch_producer_gemm_local(
    producer_input: torch.Tensor,
    producer_weight: torch.Tensor,
    output: torch.Tensor,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    """Launch the local-output producer."""
    grid = (
        math.ceil(output.shape[0] / tile_m)
        * math.ceil(output.shape[1] / tile_n),
    )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        producer_gemm_local_kernel,
        (
            producer_input,
            producer_weight,
            output,
            tile_m,
            tile_n,
            tile_k,
        ),
    )


def launch_producer_gemm_direct_out(
    producer_input: torch.Tensor,
    producer_weight: torch.Tensor,
    destinations: torch.Tensor,
    mapped_output: torch.Tensor,
    *,
    nvs: int,
    nvs_padded: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> None:
    """Launch the producer with a remote final-owner store epilogue."""
    grid = (
        math.ceil(producer_input.shape[0] / tile_m)
        * math.ceil(mapped_output.shape[1] / tile_n),
    )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        producer_gemm_direct_out_kernel,
        (
            producer_input,
            producer_weight,
            destinations,
            mapped_output,
            nvs,
            nvs_padded,
            topk,
            tile_m,
            tile_n,
            tile_k,
        ),
    )
