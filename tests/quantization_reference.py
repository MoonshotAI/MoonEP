"""PyTorch reference for DeepSeek/DeepEP 1x128 FP8 activation tiles.

These helpers are for correctness tests and untimed benchmark preparation.
Communication accepts already quantized activations; it does not quantize.
"""

import torch


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2 and x.shape[1] % 128 == 0
    tiles = x.float().reshape(x.shape[0], x.shape[1] // 128, 128)
    amax = tiles.abs().amax(dim=-1).clamp_min(1e-4)
    scales = amax / 448.0
    data = (tiles * (448.0 / amax).unsqueeze(-1)).to(torch.float8_e4m3fn)
    return data.reshape(x.shape).contiguous(), scales.contiguous()


def dequantize_fp8(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    tiles = data.float().reshape(data.shape[0], data.shape[1] // 128, 128)
    return (tiles * scales.unsqueeze(-1)).reshape(data.shape).to(torch.bfloat16)
