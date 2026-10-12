
# ============================================================
# Bit widths for the packed int32 dedup encodings (single source of
# truth for planning / dispatch kernels and tests).
# ============================================================
# 2^7 = 128, which is enough for current mainstream EP settings
RANK_BITS = 7
# Topk up to 32 uses a single b32 kmask; KIDX_BITS is wider so the packed
# kidx field keeps room for future non-bitmask encodings.
KIDX_BITS = 7

# Number of dedup-builder warps per dispatch CTA. The builder's chunk scans
# are latency-bound with a single warp (one dependent-load chain per SM);
# splitting each CTA chunk over several warps divides that wall time so the
# structure generation hides under the dispatch NVLink transfer. The exposed
# tail flattens out beyond a few warps, so 4 is the sweet spot.
DEDUP_BUILDER_WARPS = 4

# DeepSeek activation quantization: one fp32 dequantization scale per 1x128 tile.
FP8_TILE_SIZE = 128
