"""Triton-accelerated RoPE kernels for PoM (Polynomial Mixer).

Fuses: pom_activation + polynomial expansion + RoPE rotation + aggregation
into a single pass over the input, eliminating the (B, N, D) intermediate
produced by the PyTorch path (poly_features → apply_rope_1d → aggregate).

Two variants are provided:

  poly_agg_rope_mean_triton   — global mean (no mask),   output (B, 1, D)
  poly_agg_rope_causal_triton — causal prefix mean,       output (B, N, D)

Grid structure
--------------
The rotate-half convention pairs dimension d with dimension d + D/2, so
both must be live simultaneously to compute the rotation.  The kernels use:

  grid = (B, ceil(HALF_D / BLOCK_P)),  HALF_D = D // 2

Each program handles BLOCK_P pairs: columns d_off and d_off + HALF_D.
Two coeff blocks (c1_k for d_off, c2_k for d_off + HALF_D) are preloaded
into registers; two accumulator arrays are maintained.

BLOCK_P is chosen as BLOCK_D / 2 so that total register usage matches the
non-RoPE kernels (half the block width, but twice as many coefficient arrays).

RoPE backward
-------------
The 2×2 rotation [[cos, -sin], [sin, cos]] is orthogonal, so its transpose is
[[cos, sin], [-sin, cos]], giving:

  d_poly1 = d_r1 * cos + d_r2 * sin
  d_poly2 = -d_r1 * sin + d_r2 * cos

The remainder of the backward (power-sharing for grad_h, atomic_add for
grad_coeff) follows pom_triton.py and pom_triton_causal.py exactly.

Positions
---------
Both functions accept an optional integer positions tensor (N,) specifying
the token indices used to look up freqs_cos / freqs_sin.  If None the wrapper
creates 0..N-1 automatically.  Custom positions (chunked prefill, packing)
are supported at no kernel-logic cost — the only overhead is one int64 load
per N step.

Exposed API
-----------
TRITON_ROPE_AVAILABLE : bool
poly_agg_rope_mean_triton(x, coeff, k, freqs_cos, freqs_sin,
                          positions=None) -> (B, 1, D)
poly_agg_rope_causal_triton(x, coeff, k, freqs_cos, freqs_sin,
                            positions=None) -> (B, N, D)
"""
import os
import torch

try:
    import triton
    import triton.language as tl
    TRITON_ROPE_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_ROPE_AVAILABLE = False


if TRITON_ROPE_AVAILABLE:

    # -------------------------------------------------------------------------
    # BLOCK_P heuristics
    #
    # Each program covers BLOCK_P pairs (2 * BLOCK_P total channels).
    # Set to half of the standard BLOCK_D caps so register pressure is the same:
    #   forward: BLOCK_P ≤ 128  (2×128 channels ↔ standard 1×256 BLOCK_D)
    #   backward: BLOCK_P ≤ 64  (2×64  channels ↔ standard 1×128 BLOCK_D)
    # -------------------------------------------------------------------------

    def _fwd_block_p(D: int) -> int:
        half = D // 2
        return min(128, 1 << (half - 1).bit_length())

    def _bwd_block_p(D: int) -> int:
        half = D // 2
        return min(64, 1 << (half - 1).bit_length())

    # -------------------------------------------------------------------------
    # Unmasked (global mean) — forward kernel
    #
    # Grid: (B, ceil(HALF_D / BLOCK_P))
    # Each program streams n = 0..N-1, accumulating:
    #   acc1 += poly(act(x[:,d_off]))    * cos_n  −  poly(act(x[:,d_off+H])) * sin_n
    #   acc2 += poly(act(x[:,d_off+H])) * cos_n  +  poly(act(x[:,d_off]))   * sin_n
    # and stores acc/N to the output.
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_rope_mean_fwd(
        X_ptr,          # (B, N, D) input, any dtype
        C_ptr,          # (D, K)    coefficients, fp32
        FC_ptr,         # (S, H)    freqs_cos, fp32  S=max_seq_len, H=D//2
        FS_ptr,         # (S, H)    freqs_sin, fp32
        POS_ptr,        # (N,)      position indices, int64
        O_ptr,          # (B, D)    output, fp32
        N, D, HALF_D,
        stride_xb, stride_xn,
        stride_fc,      # freqs_cos.stride(0) = H
        K: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        b     = tl.program_id(0)
        p_blk = tl.program_id(1)
        d_off = p_blk * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = d_off < HALF_D

        # Preload coefficients for both halves into registers.
        c1_0  = tl.load(C_ptr + d_off * K + 0,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1_1  = tl.load(C_ptr + d_off * K + 1,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c1_2  = tl.load(C_ptr + d_off * K + 2,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c1_3  = tl.load(C_ptr + d_off * K + 3,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c1_4  = tl.load(C_ptr + d_off * K + 4,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c1_5  = tl.load(C_ptr + d_off * K + 5,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c1_6  = tl.load(C_ptr + d_off * K + 6,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c1_7  = tl.load(C_ptr + d_off * K + 7,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c1_8  = tl.load(C_ptr + d_off * K + 8,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c1_9  = tl.load(C_ptr + d_off * K + 9,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c1_10 = tl.load(C_ptr + d_off * K + 10,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c1_11 = tl.load(C_ptr + d_off * K + 11,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c1_12 = tl.load(C_ptr + d_off * K + 12,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c1_13 = tl.load(C_ptr + d_off * K + 13,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c1_14 = tl.load(C_ptr + d_off * K + 14,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c1_15 = tl.load(C_ptr + d_off * K + 15,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        c2_0  = tl.load(C_ptr + (d_off + HALF_D) * K + 0,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c2_1  = tl.load(C_ptr + (d_off + HALF_D) * K + 1,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2_2  = tl.load(C_ptr + (d_off + HALF_D) * K + 2,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c2_3  = tl.load(C_ptr + (d_off + HALF_D) * K + 3,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c2_4  = tl.load(C_ptr + (d_off + HALF_D) * K + 4,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c2_5  = tl.load(C_ptr + (d_off + HALF_D) * K + 5,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c2_6  = tl.load(C_ptr + (d_off + HALF_D) * K + 6,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c2_7  = tl.load(C_ptr + (d_off + HALF_D) * K + 7,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c2_8  = tl.load(C_ptr + (d_off + HALF_D) * K + 8,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c2_9  = tl.load(C_ptr + (d_off + HALF_D) * K + 9,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c2_10 = tl.load(C_ptr + (d_off + HALF_D) * K + 10, mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c2_11 = tl.load(C_ptr + (d_off + HALF_D) * K + 11, mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c2_12 = tl.load(C_ptr + (d_off + HALF_D) * K + 12, mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c2_13 = tl.load(C_ptr + (d_off + HALF_D) * K + 13, mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c2_14 = tl.load(C_ptr + (d_off + HALF_D) * K + 14, mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c2_15 = tl.load(C_ptr + (d_off + HALF_D) * K + 15, mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc1 = tl.zeros((BLOCK_P,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for n in range(N):
            pos_n = tl.load(POS_ptr + n)
            cos_n = tl.load(FC_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)
            sin_n = tl.load(FS_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)

            x1 = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=pmask, other=0.0, eviction_policy="evict_first",
            ).to(tl.float32)
            x2 = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off + HALF_D,
                mask=pmask, other=0.0, eviction_policy="evict_first",
            ).to(tl.float32)

            # pom_activation
            h1 = tl.where(x1 >= 0.0, x1, x1 * 0.01)
            h1 = tl.maximum(h1, -0.1); h1 = tl.minimum(h1, 6.0)
            h2 = tl.where(x2 >= 0.0, x2, x2 * 0.01)
            h2 = tl.maximum(h2, -0.1); h2 = tl.minimum(h2, 6.0)

            # Polynomial expansion — first half
            poly1 = tl.zeros((BLOCK_P,), tl.float32)
            hp1 = h1
            if K > 0:  poly1 += c1_0 * hp1
            if K > 1:  hp1 *= h1; poly1 += c1_1 * hp1
            if K > 2:  hp1 *= h1; poly1 += c1_2 * hp1
            if K > 3:  hp1 *= h1; poly1 += c1_3 * hp1
            if K > 4:  hp1 *= h1; poly1 += c1_4 * hp1
            if K > 5:  hp1 *= h1; poly1 += c1_5 * hp1
            if K > 6:  hp1 *= h1; poly1 += c1_6 * hp1
            if K > 7:  hp1 *= h1; poly1 += c1_7 * hp1
            if K > 8:  hp1 *= h1; poly1 += c1_8 * hp1
            if K > 9:  hp1 *= h1; poly1 += c1_9 * hp1
            if K > 10: hp1 *= h1; poly1 += c1_10 * hp1
            if K > 11: hp1 *= h1; poly1 += c1_11 * hp1
            if K > 12: hp1 *= h1; poly1 += c1_12 * hp1
            if K > 13: hp1 *= h1; poly1 += c1_13 * hp1
            if K > 14: hp1 *= h1; poly1 += c1_14 * hp1
            if K > 15: hp1 *= h1; poly1 += c1_15 * hp1

            # Polynomial expansion — second half
            poly2 = tl.zeros((BLOCK_P,), tl.float32)
            hp2 = h2
            if K > 0:  poly2 += c2_0 * hp2
            if K > 1:  hp2 *= h2; poly2 += c2_1 * hp2
            if K > 2:  hp2 *= h2; poly2 += c2_2 * hp2
            if K > 3:  hp2 *= h2; poly2 += c2_3 * hp2
            if K > 4:  hp2 *= h2; poly2 += c2_4 * hp2
            if K > 5:  hp2 *= h2; poly2 += c2_5 * hp2
            if K > 6:  hp2 *= h2; poly2 += c2_6 * hp2
            if K > 7:  hp2 *= h2; poly2 += c2_7 * hp2
            if K > 8:  hp2 *= h2; poly2 += c2_8 * hp2
            if K > 9:  hp2 *= h2; poly2 += c2_9 * hp2
            if K > 10: hp2 *= h2; poly2 += c2_10 * hp2
            if K > 11: hp2 *= h2; poly2 += c2_11 * hp2
            if K > 12: hp2 *= h2; poly2 += c2_12 * hp2
            if K > 13: hp2 *= h2; poly2 += c2_13 * hp2
            if K > 14: hp2 *= h2; poly2 += c2_14 * hp2
            if K > 15: hp2 *= h2; poly2 += c2_15 * hp2

            # RoPE rotate-half: [r1, r2] = [[cos,-sin],[sin,cos]] [poly1, poly2]
            acc1 += poly1 * cos_n - poly2 * sin_n
            acc2 += poly2 * cos_n + poly1 * sin_n

        tl.store(O_ptr + b * D + d_off,          acc1 / N, mask=pmask)
        tl.store(O_ptr + b * D + d_off + HALF_D, acc2 / N, mask=pmask)

    # -------------------------------------------------------------------------
    # Unmasked — backward kernel
    #
    # go has shape (B, D); loaded once into registers.
    # Effective upstream gradient per step n: eff_go = go / N.
    # RoPE backward (transpose): [[cos,sin],[-sin,cos]] * [eff_go1, eff_go2]
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_rope_mean_bwd(
        GO_ptr,         # (B, D)    upstream gradient, fp32
        X_ptr,          # (B, N, D) saved input
        C_ptr,          # (D, K)    coefficients, fp32
        FC_ptr,         # (S, H)    freqs_cos, fp32
        FS_ptr,         # (S, H)    freqs_sin, fp32
        POS_ptr,        # (N,)      position indices, int64
        GX_ptr,         # (B, N, D) grad w.r.t. X, fp32
        GC_ptr,         # (D, K)    grad w.r.t. coeff, fp32 (zero-init, atomic)
        B, N, D, HALF_D,
        stride_xb, stride_xn,
        stride_fc,
        K: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        b     = tl.program_id(0)
        p_blk = tl.program_id(1)
        d_off = p_blk * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = d_off < HALF_D

        inv_N = 1.0 / N
        eff_go1 = tl.load(GO_ptr + b * D + d_off,          mask=pmask, other=0.0).to(tl.float32) * inv_N
        eff_go2 = tl.load(GO_ptr + b * D + d_off + HALF_D, mask=pmask, other=0.0).to(tl.float32) * inv_N

        c1_0  = tl.load(C_ptr + d_off * K + 0,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1_1  = tl.load(C_ptr + d_off * K + 1,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c1_2  = tl.load(C_ptr + d_off * K + 2,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c1_3  = tl.load(C_ptr + d_off * K + 3,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c1_4  = tl.load(C_ptr + d_off * K + 4,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c1_5  = tl.load(C_ptr + d_off * K + 5,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c1_6  = tl.load(C_ptr + d_off * K + 6,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c1_7  = tl.load(C_ptr + d_off * K + 7,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c1_8  = tl.load(C_ptr + d_off * K + 8,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c1_9  = tl.load(C_ptr + d_off * K + 9,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c1_10 = tl.load(C_ptr + d_off * K + 10,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c1_11 = tl.load(C_ptr + d_off * K + 11,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c1_12 = tl.load(C_ptr + d_off * K + 12,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c1_13 = tl.load(C_ptr + d_off * K + 13,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c1_14 = tl.load(C_ptr + d_off * K + 14,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c1_15 = tl.load(C_ptr + d_off * K + 15,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        c2_0  = tl.load(C_ptr + (d_off + HALF_D) * K + 0,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c2_1  = tl.load(C_ptr + (d_off + HALF_D) * K + 1,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2_2  = tl.load(C_ptr + (d_off + HALF_D) * K + 2,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c2_3  = tl.load(C_ptr + (d_off + HALF_D) * K + 3,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c2_4  = tl.load(C_ptr + (d_off + HALF_D) * K + 4,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c2_5  = tl.load(C_ptr + (d_off + HALF_D) * K + 5,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c2_6  = tl.load(C_ptr + (d_off + HALF_D) * K + 6,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c2_7  = tl.load(C_ptr + (d_off + HALF_D) * K + 7,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c2_8  = tl.load(C_ptr + (d_off + HALF_D) * K + 8,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c2_9  = tl.load(C_ptr + (d_off + HALF_D) * K + 9,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c2_10 = tl.load(C_ptr + (d_off + HALF_D) * K + 10, mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c2_11 = tl.load(C_ptr + (d_off + HALF_D) * K + 11, mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c2_12 = tl.load(C_ptr + (d_off + HALF_D) * K + 12, mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c2_13 = tl.load(C_ptr + (d_off + HALF_D) * K + 13, mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c2_14 = tl.load(C_ptr + (d_off + HALF_D) * K + 14, mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c2_15 = tl.load(C_ptr + (d_off + HALF_D) * K + 15, mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        a1_0  = tl.zeros((BLOCK_P,), tl.float32)
        a1_1  = tl.zeros((BLOCK_P,), tl.float32)
        a1_2  = tl.zeros((BLOCK_P,), tl.float32)
        a1_3  = tl.zeros((BLOCK_P,), tl.float32)
        a1_4  = tl.zeros((BLOCK_P,), tl.float32)
        a1_5  = tl.zeros((BLOCK_P,), tl.float32)
        a1_6  = tl.zeros((BLOCK_P,), tl.float32)
        a1_7  = tl.zeros((BLOCK_P,), tl.float32)
        a1_8  = tl.zeros((BLOCK_P,), tl.float32)
        a1_9  = tl.zeros((BLOCK_P,), tl.float32)
        a1_10 = tl.zeros((BLOCK_P,), tl.float32)
        a1_11 = tl.zeros((BLOCK_P,), tl.float32)
        a1_12 = tl.zeros((BLOCK_P,), tl.float32)
        a1_13 = tl.zeros((BLOCK_P,), tl.float32)
        a1_14 = tl.zeros((BLOCK_P,), tl.float32)
        a1_15 = tl.zeros((BLOCK_P,), tl.float32)

        a2_0  = tl.zeros((BLOCK_P,), tl.float32)
        a2_1  = tl.zeros((BLOCK_P,), tl.float32)
        a2_2  = tl.zeros((BLOCK_P,), tl.float32)
        a2_3  = tl.zeros((BLOCK_P,), tl.float32)
        a2_4  = tl.zeros((BLOCK_P,), tl.float32)
        a2_5  = tl.zeros((BLOCK_P,), tl.float32)
        a2_6  = tl.zeros((BLOCK_P,), tl.float32)
        a2_7  = tl.zeros((BLOCK_P,), tl.float32)
        a2_8  = tl.zeros((BLOCK_P,), tl.float32)
        a2_9  = tl.zeros((BLOCK_P,), tl.float32)
        a2_10 = tl.zeros((BLOCK_P,), tl.float32)
        a2_11 = tl.zeros((BLOCK_P,), tl.float32)
        a2_12 = tl.zeros((BLOCK_P,), tl.float32)
        a2_13 = tl.zeros((BLOCK_P,), tl.float32)
        a2_14 = tl.zeros((BLOCK_P,), tl.float32)
        a2_15 = tl.zeros((BLOCK_P,), tl.float32)

        for n in range(N):
            pos_n = tl.load(POS_ptr + n)
            cos_n = tl.load(FC_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)
            sin_n = tl.load(FS_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)

            # RoPE^T: d_poly = [[cos,sin],[-sin,cos]] * eff_go
            d_poly1 = eff_go1 * cos_n + eff_go2 * sin_n
            d_poly2 = -eff_go1 * sin_n + eff_go2 * cos_n

            x1 = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=pmask, other=0.0, eviction_policy="evict_first",
            ).to(tl.float32)
            x2 = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off + HALF_D,
                mask=pmask, other=0.0, eviction_policy="evict_first",
            ).to(tl.float32)

            h1 = tl.where(x1 >= 0.0, x1, x1 * 0.01)
            h1 = tl.maximum(h1, -0.1); h1 = tl.minimum(h1, 6.0)
            h2 = tl.where(x2 >= 0.0, x2, x2 * 0.01)
            h2 = tl.maximum(h2, -0.1); h2 = tl.minimum(h2, 6.0)

            d_act1 = tl.where((x1 >= 0.0) & (x1 <= 6.0), 1.0,
                              tl.where((x1 < 0.0) & (x1 >= -10.0), 0.01, 0.0))
            d_act2 = tl.where((x2 >= 0.0) & (x2 <= 6.0), 1.0,
                              tl.where((x2 < 0.0) & (x2 >= -10.0), 0.01, 0.0))

            # Power-sharing backward — first half
            grad_h1 = tl.zeros((BLOCK_P,), tl.float32)
            hp1 = tl.full((BLOCK_P,), 1.0, tl.float32)
            if K > 0:
                hph1 = hp1 * h1; grad_h1 += c1_0 * hp1;          a1_0  += d_poly1 * hph1; hp1 = hph1
            if K > 1:
                hph1 = hp1 * h1; grad_h1 += c1_1 * 2.0 * hp1;   a1_1  += d_poly1 * hph1; hp1 = hph1
            if K > 2:
                hph1 = hp1 * h1; grad_h1 += c1_2 * 3.0 * hp1;   a1_2  += d_poly1 * hph1; hp1 = hph1
            if K > 3:
                hph1 = hp1 * h1; grad_h1 += c1_3 * 4.0 * hp1;   a1_3  += d_poly1 * hph1; hp1 = hph1
            if K > 4:
                hph1 = hp1 * h1; grad_h1 += c1_4 * 5.0 * hp1;   a1_4  += d_poly1 * hph1; hp1 = hph1
            if K > 5:
                hph1 = hp1 * h1; grad_h1 += c1_5 * 6.0 * hp1;   a1_5  += d_poly1 * hph1; hp1 = hph1
            if K > 6:
                hph1 = hp1 * h1; grad_h1 += c1_6 * 7.0 * hp1;   a1_6  += d_poly1 * hph1; hp1 = hph1
            if K > 7:
                hph1 = hp1 * h1; grad_h1 += c1_7 * 8.0 * hp1;   a1_7  += d_poly1 * hph1; hp1 = hph1
            if K > 8:
                hph1 = hp1 * h1; grad_h1 += c1_8 * 9.0 * hp1;   a1_8  += d_poly1 * hph1; hp1 = hph1
            if K > 9:
                hph1 = hp1 * h1; grad_h1 += c1_9 * 10.0 * hp1;  a1_9  += d_poly1 * hph1; hp1 = hph1
            if K > 10:
                hph1 = hp1 * h1; grad_h1 += c1_10 * 11.0 * hp1; a1_10 += d_poly1 * hph1; hp1 = hph1
            if K > 11:
                hph1 = hp1 * h1; grad_h1 += c1_11 * 12.0 * hp1; a1_11 += d_poly1 * hph1; hp1 = hph1
            if K > 12:
                hph1 = hp1 * h1; grad_h1 += c1_12 * 13.0 * hp1; a1_12 += d_poly1 * hph1; hp1 = hph1
            if K > 13:
                hph1 = hp1 * h1; grad_h1 += c1_13 * 14.0 * hp1; a1_13 += d_poly1 * hph1; hp1 = hph1
            if K > 14:
                hph1 = hp1 * h1; grad_h1 += c1_14 * 15.0 * hp1; a1_14 += d_poly1 * hph1; hp1 = hph1
            if K > 15:
                hph1 = hp1 * h1; grad_h1 += c1_15 * 16.0 * hp1; a1_15 += d_poly1 * hph1; hp1 = hph1

            # Power-sharing backward — second half
            grad_h2 = tl.zeros((BLOCK_P,), tl.float32)
            hp2 = tl.full((BLOCK_P,), 1.0, tl.float32)
            if K > 0:
                hph2 = hp2 * h2; grad_h2 += c2_0 * hp2;          a2_0  += d_poly2 * hph2; hp2 = hph2
            if K > 1:
                hph2 = hp2 * h2; grad_h2 += c2_1 * 2.0 * hp2;   a2_1  += d_poly2 * hph2; hp2 = hph2
            if K > 2:
                hph2 = hp2 * h2; grad_h2 += c2_2 * 3.0 * hp2;   a2_2  += d_poly2 * hph2; hp2 = hph2
            if K > 3:
                hph2 = hp2 * h2; grad_h2 += c2_3 * 4.0 * hp2;   a2_3  += d_poly2 * hph2; hp2 = hph2
            if K > 4:
                hph2 = hp2 * h2; grad_h2 += c2_4 * 5.0 * hp2;   a2_4  += d_poly2 * hph2; hp2 = hph2
            if K > 5:
                hph2 = hp2 * h2; grad_h2 += c2_5 * 6.0 * hp2;   a2_5  += d_poly2 * hph2; hp2 = hph2
            if K > 6:
                hph2 = hp2 * h2; grad_h2 += c2_6 * 7.0 * hp2;   a2_6  += d_poly2 * hph2; hp2 = hph2
            if K > 7:
                hph2 = hp2 * h2; grad_h2 += c2_7 * 8.0 * hp2;   a2_7  += d_poly2 * hph2; hp2 = hph2
            if K > 8:
                hph2 = hp2 * h2; grad_h2 += c2_8 * 9.0 * hp2;   a2_8  += d_poly2 * hph2; hp2 = hph2
            if K > 9:
                hph2 = hp2 * h2; grad_h2 += c2_9 * 10.0 * hp2;  a2_9  += d_poly2 * hph2; hp2 = hph2
            if K > 10:
                hph2 = hp2 * h2; grad_h2 += c2_10 * 11.0 * hp2; a2_10 += d_poly2 * hph2; hp2 = hph2
            if K > 11:
                hph2 = hp2 * h2; grad_h2 += c2_11 * 12.0 * hp2; a2_11 += d_poly2 * hph2; hp2 = hph2
            if K > 12:
                hph2 = hp2 * h2; grad_h2 += c2_12 * 13.0 * hp2; a2_12 += d_poly2 * hph2; hp2 = hph2
            if K > 13:
                hph2 = hp2 * h2; grad_h2 += c2_13 * 14.0 * hp2; a2_13 += d_poly2 * hph2; hp2 = hph2
            if K > 14:
                hph2 = hp2 * h2; grad_h2 += c2_14 * 15.0 * hp2; a2_14 += d_poly2 * hph2; hp2 = hph2
            if K > 15:
                hph2 = hp2 * h2; grad_h2 += c2_15 * 16.0 * hp2; a2_15 += d_poly2 * hph2; hp2 = hph2

            tl.store(GX_ptr + b * stride_xb + n * stride_xn + d_off,
                     d_poly1 * grad_h1 * d_act1, mask=pmask)
            tl.store(GX_ptr + b * stride_xb + n * stride_xn + d_off + HALF_D,
                     d_poly2 * grad_h2 * d_act2, mask=pmask)

        # Atomic-add partial grad_coeff sums (reduces across B).
        if K > 0:
            tl.atomic_add(GC_ptr + d_off * K + 0,           a1_0,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 0, a2_0, mask=pmask)
        if K > 1:
            tl.atomic_add(GC_ptr + d_off * K + 1,           a1_1, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 1, a2_1, mask=pmask)
        if K > 2:
            tl.atomic_add(GC_ptr + d_off * K + 2,           a1_2, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 2, a2_2, mask=pmask)
        if K > 3:
            tl.atomic_add(GC_ptr + d_off * K + 3,           a1_3, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 3, a2_3, mask=pmask)
        if K > 4:
            tl.atomic_add(GC_ptr + d_off * K + 4,           a1_4, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 4, a2_4, mask=pmask)
        if K > 5:
            tl.atomic_add(GC_ptr + d_off * K + 5,           a1_5, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 5, a2_5, mask=pmask)
        if K > 6:
            tl.atomic_add(GC_ptr + d_off * K + 6,           a1_6, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 6, a2_6, mask=pmask)
        if K > 7:
            tl.atomic_add(GC_ptr + d_off * K + 7,           a1_7, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 7, a2_7, mask=pmask)
        if K > 8:
            tl.atomic_add(GC_ptr + d_off * K + 8,           a1_8, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 8, a2_8, mask=pmask)
        if K > 9:
            tl.atomic_add(GC_ptr + d_off * K + 9,           a1_9, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 9, a2_9, mask=pmask)
        if K > 10:
            tl.atomic_add(GC_ptr + d_off * K + 10,           a1_10, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 10, a2_10, mask=pmask)
        if K > 11:
            tl.atomic_add(GC_ptr + d_off * K + 11,           a1_11, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 11, a2_11, mask=pmask)
        if K > 12:
            tl.atomic_add(GC_ptr + d_off * K + 12,           a1_12, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 12, a2_12, mask=pmask)
        if K > 13:
            tl.atomic_add(GC_ptr + d_off * K + 13,           a1_13, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 13, a2_13, mask=pmask)
        if K > 14:
            tl.atomic_add(GC_ptr + d_off * K + 14,           a1_14, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 14, a2_14, mask=pmask)
        if K > 15:
            tl.atomic_add(GC_ptr + d_off * K + 15,           a1_15, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 15, a2_15, mask=pmask)

    # -------------------------------------------------------------------------
    # Causal — forward kernel
    #
    # Writes running mean acc / (n+1) at each step n.
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_rope_causal_fwd(
        X_ptr,
        C_ptr,
        FC_ptr,
        FS_ptr,
        POS_ptr,
        O_ptr,          # (B, N, D) output, fp32
        N, D, HALF_D,
        stride_xb, stride_xn,
        stride_ob, stride_on,
        stride_fc,
        K: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        b     = tl.program_id(0)
        p_blk = tl.program_id(1)
        d_off = p_blk * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = d_off < HALF_D

        c1_0  = tl.load(C_ptr + d_off * K + 0,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1_1  = tl.load(C_ptr + d_off * K + 1,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c1_2  = tl.load(C_ptr + d_off * K + 2,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c1_3  = tl.load(C_ptr + d_off * K + 3,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c1_4  = tl.load(C_ptr + d_off * K + 4,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c1_5  = tl.load(C_ptr + d_off * K + 5,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c1_6  = tl.load(C_ptr + d_off * K + 6,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c1_7  = tl.load(C_ptr + d_off * K + 7,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c1_8  = tl.load(C_ptr + d_off * K + 8,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c1_9  = tl.load(C_ptr + d_off * K + 9,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c1_10 = tl.load(C_ptr + d_off * K + 10,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c1_11 = tl.load(C_ptr + d_off * K + 11,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c1_12 = tl.load(C_ptr + d_off * K + 12,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c1_13 = tl.load(C_ptr + d_off * K + 13,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c1_14 = tl.load(C_ptr + d_off * K + 14,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c1_15 = tl.load(C_ptr + d_off * K + 15,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        c2_0  = tl.load(C_ptr + (d_off + HALF_D) * K + 0,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c2_1  = tl.load(C_ptr + (d_off + HALF_D) * K + 1,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2_2  = tl.load(C_ptr + (d_off + HALF_D) * K + 2,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c2_3  = tl.load(C_ptr + (d_off + HALF_D) * K + 3,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c2_4  = tl.load(C_ptr + (d_off + HALF_D) * K + 4,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c2_5  = tl.load(C_ptr + (d_off + HALF_D) * K + 5,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c2_6  = tl.load(C_ptr + (d_off + HALF_D) * K + 6,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c2_7  = tl.load(C_ptr + (d_off + HALF_D) * K + 7,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c2_8  = tl.load(C_ptr + (d_off + HALF_D) * K + 8,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c2_9  = tl.load(C_ptr + (d_off + HALF_D) * K + 9,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c2_10 = tl.load(C_ptr + (d_off + HALF_D) * K + 10, mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c2_11 = tl.load(C_ptr + (d_off + HALF_D) * K + 11, mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c2_12 = tl.load(C_ptr + (d_off + HALF_D) * K + 12, mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c2_13 = tl.load(C_ptr + (d_off + HALF_D) * K + 13, mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c2_14 = tl.load(C_ptr + (d_off + HALF_D) * K + 14, mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c2_15 = tl.load(C_ptr + (d_off + HALF_D) * K + 15, mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc1 = tl.zeros((BLOCK_P,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for n in range(N):
            pos_n = tl.load(POS_ptr + n)
            cos_n = tl.load(FC_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)
            sin_n = tl.load(FS_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)

            x1 = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=pmask, other=0.0, eviction_policy="evict_first",
            ).to(tl.float32)
            x2 = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off + HALF_D,
                mask=pmask, other=0.0, eviction_policy="evict_first",
            ).to(tl.float32)

            h1 = tl.where(x1 >= 0.0, x1, x1 * 0.01)
            h1 = tl.maximum(h1, -0.1); h1 = tl.minimum(h1, 6.0)
            h2 = tl.where(x2 >= 0.0, x2, x2 * 0.01)
            h2 = tl.maximum(h2, -0.1); h2 = tl.minimum(h2, 6.0)

            poly1 = tl.zeros((BLOCK_P,), tl.float32)
            hp1 = h1
            if K > 0:  poly1 += c1_0 * hp1
            if K > 1:  hp1 *= h1; poly1 += c1_1 * hp1
            if K > 2:  hp1 *= h1; poly1 += c1_2 * hp1
            if K > 3:  hp1 *= h1; poly1 += c1_3 * hp1
            if K > 4:  hp1 *= h1; poly1 += c1_4 * hp1
            if K > 5:  hp1 *= h1; poly1 += c1_5 * hp1
            if K > 6:  hp1 *= h1; poly1 += c1_6 * hp1
            if K > 7:  hp1 *= h1; poly1 += c1_7 * hp1
            if K > 8:  hp1 *= h1; poly1 += c1_8 * hp1
            if K > 9:  hp1 *= h1; poly1 += c1_9 * hp1
            if K > 10: hp1 *= h1; poly1 += c1_10 * hp1
            if K > 11: hp1 *= h1; poly1 += c1_11 * hp1
            if K > 12: hp1 *= h1; poly1 += c1_12 * hp1
            if K > 13: hp1 *= h1; poly1 += c1_13 * hp1
            if K > 14: hp1 *= h1; poly1 += c1_14 * hp1
            if K > 15: hp1 *= h1; poly1 += c1_15 * hp1

            poly2 = tl.zeros((BLOCK_P,), tl.float32)
            hp2 = h2
            if K > 0:  poly2 += c2_0 * hp2
            if K > 1:  hp2 *= h2; poly2 += c2_1 * hp2
            if K > 2:  hp2 *= h2; poly2 += c2_2 * hp2
            if K > 3:  hp2 *= h2; poly2 += c2_3 * hp2
            if K > 4:  hp2 *= h2; poly2 += c2_4 * hp2
            if K > 5:  hp2 *= h2; poly2 += c2_5 * hp2
            if K > 6:  hp2 *= h2; poly2 += c2_6 * hp2
            if K > 7:  hp2 *= h2; poly2 += c2_7 * hp2
            if K > 8:  hp2 *= h2; poly2 += c2_8 * hp2
            if K > 9:  hp2 *= h2; poly2 += c2_9 * hp2
            if K > 10: hp2 *= h2; poly2 += c2_10 * hp2
            if K > 11: hp2 *= h2; poly2 += c2_11 * hp2
            if K > 12: hp2 *= h2; poly2 += c2_12 * hp2
            if K > 13: hp2 *= h2; poly2 += c2_13 * hp2
            if K > 14: hp2 *= h2; poly2 += c2_14 * hp2
            if K > 15: hp2 *= h2; poly2 += c2_15 * hp2

            acc1 += poly1 * cos_n - poly2 * sin_n
            acc2 += poly2 * cos_n + poly1 * sin_n

            inv_np1 = 1.0 / (n + 1)
            tl.store(O_ptr + b * stride_ob + n * stride_on + d_off,
                     acc1 * inv_np1, mask=pmask)
            tl.store(O_ptr + b * stride_ob + n * stride_on + d_off + HALF_D,
                     acc2 * inv_np1, mask=pmask)

    # -------------------------------------------------------------------------
    # Causal — backward kernel
    #
    # Iterates n = N-1..0 accumulating the suffix-weighted upstream gradients:
    #   suffix_w1[n] = sum_{m=n}^{N-1} go[b,m,d_off]       / (m+1)
    #   suffix_w2[n] = sum_{m=n}^{N-1} go[b,m,d_off+HALF_D] / (m+1)
    # Then transposes the RoPE rotation at position n and applies the standard
    # power-sharing polynomial backward for both halves.
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_rope_causal_bwd(
        GO_ptr,         # (B, N, D)  upstream gradient, fp32
        X_ptr,          # (B, N, D)  saved input
        C_ptr,          # (D, K)     coefficients, fp32
        FC_ptr,         # (S, H)     freqs_cos, fp32
        FS_ptr,         # (S, H)     freqs_sin, fp32
        POS_ptr,        # (N,)       position indices, int64
        GX_ptr,         # (B, N, D)  grad_x output, fp32
        GC_ptr,         # (D, K)     grad_coeff (zero-init, atomic), fp32
        B, N, D, HALF_D,
        stride_xb, stride_xn,
        stride_gob, stride_gon,
        stride_fc,
        K: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        b     = tl.program_id(0)
        p_blk = tl.program_id(1)
        d_off = p_blk * BLOCK_P + tl.arange(0, BLOCK_P)
        pmask = d_off < HALF_D

        c1_0  = tl.load(C_ptr + d_off * K + 0,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1_1  = tl.load(C_ptr + d_off * K + 1,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c1_2  = tl.load(C_ptr + d_off * K + 2,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c1_3  = tl.load(C_ptr + d_off * K + 3,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c1_4  = tl.load(C_ptr + d_off * K + 4,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c1_5  = tl.load(C_ptr + d_off * K + 5,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c1_6  = tl.load(C_ptr + d_off * K + 6,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c1_7  = tl.load(C_ptr + d_off * K + 7,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c1_8  = tl.load(C_ptr + d_off * K + 8,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c1_9  = tl.load(C_ptr + d_off * K + 9,           mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c1_10 = tl.load(C_ptr + d_off * K + 10,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c1_11 = tl.load(C_ptr + d_off * K + 11,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c1_12 = tl.load(C_ptr + d_off * K + 12,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c1_13 = tl.load(C_ptr + d_off * K + 13,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c1_14 = tl.load(C_ptr + d_off * K + 14,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c1_15 = tl.load(C_ptr + d_off * K + 15,          mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        c2_0  = tl.load(C_ptr + (d_off + HALF_D) * K + 0,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c2_1  = tl.load(C_ptr + (d_off + HALF_D) * K + 1,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2_2  = tl.load(C_ptr + (d_off + HALF_D) * K + 2,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c2_3  = tl.load(C_ptr + (d_off + HALF_D) * K + 3,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c2_4  = tl.load(C_ptr + (d_off + HALF_D) * K + 4,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c2_5  = tl.load(C_ptr + (d_off + HALF_D) * K + 5,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c2_6  = tl.load(C_ptr + (d_off + HALF_D) * K + 6,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c2_7  = tl.load(C_ptr + (d_off + HALF_D) * K + 7,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c2_8  = tl.load(C_ptr + (d_off + HALF_D) * K + 8,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c2_9  = tl.load(C_ptr + (d_off + HALF_D) * K + 9,  mask=pmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c2_10 = tl.load(C_ptr + (d_off + HALF_D) * K + 10, mask=pmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c2_11 = tl.load(C_ptr + (d_off + HALF_D) * K + 11, mask=pmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c2_12 = tl.load(C_ptr + (d_off + HALF_D) * K + 12, mask=pmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c2_13 = tl.load(C_ptr + (d_off + HALF_D) * K + 13, mask=pmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c2_14 = tl.load(C_ptr + (d_off + HALF_D) * K + 14, mask=pmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c2_15 = tl.load(C_ptr + (d_off + HALF_D) * K + 15, mask=pmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        a1_0  = tl.zeros((BLOCK_P,), tl.float32)
        a1_1  = tl.zeros((BLOCK_P,), tl.float32)
        a1_2  = tl.zeros((BLOCK_P,), tl.float32)
        a1_3  = tl.zeros((BLOCK_P,), tl.float32)
        a1_4  = tl.zeros((BLOCK_P,), tl.float32)
        a1_5  = tl.zeros((BLOCK_P,), tl.float32)
        a1_6  = tl.zeros((BLOCK_P,), tl.float32)
        a1_7  = tl.zeros((BLOCK_P,), tl.float32)
        a1_8  = tl.zeros((BLOCK_P,), tl.float32)
        a1_9  = tl.zeros((BLOCK_P,), tl.float32)
        a1_10 = tl.zeros((BLOCK_P,), tl.float32)
        a1_11 = tl.zeros((BLOCK_P,), tl.float32)
        a1_12 = tl.zeros((BLOCK_P,), tl.float32)
        a1_13 = tl.zeros((BLOCK_P,), tl.float32)
        a1_14 = tl.zeros((BLOCK_P,), tl.float32)
        a1_15 = tl.zeros((BLOCK_P,), tl.float32)

        a2_0  = tl.zeros((BLOCK_P,), tl.float32)
        a2_1  = tl.zeros((BLOCK_P,), tl.float32)
        a2_2  = tl.zeros((BLOCK_P,), tl.float32)
        a2_3  = tl.zeros((BLOCK_P,), tl.float32)
        a2_4  = tl.zeros((BLOCK_P,), tl.float32)
        a2_5  = tl.zeros((BLOCK_P,), tl.float32)
        a2_6  = tl.zeros((BLOCK_P,), tl.float32)
        a2_7  = tl.zeros((BLOCK_P,), tl.float32)
        a2_8  = tl.zeros((BLOCK_P,), tl.float32)
        a2_9  = tl.zeros((BLOCK_P,), tl.float32)
        a2_10 = tl.zeros((BLOCK_P,), tl.float32)
        a2_11 = tl.zeros((BLOCK_P,), tl.float32)
        a2_12 = tl.zeros((BLOCK_P,), tl.float32)
        a2_13 = tl.zeros((BLOCK_P,), tl.float32)
        a2_14 = tl.zeros((BLOCK_P,), tl.float32)
        a2_15 = tl.zeros((BLOCK_P,), tl.float32)

        suffix_w1 = tl.zeros((BLOCK_P,), tl.float32)
        suffix_w2 = tl.zeros((BLOCK_P,), tl.float32)

        for i in range(N):
            n = N - 1 - i

            go1_n = tl.load(GO_ptr + b * stride_gob + n * stride_gon + d_off,
                            mask=pmask, other=0.0,
                            eviction_policy="evict_first").to(tl.float32)
            go2_n = tl.load(GO_ptr + b * stride_gob + n * stride_gon + d_off + HALF_D,
                            mask=pmask, other=0.0,
                            eviction_policy="evict_first").to(tl.float32)
            inv_np1 = 1.0 / (n + 1)
            suffix_w1 += go1_n * inv_np1
            suffix_w2 += go2_n * inv_np1

            pos_n = tl.load(POS_ptr + n)
            cos_n = tl.load(FC_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)
            sin_n = tl.load(FS_ptr + pos_n * stride_fc + d_off,
                            mask=pmask, other=0.0).to(tl.float32)

            # RoPE^T: [[cos,sin],[-sin,cos]] * [suffix_w1, suffix_w2]
            d_poly1 = suffix_w1 * cos_n + suffix_w2 * sin_n
            d_poly2 = -suffix_w1 * sin_n + suffix_w2 * cos_n

            x1 = tl.load(X_ptr + b * stride_xb + n * stride_xn + d_off,
                         mask=pmask, other=0.0,
                         eviction_policy="evict_first").to(tl.float32)
            x2 = tl.load(X_ptr + b * stride_xb + n * stride_xn + d_off + HALF_D,
                         mask=pmask, other=0.0,
                         eviction_policy="evict_first").to(tl.float32)

            h1 = tl.where(x1 >= 0.0, x1, x1 * 0.01)
            h1 = tl.maximum(h1, -0.1); h1 = tl.minimum(h1, 6.0)
            h2 = tl.where(x2 >= 0.0, x2, x2 * 0.01)
            h2 = tl.maximum(h2, -0.1); h2 = tl.minimum(h2, 6.0)

            d_act1 = tl.where((x1 >= 0.0) & (x1 <= 6.0), 1.0,
                              tl.where((x1 < 0.0) & (x1 >= -10.0), 0.01, 0.0))
            d_act2 = tl.where((x2 >= 0.0) & (x2 <= 6.0), 1.0,
                              tl.where((x2 < 0.0) & (x2 >= -10.0), 0.01, 0.0))

            grad_h1 = tl.zeros((BLOCK_P,), tl.float32)
            hp1 = tl.full((BLOCK_P,), 1.0, tl.float32)
            if K > 0:
                hph1 = hp1 * h1; grad_h1 += c1_0 * hp1;          a1_0  += d_poly1 * hph1; hp1 = hph1
            if K > 1:
                hph1 = hp1 * h1; grad_h1 += c1_1 * 2.0 * hp1;   a1_1  += d_poly1 * hph1; hp1 = hph1
            if K > 2:
                hph1 = hp1 * h1; grad_h1 += c1_2 * 3.0 * hp1;   a1_2  += d_poly1 * hph1; hp1 = hph1
            if K > 3:
                hph1 = hp1 * h1; grad_h1 += c1_3 * 4.0 * hp1;   a1_3  += d_poly1 * hph1; hp1 = hph1
            if K > 4:
                hph1 = hp1 * h1; grad_h1 += c1_4 * 5.0 * hp1;   a1_4  += d_poly1 * hph1; hp1 = hph1
            if K > 5:
                hph1 = hp1 * h1; grad_h1 += c1_5 * 6.0 * hp1;   a1_5  += d_poly1 * hph1; hp1 = hph1
            if K > 6:
                hph1 = hp1 * h1; grad_h1 += c1_6 * 7.0 * hp1;   a1_6  += d_poly1 * hph1; hp1 = hph1
            if K > 7:
                hph1 = hp1 * h1; grad_h1 += c1_7 * 8.0 * hp1;   a1_7  += d_poly1 * hph1; hp1 = hph1
            if K > 8:
                hph1 = hp1 * h1; grad_h1 += c1_8 * 9.0 * hp1;   a1_8  += d_poly1 * hph1; hp1 = hph1
            if K > 9:
                hph1 = hp1 * h1; grad_h1 += c1_9 * 10.0 * hp1;  a1_9  += d_poly1 * hph1; hp1 = hph1
            if K > 10:
                hph1 = hp1 * h1; grad_h1 += c1_10 * 11.0 * hp1; a1_10 += d_poly1 * hph1; hp1 = hph1
            if K > 11:
                hph1 = hp1 * h1; grad_h1 += c1_11 * 12.0 * hp1; a1_11 += d_poly1 * hph1; hp1 = hph1
            if K > 12:
                hph1 = hp1 * h1; grad_h1 += c1_12 * 13.0 * hp1; a1_12 += d_poly1 * hph1; hp1 = hph1
            if K > 13:
                hph1 = hp1 * h1; grad_h1 += c1_13 * 14.0 * hp1; a1_13 += d_poly1 * hph1; hp1 = hph1
            if K > 14:
                hph1 = hp1 * h1; grad_h1 += c1_14 * 15.0 * hp1; a1_14 += d_poly1 * hph1; hp1 = hph1
            if K > 15:
                hph1 = hp1 * h1; grad_h1 += c1_15 * 16.0 * hp1; a1_15 += d_poly1 * hph1; hp1 = hph1

            grad_h2 = tl.zeros((BLOCK_P,), tl.float32)
            hp2 = tl.full((BLOCK_P,), 1.0, tl.float32)
            if K > 0:
                hph2 = hp2 * h2; grad_h2 += c2_0 * hp2;          a2_0  += d_poly2 * hph2; hp2 = hph2
            if K > 1:
                hph2 = hp2 * h2; grad_h2 += c2_1 * 2.0 * hp2;   a2_1  += d_poly2 * hph2; hp2 = hph2
            if K > 2:
                hph2 = hp2 * h2; grad_h2 += c2_2 * 3.0 * hp2;   a2_2  += d_poly2 * hph2; hp2 = hph2
            if K > 3:
                hph2 = hp2 * h2; grad_h2 += c2_3 * 4.0 * hp2;   a2_3  += d_poly2 * hph2; hp2 = hph2
            if K > 4:
                hph2 = hp2 * h2; grad_h2 += c2_4 * 5.0 * hp2;   a2_4  += d_poly2 * hph2; hp2 = hph2
            if K > 5:
                hph2 = hp2 * h2; grad_h2 += c2_5 * 6.0 * hp2;   a2_5  += d_poly2 * hph2; hp2 = hph2
            if K > 6:
                hph2 = hp2 * h2; grad_h2 += c2_6 * 7.0 * hp2;   a2_6  += d_poly2 * hph2; hp2 = hph2
            if K > 7:
                hph2 = hp2 * h2; grad_h2 += c2_7 * 8.0 * hp2;   a2_7  += d_poly2 * hph2; hp2 = hph2
            if K > 8:
                hph2 = hp2 * h2; grad_h2 += c2_8 * 9.0 * hp2;   a2_8  += d_poly2 * hph2; hp2 = hph2
            if K > 9:
                hph2 = hp2 * h2; grad_h2 += c2_9 * 10.0 * hp2;  a2_9  += d_poly2 * hph2; hp2 = hph2
            if K > 10:
                hph2 = hp2 * h2; grad_h2 += c2_10 * 11.0 * hp2; a2_10 += d_poly2 * hph2; hp2 = hph2
            if K > 11:
                hph2 = hp2 * h2; grad_h2 += c2_11 * 12.0 * hp2; a2_11 += d_poly2 * hph2; hp2 = hph2
            if K > 12:
                hph2 = hp2 * h2; grad_h2 += c2_12 * 13.0 * hp2; a2_12 += d_poly2 * hph2; hp2 = hph2
            if K > 13:
                hph2 = hp2 * h2; grad_h2 += c2_13 * 14.0 * hp2; a2_13 += d_poly2 * hph2; hp2 = hph2
            if K > 14:
                hph2 = hp2 * h2; grad_h2 += c2_14 * 15.0 * hp2; a2_14 += d_poly2 * hph2; hp2 = hph2
            if K > 15:
                hph2 = hp2 * h2; grad_h2 += c2_15 * 16.0 * hp2; a2_15 += d_poly2 * hph2; hp2 = hph2

            tl.store(GX_ptr + b * stride_xb + n * stride_xn + d_off,
                     d_poly1 * grad_h1 * d_act1, mask=pmask)
            tl.store(GX_ptr + b * stride_xb + n * stride_xn + d_off + HALF_D,
                     d_poly2 * grad_h2 * d_act2, mask=pmask)

        if K > 0:
            tl.atomic_add(GC_ptr + d_off * K + 0,            a1_0,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 0, a2_0,  mask=pmask)
        if K > 1:
            tl.atomic_add(GC_ptr + d_off * K + 1,            a1_1,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 1, a2_1,  mask=pmask)
        if K > 2:
            tl.atomic_add(GC_ptr + d_off * K + 2,            a1_2,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 2, a2_2,  mask=pmask)
        if K > 3:
            tl.atomic_add(GC_ptr + d_off * K + 3,            a1_3,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 3, a2_3,  mask=pmask)
        if K > 4:
            tl.atomic_add(GC_ptr + d_off * K + 4,            a1_4,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 4, a2_4,  mask=pmask)
        if K > 5:
            tl.atomic_add(GC_ptr + d_off * K + 5,            a1_5,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 5, a2_5,  mask=pmask)
        if K > 6:
            tl.atomic_add(GC_ptr + d_off * K + 6,            a1_6,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 6, a2_6,  mask=pmask)
        if K > 7:
            tl.atomic_add(GC_ptr + d_off * K + 7,            a1_7,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 7, a2_7,  mask=pmask)
        if K > 8:
            tl.atomic_add(GC_ptr + d_off * K + 8,            a1_8,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 8, a2_8,  mask=pmask)
        if K > 9:
            tl.atomic_add(GC_ptr + d_off * K + 9,            a1_9,  mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 9, a2_9,  mask=pmask)
        if K > 10:
            tl.atomic_add(GC_ptr + d_off * K + 10,            a1_10, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 10, a2_10, mask=pmask)
        if K > 11:
            tl.atomic_add(GC_ptr + d_off * K + 11,            a1_11, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 11, a2_11, mask=pmask)
        if K > 12:
            tl.atomic_add(GC_ptr + d_off * K + 12,            a1_12, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 12, a2_12, mask=pmask)
        if K > 13:
            tl.atomic_add(GC_ptr + d_off * K + 13,            a1_13, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 13, a2_13, mask=pmask)
        if K > 14:
            tl.atomic_add(GC_ptr + d_off * K + 14,            a1_14, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 14, a2_14, mask=pmask)
        if K > 15:
            tl.atomic_add(GC_ptr + d_off * K + 15,            a1_15, mask=pmask)
            tl.atomic_add(GC_ptr + (d_off + HALF_D) * K + 15, a2_15, mask=pmask)

    # -------------------------------------------------------------------------
    # autograd.Function wrappers
    # -------------------------------------------------------------------------

    class _PolyAggRopeMean(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, coeff, k, freqs_cos, freqs_sin, positions):
            B, N, D = x.shape
            HALF_D  = D // 2
            coeff_c = coeff.float().contiguous()
            fc      = freqs_cos.float().contiguous()
            fs      = freqs_sin.float().contiguous()
            pos     = positions.contiguous()
            out     = torch.empty(B, D, dtype=torch.float32, device=x.device)

            BLOCK_P = _fwd_block_p(D)
            grid = (B, triton.cdiv(HALF_D, BLOCK_P))
            _poly_agg_rope_mean_fwd[grid](
                x, coeff_c, fc, fs, pos, out,
                N, D, HALF_D,
                x.stride(0), x.stride(1),
                fc.stride(0),
                K=k, BLOCK_P=BLOCK_P,
            )

            ctx.save_for_backward(x, coeff_c, fc, fs, pos)
            ctx.k = k
            return out.unsqueeze(1).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out):
            x_saved, coeff, fc, fs, pos = ctx.saved_tensors
            k = ctx.k
            x       = x_saved.contiguous()
            B, N, D = x.shape
            HALF_D  = D // 2

            go = grad_out.squeeze(1).float().contiguous()   # (B, D)

            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            grad_c     = torch.zeros(D, k,   dtype=torch.float32, device=x.device)

            BLOCK_P = _bwd_block_p(D)
            grid = (B, triton.cdiv(HALF_D, BLOCK_P))
            _poly_agg_rope_mean_bwd[grid](
                go, x, coeff, fc, fs, pos, grad_x_buf, grad_c,
                B, N, D, HALF_D,
                x.stride(0), x.stride(1),
                fc.stride(0),
                K=k, BLOCK_P=BLOCK_P,
            )

            return grad_x_buf.to(x_saved.dtype), grad_c.to(coeff.dtype), None, None, None, None

    class _PolyAggRopeCausal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, coeff, k, freqs_cos, freqs_sin, positions):
            B, N, D = x.shape
            HALF_D  = D // 2
            coeff_c = coeff.float().contiguous()
            fc      = freqs_cos.float().contiguous()
            fs      = freqs_sin.float().contiguous()
            pos     = positions.contiguous()
            out     = torch.empty(B, N, D, dtype=torch.float32, device=x.device)

            BLOCK_P = _fwd_block_p(D)
            grid = (B, triton.cdiv(HALF_D, BLOCK_P))
            _poly_agg_rope_causal_fwd[grid](
                x, coeff_c, fc, fs, pos, out,
                N, D, HALF_D,
                x.stride(0), x.stride(1),
                out.stride(0), out.stride(1),
                fc.stride(0),
                K=k, BLOCK_P=BLOCK_P,
            )

            ctx.save_for_backward(x, coeff_c, fc, fs, pos)
            ctx.k = k
            return out.to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out):
            x_saved, coeff, fc, fs, pos = ctx.saved_tensors
            k = ctx.k
            x       = x_saved.contiguous()
            B, N, D = x.shape
            HALF_D  = D // 2

            go = grad_out.float().contiguous()   # (B, N, D)

            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            grad_c     = torch.zeros(D, k,   dtype=torch.float32, device=x.device)

            BLOCK_P = _bwd_block_p(D)
            grid = (B, triton.cdiv(HALF_D, BLOCK_P))
            _poly_agg_rope_causal_bwd[grid](
                go, x, coeff, fc, fs, pos, grad_x_buf, grad_c,
                B, N, D, HALF_D,
                x.stride(0), x.stride(1),
                go.stride(0), go.stride(1),
                fc.stride(0),
                K=k, BLOCK_P=BLOCK_P,
            )

            return grad_x_buf.to(x_saved.dtype), grad_c.to(coeff.dtype), None, None, None, None

    # -------------------------------------------------------------------------
    # Public entry points
    # -------------------------------------------------------------------------

    def poly_agg_rope_mean_triton(
        x: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fused polynomial aggregation + RoPE rotation + global mean.

        Equivalent to:
            h = poly_features(x, coeff, k)       # (B, N, D)
            h = apply_rope_1d(h, positions, ...)  # (B, N, D)
            return h.mean(dim=1, keepdim=True)    # (B, 1, D)

        Args:
            x         : (B, N, D) input — D must be even
            coeff     : (D, K) polynomial coefficients
            k         : polynomial degree (≤ 16)
            freqs_cos : (S, D//2) precomputed cosines (S ≥ max position)
            freqs_sin : (S, D//2) precomputed sines
            positions : (N,) int64 position indices; defaults to 0..N-1

        Returns:
            (B, 1, D) rotated-and-aggregated features
        """
        if k > 16:
            raise NotImplementedError(f"k ≤ 16 supported (got {k})")
        B, N, D = x.shape
        if D % 2 != 0:
            raise ValueError(f"D must be even for RoPE (got {D})")
        if positions is None:
            positions = torch.arange(N, device=x.device, dtype=torch.int64)
        return _PolyAggRopeMean.apply(x, coeff, k, freqs_cos, freqs_sin, positions)

    def poly_agg_rope_causal_triton(
        x: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        positions: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fused polynomial aggregation + RoPE rotation + causal prefix mean.

        Equivalent to:
            h = poly_features(x, coeff, k)        # (B, N, D)
            h = apply_rope_1d(h, positions, ...)   # (B, N, D)
            return causal_mean(h)                  # (B, N, D)

        Args:
            x         : (B, N, D) input — D must be even
            coeff     : (D, K) polynomial coefficients
            k         : polynomial degree (≤ 16)
            freqs_cos : (S, D//2) precomputed cosines
            freqs_sin : (S, D//2) precomputed sines
            positions : (N,) int64 position indices; defaults to 0..N-1

        Returns:
            (B, N, D) causally-aggregated rotated features
        """
        if k > 16:
            raise NotImplementedError(f"k ≤ 16 supported (got {k})")
        B, N, D = x.shape
        if D % 2 != 0:
            raise ValueError(f"D must be even for RoPE (got {D})")
        if positions is None:
            positions = torch.arange(N, device=x.device, dtype=torch.int64)
        return _PolyAggRopeCausal.apply(x, coeff, k, freqs_cos, freqs_sin, positions)
