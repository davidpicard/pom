"""Triton-accelerated 1-D mask kernel for PoM (Polynomial Mixer).

For a (B, N) float mask, output position 0 aggregates the masked tokens:

  out[b, d] = sum_{n: mask[b,n]!=0} mask[b,n] * poly(act(x[b,n,d]))
              ---------------------------------------------------
                          sum_{n} mask[b,n]

which is a weighted mean (or binary mean when mask values are 0/1).
Output shape: (B, 1, D) — same as the no-mask case.

This fuses activation + polynomial expansion + masked sum + normalisation
into a single GPU pass, eliminating the (B, N, D, K) intermediate that the
PyTorch fallback materialises.

Forward kernel
--------------
Grid: (B, ceil(D/BLOCK_D))
Each program accumulates a scalar `cnt` and a (BLOCK_D,) `acc` in one loop
over N.  The mask value m[b, n] is a scalar load; it gates both `acc` and
`cnt`.

Backward kernel
---------------
Grid: (B, ceil(D/BLOCK_D))
Gradients:

  grad_x[b, n, d] = go[b,d] * (m[b,n] / cnt[b]) * d_poly/d_h * d_act/d_x
  grad_coeff[d,k]  = sum_{b,n} go[b,d] * (m[b,n] / cnt[b]) * h[b,n,d]^(k+1)

Implementation uses two sequential loops over N within the same program:
  Loop 1 (mask-only): compute cnt[b] from N scalar mask loads.
  Loop 2 (mask + x):  compute grad_x and accumulate grad_coeff.
The first loop is very cheap (N scalar loads vs the N*BLOCK_D x-loads of
loop 2) and avoids the complexity of saving cnt across forward/backward.

The `go_eff = go / cnt` vector is precomputed once after loop 1 and reused
in loop 2, enabling the same power-sharing pattern as the other kernels.

Exposed API
-----------
TRITON_MASKED_AVAILABLE : bool
poly_agg_masked_triton(x, mask, coeff, k) -> Tensor  (B, 1, D)
"""
import os
import torch

try:
    import triton
    import triton.language as tl
    TRITON_MASKED_AVAILABLE = not os.environ.get("POM_DISABLE_TRITON", "")
except ImportError:
    TRITON_MASKED_AVAILABLE = False


if TRITON_MASKED_AVAILABLE:

    # -------------------------------------------------------------------------
    # BLOCK_D heuristics – identical policy to the other kernels.
    # -------------------------------------------------------------------------

    def _fwd_block_d(D: int) -> int:
        return min(256, 1 << (D - 1).bit_length())

    def _bwd_block_d(D: int) -> int:
        return min(128, 1 << (D - 1).bit_length())

    # -------------------------------------------------------------------------
    # Forward kernel
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_masked_fwd(
        X_ptr,          # (B, N, D) – input, any dtype
        M_ptr,          # (B, N)    – mask, any dtype (treated as float)
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        O_ptr,          # (B, D)    – output, fp32
        N,              # sequence length  (runtime)
        D,              # feature dim      (runtime)
        stride_xb,      # X.stride(0)
        stride_xn,      # X.stride(1)
        stride_mb,      # M.stride(0)
        stride_mn,      # M.stride(1)
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload coeff[d_off, 0..K-1] into registers once.
        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        cnt = 0.0   # scalar: sum of mask weights

        for n in range(N):
            # Scalar mask load — same value for all d in this block.
            m_val = tl.load(M_ptr + b * stride_mb + n * stride_mn).to(tl.float32)

            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            # pom_activation
            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            poly = tl.zeros((BLOCK_D,), dtype=tl.float32)
            hp   = h
            if K > 0:
                poly += c0 * hp
            if K > 1:
                hp *= h; poly += c1 * hp
            if K > 2:
                hp *= h; poly += c2 * hp
            if K > 3:
                hp *= h; poly += c3 * hp
            if K > 4:
                hp *= h; poly += c4 * hp
            if K > 5:
                hp *= h; poly += c5 * hp
            if K > 6:
                hp *= h; poly += c6 * hp
            if K > 7:
                hp *= h; poly += c7 * hp
            if K > 8:
                hp *= h; poly += c8 * hp
            if K > 9:
                hp *= h; poly += c9 * hp
            if K > 10:
                hp *= h; poly += c10 * hp
            if K > 11:
                hp *= h; poly += c11 * hp
            if K > 12:
                hp *= h; poly += c12 * hp
            if K > 13:
                hp *= h; poly += c13 * hp
            if K > 14:
                hp *= h; poly += c14 * hp
            if K > 15:
                hp *= h; poly += c15 * hp

            acc += m_val * poly
            cnt += m_val

        # Safe division: all-masked-out rows produce zero output.
        inv_cnt = tl.where(cnt > 0.0, 1.0 / cnt, 0.0)
        tl.store(O_ptr + b * D + d_off, acc * inv_cnt, mask=dmask)

    # -------------------------------------------------------------------------
    # Backward kernel
    #
    # Two-pass design within a single program:
    #   Pass 1: scan M to accumulate cnt (N scalar loads — very cheap).
    #   Pass 2: process X and M together, using the precomputed inv_cnt.
    #
    # go_eff = go / cnt  is computed once; grad_x and a[k] both scale by it.
    # This matches the power-sharing structure of the other backward kernels.
    # -------------------------------------------------------------------------

    @triton.jit
    def _poly_agg_masked_bwd(
        GO_ptr,         # (B, D)    – upstream gradient, fp32
        X_ptr,          # (B, N, D) – saved input
        M_ptr,          # (B, N)    – mask
        C_ptr,          # (D, K)    – polynomial coefficients, fp32
        GX_ptr,         # (B, N, D) – grad w.r.t. X, fp32
        GC_ptr,         # (D, K)    – grad w.r.t. coeff, fp32 (zero-init, atomic)
        B, N, D,
        stride_xb, stride_xn,
        stride_mb, stride_mn,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        b     = tl.program_id(0)
        d_blk = tl.program_id(1)
        d_off = d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        dmask = d_off < D

        # Preload go[b, d] and coefficients (constant over both N loops).
        go    = tl.load(GO_ptr + b * D + d_off, mask=dmask, other=0.0).to(tl.float32)

        c0  = tl.load(C_ptr + d_off * K + 0,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 0  else 0.0
        c1  = tl.load(C_ptr + d_off * K + 1,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 1  else 0.0
        c2  = tl.load(C_ptr + d_off * K + 2,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 2  else 0.0
        c3  = tl.load(C_ptr + d_off * K + 3,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 3  else 0.0
        c4  = tl.load(C_ptr + d_off * K + 4,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 4  else 0.0
        c5  = tl.load(C_ptr + d_off * K + 5,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 5  else 0.0
        c6  = tl.load(C_ptr + d_off * K + 6,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 6  else 0.0
        c7  = tl.load(C_ptr + d_off * K + 7,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 7  else 0.0
        c8  = tl.load(C_ptr + d_off * K + 8,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 8  else 0.0
        c9  = tl.load(C_ptr + d_off * K + 9,  mask=dmask, other=0.0, cache_modifier=".ca") if K > 9  else 0.0
        c10 = tl.load(C_ptr + d_off * K + 10, mask=dmask, other=0.0, cache_modifier=".ca") if K > 10 else 0.0
        c11 = tl.load(C_ptr + d_off * K + 11, mask=dmask, other=0.0, cache_modifier=".ca") if K > 11 else 0.0
        c12 = tl.load(C_ptr + d_off * K + 12, mask=dmask, other=0.0, cache_modifier=".ca") if K > 12 else 0.0
        c13 = tl.load(C_ptr + d_off * K + 13, mask=dmask, other=0.0, cache_modifier=".ca") if K > 13 else 0.0
        c14 = tl.load(C_ptr + d_off * K + 14, mask=dmask, other=0.0, cache_modifier=".ca") if K > 14 else 0.0
        c15 = tl.load(C_ptr + d_off * K + 15, mask=dmask, other=0.0, cache_modifier=".ca") if K > 15 else 0.0

        # ---- Pass 1: count masked tokens (N scalar loads) --------------------
        cnt = 0.0
        for n in range(N):
            m_val = tl.load(M_ptr + b * stride_mb + n * stride_mn).to(tl.float32)
            cnt += m_val

        inv_cnt  = tl.where(cnt > 0.0, 1.0 / cnt, 0.0)
        # Effective upstream gradient: go / cnt  (scaled once, reused per step)
        go_eff   = go * inv_cnt      # (BLOCK_D,)

        # ---- grad_coeff accumulators -----------------------------------------
        a0  = tl.zeros((BLOCK_D,), tl.float32)
        a1  = tl.zeros((BLOCK_D,), tl.float32)
        a2  = tl.zeros((BLOCK_D,), tl.float32)
        a3  = tl.zeros((BLOCK_D,), tl.float32)
        a4  = tl.zeros((BLOCK_D,), tl.float32)
        a5  = tl.zeros((BLOCK_D,), tl.float32)
        a6  = tl.zeros((BLOCK_D,), tl.float32)
        a7  = tl.zeros((BLOCK_D,), tl.float32)
        a8  = tl.zeros((BLOCK_D,), tl.float32)
        a9  = tl.zeros((BLOCK_D,), tl.float32)
        a10 = tl.zeros((BLOCK_D,), tl.float32)
        a11 = tl.zeros((BLOCK_D,), tl.float32)
        a12 = tl.zeros((BLOCK_D,), tl.float32)
        a13 = tl.zeros((BLOCK_D,), tl.float32)
        a14 = tl.zeros((BLOCK_D,), tl.float32)
        a15 = tl.zeros((BLOCK_D,), tl.float32)

        # ---- Pass 2: compute grad_x and accumulate grad_coeff ----------------
        for n in range(N):
            m_val = tl.load(M_ptr + b * stride_mb + n * stride_mn).to(tl.float32)

            x = tl.load(
                X_ptr + b * stride_xb + n * stride_xn + d_off,
                mask=dmask, other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            h = tl.where(x >= 0.0, x, x * 0.01)
            h = tl.maximum(h, -0.1)
            h = tl.minimum(h,  6.0)

            d_act = tl.where(
                (x >= 0.0) & (x <= 6.0), 1.0,
                tl.where((x < 0.0) & (x >= -10.0), 0.01, 0.0),
            )

            # Power-sharing: same pattern as the other backward kernels.
            # go_eff_m = m_val * go_eff  scales both grad_x and coeff accum.
            go_eff_m = m_val * go_eff   # (BLOCK_D,)

            grad_h = tl.zeros((BLOCK_D,), tl.float32)
            hp     = tl.full((BLOCK_D,), 1.0, tl.float32)   # h^0

            if K > 0:
                hph = hp * h
                grad_h += c0 * hp;          a0  += go_eff_m * hph;  hp = hph
            if K > 1:
                hph = hp * h
                grad_h += c1 * 2.0 * hp;   a1  += go_eff_m * hph;  hp = hph
            if K > 2:
                hph = hp * h
                grad_h += c2 * 3.0 * hp;   a2  += go_eff_m * hph;  hp = hph
            if K > 3:
                hph = hp * h
                grad_h += c3 * 4.0 * hp;   a3  += go_eff_m * hph;  hp = hph
            if K > 4:
                hph = hp * h
                grad_h += c4 * 5.0 * hp;   a4  += go_eff_m * hph;  hp = hph
            if K > 5:
                hph = hp * h
                grad_h += c5 * 6.0 * hp;   a5  += go_eff_m * hph;  hp = hph
            if K > 6:
                hph = hp * h
                grad_h += c6 * 7.0 * hp;   a6  += go_eff_m * hph;  hp = hph
            if K > 7:
                hph = hp * h
                grad_h += c7 * 8.0 * hp;   a7  += go_eff_m * hph;  hp = hph
            if K > 8:
                hph = hp * h
                grad_h += c8 * 9.0 * hp;   a8  += go_eff_m * hph;  hp = hph
            if K > 9:
                hph = hp * h
                grad_h += c9 * 10.0 * hp;  a9  += go_eff_m * hph;  hp = hph
            if K > 10:
                hph = hp * h
                grad_h += c10 * 11.0 * hp; a10 += go_eff_m * hph;  hp = hph
            if K > 11:
                hph = hp * h
                grad_h += c11 * 12.0 * hp; a11 += go_eff_m * hph;  hp = hph
            if K > 12:
                hph = hp * h
                grad_h += c12 * 13.0 * hp; a12 += go_eff_m * hph;  hp = hph
            if K > 13:
                hph = hp * h
                grad_h += c13 * 14.0 * hp; a13 += go_eff_m * hph;  hp = hph
            if K > 14:
                hph = hp * h
                grad_h += c14 * 15.0 * hp; a14 += go_eff_m * hph;  hp = hph
            if K > 15:
                hph = hp * h
                grad_h += c15 * 16.0 * hp; a15 += go_eff_m * hph;  hp = hph

            tl.store(
                GX_ptr + b * stride_xb + n * stride_xn + d_off,
                go_eff_m * grad_h * d_act,
                mask=dmask,
            )

        # Atomic-add partial grad_coeff sums into GC (reduces across B).
        if K > 0:  tl.atomic_add(GC_ptr + d_off * K + 0,  a0,  mask=dmask)
        if K > 1:  tl.atomic_add(GC_ptr + d_off * K + 1,  a1,  mask=dmask)
        if K > 2:  tl.atomic_add(GC_ptr + d_off * K + 2,  a2,  mask=dmask)
        if K > 3:  tl.atomic_add(GC_ptr + d_off * K + 3,  a3,  mask=dmask)
        if K > 4:  tl.atomic_add(GC_ptr + d_off * K + 4,  a4,  mask=dmask)
        if K > 5:  tl.atomic_add(GC_ptr + d_off * K + 5,  a5,  mask=dmask)
        if K > 6:  tl.atomic_add(GC_ptr + d_off * K + 6,  a6,  mask=dmask)
        if K > 7:  tl.atomic_add(GC_ptr + d_off * K + 7,  a7,  mask=dmask)
        if K > 8:  tl.atomic_add(GC_ptr + d_off * K + 8,  a8,  mask=dmask)
        if K > 9:  tl.atomic_add(GC_ptr + d_off * K + 9,  a9,  mask=dmask)
        if K > 10: tl.atomic_add(GC_ptr + d_off * K + 10, a10, mask=dmask)
        if K > 11: tl.atomic_add(GC_ptr + d_off * K + 11, a11, mask=dmask)
        if K > 12: tl.atomic_add(GC_ptr + d_off * K + 12, a12, mask=dmask)
        if K > 13: tl.atomic_add(GC_ptr + d_off * K + 13, a13, mask=dmask)
        if K > 14: tl.atomic_add(GC_ptr + d_off * K + 14, a14, mask=dmask)
        if K > 15: tl.atomic_add(GC_ptr + d_off * K + 15, a15, mask=dmask)

    # -------------------------------------------------------------------------
    # autograd.Function wrapper
    # -------------------------------------------------------------------------

    class _PolyAggMasked(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, mask: torch.Tensor,
                    coeff: torch.Tensor, k: int):
            B, N, D = x.shape
            coeff_c = coeff.float().contiguous()
            mask_c  = mask.float().contiguous()
            out     = torch.empty(B, D, dtype=torch.float32, device=x.device)

            BLOCK_D = _fwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_masked_fwd[grid](
                x, mask_c, coeff_c, out,
                N, D,
                x.stride(0), x.stride(1),
                mask_c.stride(0), mask_c.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            ctx.save_for_backward(x, mask_c, coeff_c)
            ctx.k = k
            return out.unsqueeze(1).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x_saved, mask_c, coeff = ctx.saved_tensors
            k = ctx.k
            x       = x_saved.contiguous()
            B, N, D = x.shape

            go = grad_out.squeeze(1).float().contiguous()   # (B, D)

            grad_x_buf = torch.empty(B, N, D, dtype=torch.float32, device=x.device)
            grad_c     = torch.zeros(D, k, dtype=torch.float32, device=x.device)

            BLOCK_D = _bwd_block_d(D)
            grid = (B, triton.cdiv(D, BLOCK_D))
            _poly_agg_masked_bwd[grid](
                go, x, mask_c, coeff, grad_x_buf, grad_c,
                B, N, D,
                x.stride(0), x.stride(1),
                mask_c.stride(0), mask_c.stride(1),
                K=k, BLOCK_D=BLOCK_D,
            )

            return grad_x_buf.to(x_saved.dtype), None, grad_c.to(coeff.dtype), None

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def poly_agg_masked_triton(
        x: torch.Tensor,
        mask: torch.Tensor,
        coeff: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Fused masked polynomial aggregation.

        Computes the mask-weighted polynomial mean over the sequence:
          out[b, 0, d] = (sum_n mask[b,n] * poly(act(x[b,n,d])))
                         / (sum_n mask[b,n])

        Args:
            x     : (B, N, D) input tensor
            mask  : (B, N)    float mask (0/1 or continuous weights)
            coeff : (D, K)    polynomial coefficients
            k     : polynomial degree (≤ 16)

        Returns:
            (B, 1, D) mask-weighted-mean aggregated polynomial features
        """
        if k > 16:
            raise NotImplementedError(
                f"Triton masked kernel supports k ≤ 16 (got {k}). "
                "Extend the c0..c15 / a0..a15 pattern or use the PyTorch fallback."
            )
        return _PolyAggMasked.apply(x, mask, coeff, k)
