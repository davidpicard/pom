import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

try:
    from .pom_triton import poly_agg_mean_triton, TRITON_AVAILABLE
except ImportError:
    print("Triton not available, PoM might be slow")
    TRITON_AVAILABLE = False

try:
    from .pom_triton_causal import poly_agg_causal_triton, TRITON_CAUSAL_AVAILABLE
except ImportError:
    TRITON_CAUSAL_AVAILABLE = False

try:
    from .pom_triton_masked import poly_agg_masked_triton, TRITON_MASKED_AVAILABLE
except ImportError:
    TRITON_MASKED_AVAILABLE = False


# =============================================================================
# Core activation
# =============================================================================

def pom_activation(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.leaky_relu(x, 0.01, inplace=False), min=-0.1, max=6)


# =============================================================================
# Masking and Aggregation
# =============================================================================

def mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over seq_len. mask: (B, N) → output: (B, 1, D)."""
    m = mask.unsqueeze(-1)
    return (h * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)


def full_mask_mixer(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-attention masked mean. mask: (B, M, N) → output: (B, M, D)."""
    mask = mask.to(h.dtype)
    h = torch.einsum('bnd,bmn->bmd', h, mask)
    return h / mask.sum(dim=2, keepdim=True)


# =============================================================================
# Polynomial Aggregation and Selection
# =============================================================================

def polynomial_aggregation_(
    x: torch.Tensor,
    coeff: torch.Tensor,
    k: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Polynomial aggregation over the sequence dimension.

    Args:
        x:     (B, N, D) input
        coeff: (D, K) polynomial coefficients
        k:     polynomial degree
        mask:  None, "causal", (B, N) tensor for masked mean, or (B, M, N) for cross-attention

    Returns:
        (B, 1, D) for mask=None or 2-D mask; (B, N, D) for "causal"; (B, M, D) for 3-D mask
    """
    # Fused path: no-mask CUDA → single Triton kernel (no (B,N,D,K) intermediate)
    if mask is None and TRITON_AVAILABLE and x.is_cuda:
        return poly_agg_mean_triton(x, coeff, k)

    # Fused path: causal CUDA → single Triton kernel (no N×N mask materialised)
    if mask == "causal" and TRITON_CAUSAL_AVAILABLE and x.is_cuda:
        return poly_agg_causal_triton(x, coeff, k)

    # Fused path: 1-D mask CUDA → single Triton kernel (no (B,N,D,K) intermediate)
    if (isinstance(mask, torch.Tensor) and mask.dim() == 2
            and TRITON_MASKED_AVAILABLE and x.is_cuda):
        return poly_agg_masked_triton(x, mask, coeff, k)

    # PyTorch fallback: compute polynomial powers iteratively to avoid h**i overhead
    h = pom_activation(x).unsqueeze(-1)  # (B, N, D, 1)
    hp, powers = h, [h]
    for _ in range(k - 1):
        hp = hp * h
        powers.append(hp)
    h = (torch.cat(powers, dim=-1) * coeff).sum(-1)  # (B, N, D)

    if mask is None:
        return h.mean(dim=1, keepdim=True)
    if mask == "causal":
        B, N, _ = h.shape
        causal_mask = torch.tril(torch.ones(N, N, device=h.device, dtype=h.dtype))
        return full_mask_mixer(h, causal_mask.unsqueeze(0).expand(B, -1, -1))
    if mask.dim() == 2:
        return mask_mixer(h, mask.to(h.device))
    if mask.dim() == 3:
        return full_mask_mixer(h, mask.to(h.device))
    raise ValueError(f'Unsupported mask: expected None, "causal", or a 2/3-D tensor.')


def polynomial_selection_(s: torch.Tensor, h: torch.Tensor, n_sel_heads: int) -> torch.Tensor:
    """Gated selection of aggregated polynomial features.

    Args:
        s: (B, T, D) gating signal (n_sel_heads=1) or (B, T, n_sel_heads)
        h: (B, G, D) aggregated context  (G is 1 or T)

    Returns:
        (B, max(G,T), D)
    """
    b, g, dh = h.shape
    t = s.shape[1]
    assert g == 1 or t == 1 or g == t, f"incompatible shapes: g={g} t={t}"
    if n_sel_heads <= 1:
        # n_sel_heads=0 or 1: s has the same channel dim as h → element-wise gate
        return (s * h).view(b, max(g, t), dh)
    # Multi-head: s is (B, T, n_sel_heads); broadcast over head_dim
    s = s.unsqueeze(-1)
    h = h.view(b, g, n_sel_heads, dh // n_sel_heads)
    return (s * h).view(b, max(g, t), dh)


def pom(
    xq: torch.Tensor,
    xc: torch.Tensor,
    coeff: torch.Tensor,
    k: int,
    n_sel_heads: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Polynomial Mixer: aggregate context polynomially, gate with query."""
    h = polynomial_aggregation_(xc, coeff, k, mask)
    return polynomial_selection_(xq, h, n_sel_heads)


# =============================================================================
# PoM Module
# =============================================================================

class PoM(nn.Module):
    """Polynomial Mixer (PoM) — linear-complexity alternative to self-attention.

    Aggregates context tokens via a polynomial expansion and weighted mean,
    then gates the result with a learned selection signal from the query.

    Args:
        dim:         Input/output feature dimension.
        degree:      Polynomial degree (number of powers to include).
        expand:      Channel expansion factor for the polynomial projection.
        n_groups:    Groups for the polynomial projection (>1 → grouped Conv1d).
        n_sel_heads: Selection heads (1 → scalar gating; >1 → multi-head gating).
        bias:        Add bias to linear projections.
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        expand: int,
        n_groups: int,
        n_sel_heads: int,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.order = degree
        self.order_expand = expand
        self.n_groups = n_groups
        self.n_sel_heads = n_sel_heads
        assert dim % n_groups == 0, "dim must be divisible by n_groups"
        assert n_sel_heads <= 1 or dim * expand % n_sel_heads == 0, \
            "dim * expand must be divisible by n_sel_heads"
        self.head_dim = dim * expand // max(n_sel_heads, 1)

        self._po_dim = expand * dim
        self._se_dim = n_sel_heads if n_sel_heads > 1 else expand * dim

        if n_groups > 1:
            # Grouped projection must stay as Conv1d; keep se_proj separate.
            self.po_proj = nn.Conv1d(dim, expand * dim, kernel_size=1, bias=bias, groups=n_groups)
            self.se_proj = nn.Linear(dim, self._se_dim, bias=True)
        else:
            # Fuse po_proj and se_proj into a single GEMM (qc_proj) with a
            # standalone bias for the selection branch (se_bias).
            self.qc_proj = nn.Linear(dim, self._po_dim + self._se_dim, bias=False)
            self.se_bias = nn.Parameter(torch.zeros(self._se_dim))

        self.po_coeff = nn.Parameter(torch.randn(dim * expand, degree).clamp(-0.001, 0.001))
        self.ag_proj = nn.Linear(expand * dim, dim, bias=bias)

    def _get_h_s(
        self, xq: torch.Tensor, xc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (h, s): projected context and hardsigmoid gating signal.

        For n_groups == 1 and self-mixing (xq is xc), a single fused GEMM
        produces both h and s, halving the projection cost.
        """
        if self.n_groups > 1:
            h = self.po_proj(xc.transpose(1, 2)).transpose(1, 2)
            s = F.hardsigmoid(self.se_proj(xq), inplace=True)
        elif xq is xc:
            out = self.qc_proj(xq)
            h = out[..., :self._po_dim]
            s = F.hardsigmoid(out[..., self._po_dim:] + self.se_bias, inplace=True)
        else:
            w = self.qc_proj.weight
            h = F.linear(xc, w[:self._po_dim])
            s = F.hardsigmoid(
                F.linear(xq, w[self._po_dim:]) + self.se_bias, inplace=True
            )
        return h, s

    def forward(
        self,
        xq: torch.Tensor,
        xc: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            xq:   (B, T, D) query tokens
            xc:   (B, N, D) context tokens; if None, self-mixing is performed
            mask: optional attention mask
        """
        if xc is None:
            xc = xq
        h, s = self._get_h_s(xq, xc)
        sh = pom(s, h, self.po_coeff, self.order, self.n_sel_heads, mask)
        return self.ag_proj(sh)

    def state_forward(
        self,
        xq: torch.Tensor,
        xc: Optional[torch.Tensor] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Incremental forward with running weighted-mean state.

        Args:
            xq:    (B, T, D) query tokens
            xc:    (B, N, D) context tokens; if None, self-mixing is performed
            state: {'h': running mean tensor, 'n': token count} or None

        Returns:
            (output, new_state)
        """
        if xc is None:
            xc = xq
        h_raw, s = self._get_h_s(xq, xc)
        h_current = polynomial_aggregation_(h_raw, self.po_coeff, self.order)
        n_current = h_current.shape[1]

        if state is not None:
            n_past = state['n']
            h = (n_past * state['h'] + n_current * h_current) / (n_past + n_current)
        else:
            h, n_past = h_current, 0

        sh = polynomial_selection_(s, h, self.n_sel_heads)
        return self.ag_proj(sh), {'h': h, 'n': n_past + n_current}

    @torch.no_grad()
    def ar_forward(
        self,
        xq: torch.Tensor,
        state: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Autoregressive forward (no gradient tracking)."""
        B, T, D = xq.shape
        h, s = self._get_h_s(xq, xq)
        sh = polynomial_selection_(s, h, self.n_sel_heads)
        new_state = {'max_len': state['max_len'], 'h': h, 'n': state['n'] + T}
        return self.ag_proj(sh), new_state

    def reset(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state['h'] = 0.
        state['n'] = 0
        return state
