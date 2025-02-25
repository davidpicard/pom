import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.suppress_errors = True

@torch.compile
def gelu(x: torch.Tensor):
    return F.gelu(x)

@torch.compile
def po2(x: torch.Tensor):
    h1, h2 = gelu(x).chunk(2, dim=-1)
    h2 = h2 * h1
    return torch.cat([h1, h2], dim=-1)

@torch.compile
def po3(x: torch.Tensor):
    h1, h2, h3 = gelu(x).chunk(3, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    return torch.cat([h1, h2, h3], dim=-1)

@torch.compile
def po4(x: torch.Tensor):
    h1, h2, h3, h4 = gelu(x).chunk(4, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    h4 = h4 * h3
    return torch.cat([h1, h2, h3, h4], dim=-1)


def mask_mixer(h, mask):
    return (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True)/(1.e-7 + mask.unsqueeze(-1).sum(dim=1, keepdims=True))


def full_mask_mixer(h, mask):
    mask = mask.type(h.dtype)
    h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
    h = h / (1.e-7 + mask.sum(dim=2, keepdims=True))
    return h

def polynomial_aggregation_(x: torch.Tensor, k: int, mask=None):
    if k == 2:
        h = po2(x)
    elif k == 3:
        h = po3(x)
    elif k == 4:
        h = po4(x)
    else:
        h = list(gelu(x).chunk(k, dim=-1))
        for i in range(1, k):
            h[i] = h[i] * h[i-1]
        h = torch.cat(h, dim=-1)
    if mask is None:
        h = h.mean(dim=1, keepdims=True)
    else:
        if mask.dim()==2:
            h = mask_mixer(h, mask.to(h.device))
        elif mask.dim() ==3:
            h = full_mask_mixer(h, mask.to(h.device))
        else:
            raise Exception('unsupported dim for mask (should be 2,3 or None)')
    return h

@torch.compile
def polynomial_selection_(x: torch.Tensor, h: torch.Tensor):
    return F.sigmoid(x) * h


def pom(xq: torch.Tensor, xc: torch.Tensor, k: int, mask=None):
    """
    pom function

    This function implements the polynomial mixer operation.
            Args:
            xq (torch.Tensor): The query input tensor.
            xc (torch.Tensor): The context input tensor.
            k (int): The order of the polynomial.
            mask (torch.Tensor, optional): The mask tensor for attention.
    """
    h = polynomial_aggregation_(xc, k, mask)
    o = polynomial_selection_(xq, h)
    return o


class PoM(nn.Module):
    """
    PoM (Polynomial Mixer) Module

    This class implements the PoM (Polynomial Mixer) module, which is a custom neural network layer
    designed for capturing higher-order interactions between input features. It consists of three
    linear projections and a custom PoM operation.

    Attributes:
        dim (int): The dimensionality of the input features.
        order (int): The order of the moments to be captured.
        order_expand (int): The expansion factor for the order.
        po_proj (nn.Linear): Linear projection for the polynomials.
        se_proj (nn.Linear): Linear projection for the selection.
        ag_proj (nn.Linear): Linear projection for aggregating the results.
        pom (callable): The custom polynomial mixer operation function.

    Args:
        dim (int): The dimensionality of the input features.
        degree (int): The degree of the polynomial to be captured.
        expand (int): The expansion factor for the order.
        bias (bool, optional): Whether to include a bias term in the linear projections. Default is True.
    """
    def __init__(self, dim, degree, expand, bias=True):
        super().__init__()
        self.dim = dim
        self.order = degree
        self.order_expand = expand

        self.po_proj = nn.Linear(dim, degree * expand * dim, bias=bias)
        self.se_proj = nn.Linear(dim, degree * expand * dim, bias=bias)
        self.ag_proj = nn.Linear(degree * expand * dim, dim, bias=bias)
        self.pom = pom

    def forward(self, xq, xc=None, mask=None):
        """
        Forward pass of the PoM module.

        Args:
            xq (torch.Tensor): The query input tensor of size batch x n_tokens x dimension.
            xc (torch.Tensor, optional): The context input tensor. If None, self-attention is performed.
            mask (torch.Tensor, optional): The mask tensor for attention.

        Returns:
            torch.Tensor: The output tensor after applying the PoM operation.
        """
        if xc is None:
            xc = xq # self attention

        s = self.se_proj(xq)
        h = self.po_proj(xc)
        sh = self.pom(s, h, self.order, mask)

        # aggregation
        return self.ag_proj(sh)

    def state_forward(self, xq, xc, state=None):
        if xc is None:
            xc = xq # self attention

        s = self.se_proj(xq)
        xc = self.po_proj(xc)
        h_current = polynomial_aggregation_(xc, self.order)
        n_current = h_current.shape[1]

        if state is not None:
            h_past = state['h']
            n_past = state['n']
            h = (n_past * h_past + n_current * h_current) / (n_past+n_current)
        else:
            h = h_current
            n_past = 0

        new_state = {'h': h, 'n': n_past + n_current}

        sh = polynomial_selection_(s, h)
        return self.ag_proj(sh), new_state
