import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.modules.convolution import causal_conv1d, causal_conv1d_update

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d import causal_conv1d_update as causal_conv1d_update_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update_cuda = None


class FadingCausalConv(ShortConvolution):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        backend: str | None = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
            backend=backend,
            device=device,
            dtype=dtype,
            **kwargs
        )
        self.decay = nn.Parameter(torch.randn(hidden_size))
        self.register_buffer('exponential', torch.arange(kernel_size - 1, -1, -1))

    def forward(    # type: ignore
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            chunk_indices (Optional[torch.LongTensor]):
                Chunk indices for variable-length sequences. Default: `None`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # in decoding phase, the cache (if provided) is updated inplace
        if B * T == N:
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return y, cache

        # cuda backend do not support:
        # 1. both `cu_seqlens` and `cache` being provided
        # 2. both `cu_seqlens` and `output_final_state` being provided
        if self.backend == 'cuda' and (
            (cu_seqlens is not None and cache is not None) or
            (cu_seqlens is not None and output_final_state)
        ):
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2,
            )
            self.backend = 'triton'     # type: ignore

        decay = F.sigmoid(self.decay)
        decay = decay.unsqueeze(dim=-1) ** self.exponential.unsqueeze(dim=0)    # type: ignore
        weight = rearrange(self.weight, "d 1 w -> d w") * decay

        return causal_conv1d(
            x=x,
            weight=weight,
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **kwargs,
        )

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cache: torch.Tensor | None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        decay = F.sigmoid(self.decay)
        decay = decay.unsqueeze(dim=-1) ** self.exponential.unsqueeze(dim=0)    # type: ignore
        weight = rearrange(self.weight, "d 1 w -> d w") * decay
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)
        # NOTE: we follow the fast mode that updates the cache in-place
        if self.backend == 'triton':
            return causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=weight,
                bias=self.bias,
                activation=self.activation,
            )

        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        # equivalent to:
        # cache.copy_(cache.roll(shifts=-1, dims=-1))
        # cache[:, :, -1] = x
        # y = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
        y = causal_conv1d_update_cuda(
            x=x,
            conv_state=cache,
            weight=weight,
            bias=self.bias,
            activation=self.activation,
        )
        y = y.view(shape)
        if residual is not None:
            y.add_(residual)
        return y, cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / (x.norm(dim=-1, keepdim=True) + 1e-6)).to(x)

class STETopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, dim=-1):
        topk_val, topk_idx = torch.topk(x, k, dim=dim)
        mask = torch.zeros_like(x).scatter_(dim, topk_idx, 1.0)
        return x * mask

    @staticmethod
    def backward(ctx, grad_output):    # type: ignore
        return grad_output, None, None

def ste_topk(x, k, dim=-1):
    return STETopK.apply(x, k, dim)

def _sep_fwd_func(x, weight, bias):
    u = torch.einsum('bthd,HDd->bthD', x, weight) + bias
    # u = F.linear(x, weight, bias)
    # s = ste_topk(u, k=3, dim=-1)
    # s = F.sigmoid(u)
    s = F.relu(u)
    return s

def lin_attn_fwd(q, k, v):
    A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) 
    A_qk = torch.tril(A_qk)        
    y = torch.einsum("bhnm,bhme->bhne", A_qk, v)
    # z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
    # y = y * z[..., None]
    # y = rearrange(y, 'b h l d -> b l (h d)')
    return y.transpose(1, 2)

class SepLA(nn.Module):
    def __init__(self, dim: int, expand_k: int = 1, num_heads: int = 16, layer_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.dim = dim    # type: ignore
        self.expand_k = expand_k    # type: ignore
        self.num_heads = num_heads    # type: ignore
        self.layer_idx = layer_idx    # type: ignore
        self.head_dim = self.dim // self.num_heads    # type: ignore
        self.enable_cont_loss = (self.layer_idx == 0)    # type: ignore
        # self.enable_cont_loss = True    # type: ignore
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.sep_proj = nn.Linear(self.head_dim, dim * expand_k, bias=True)    # (self.num_heads, self.head_dim * self.expand_k, self.head_dim)
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.b_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_conv1d = FadingCausalConv(
            hidden_size=self.dim,
            kernel_size=4,
            conv_bias=False,
            activation=None
            )
        self.k_conv1d = FadingCausalConv(
            hidden_size=self.dim,
            kernel_size=4,
            conv_bias=False,
            activation=None,
        )
        self.v_conv1d = FadingCausalConv(
            hidden_size=self.dim,
            kernel_size=4,
            conv_bias=False,
            activation=None
        )
        self.o_norm = FusedRMSNormGated(
            hidden_size=self.dim // self.num_heads,
            eps=1e-5,
        )
    def forward(self, hidden_states):
        q = self.q_proj(hidden_states)
        q, _ = self.q_conv1d(q, cache=None)
        k = self.k_proj(hidden_states)
        k, _ = self.k_conv1d(k, cache=None)
        v = self.v_proj(hidden_states)
        v, _ = self.v_conv1d(v, cache=None)
        q, k, v = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), (q, k, v))
        q = elu_p1(q)
        k = elu_p1(k)

        weight = self.sep_proj.weight.view(self.num_heads, self.head_dim * self.expand_k, self.head_dim)
        bias = self.sep_proj.bias.view(self.num_heads, self.head_dim * self.expand_k)

        # w = weight.detach()
        # b = bias.detach()

        q = sum_norm(q).to(q)
        k = sum_norm(k).to(k)
        sep_q = _sep_fwd_func(q, weight, bias)
        sep_k = _sep_fwd_func(k, weight, bias)

        aux_loss = None
        if self.training and self.enable_cont_loss:
            _, T, _, _ = k.shape
            cont_k = k.detach()
            inp_index = torch.randperm(T)[:T // 2]
            cont_index = torch.randperm(T)[:T // 2]
            inp_k = sep_k[:, inp_index, ...]
            cont_k = cont_k[:, cont_index, ...]
            cont_k = _sep_fwd_func(cont_k, weight, bias)
            cont_k = sum_norm(cont_k)
            inp_k = sum_norm(inp_k)
            aux_loss = (cont_k * inp_k.detach()).sum(dim=-1) ** 2
            aux_loss = aux_loss.mean()

        # beta = self.b_proj(hidden_states).unsqueeze(-1)
        # k = beta * k
        # o, _ = fused_recurrent_linear_attn(q, k, v)
        o = lin_attn_fwd(sep_q.transpose(1, 2), sep_k.transpose(1, 2), v.transpose(1, 2))
        # g = self.g_proj(hidden_states)
        # g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        # o = self.o_norm(o, g)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.out_proj(o)

        if aux_loss is not None:
            return (o, aux_loss)

        return o
       
if __name__ == '__main__':
    model = SepLA(dim=128, expand_k=4, num_heads=2).cuda()
    x = torch.randn(2, 10, 128).cuda()
    y = model(x)
    y.mean().backward()
        
    
