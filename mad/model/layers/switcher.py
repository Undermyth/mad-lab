import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule

import math

class Heaviside(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        # input, = ctx.saved_tensors
        # s = torch.sigmoid(input)
        # return 2 * s * (1 - s)

heaviside = Heaviside.apply

class STE(torch.autograd.Function):
    '''
    straight through estimator
    '''

    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste = STE.apply

class Switcher(nn.Module):
    def __init__(self, dim: int = 128, n_heads: int = 4, rotary_emb_dim: int = 8, **kwargs):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rotary_emb_dim = rotary_emb_dim

        # 确保 rotary_emb_dim 不超过 head_dim，并且是偶数
        assert self.rotary_emb_dim <= self.head_dim, "rotary_emb_dim must be less than or equal to head_dim"
        assert self.rotary_emb_dim % 2 == 0, "rotary_emb_dim must be even"
        assert self.head_dim % 2 == 0, "head_dim must be even"

        # 线性变换层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.b_proj = nn.Linear(dim, n_heads, bias=False)

        self.q_conv1d = ShortConvolution(
            hidden_size=self.dim,
            kernel_size=4,
            activation="silu",
        )

        self.k_conv1d = ShortConvolution(
            hidden_size=self.dim,
            kernel_size=4,
            activation="silu",
        )

        self.v_conv1d = ShortConvolution(
            hidden_size=self.dim, 
            kernel_size=4, 
            activation="silu"
        )

        self.o_norm = nn.RMSNorm(self.head_dim, eps=1e-5)

        self.s_proj = nn.Linear(self.dim, self.n_heads, bias=False)

        self.threshold = nn.Parameter(torch.ones(1, 1, self.n_heads, 1).cuda() * 0.5)

        # RoPE 编码器
        self.rope = RotaryEmbedding(rotary_emb_dim)

        self.mode = 'hybrid'

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, _ = self.q_conv1d(q, output_final_state=False)
        k, _ = self.k_conv1d(k, output_final_state=False)
        v, _ = self.v_conv1d(v, output_final_state=False)

        beta = self.b_proj(hidden_states).sigmoid()

        q, k, v = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_dim), (q, k, v))

        o, _ = chunk_delta_rule(
            q=q, k=k, v=v, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k)

        o_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o_attn = o_attn.transpose(1, 2).contiguous()
        o_attn = self.o_norm(o_attn)

        o = self.o_norm(o)  

        if self.mode == 'random':
            s = torch.ones(o_attn.shape[0], o_attn.shape[1], o_attn.shape[2], 1, device=o.device) * 0.1
            s = torch.bernoulli(s).detach()
        else:
            o_reshaped = rearrange(o, "b t h d -> b t (h d)")
            s = self.s_proj(o_reshaped).sigmoid().unsqueeze(-1)    # [VARI] maybe before norm?
            s = heaviside(s - self.threshold)

        if self.mode == 'quadratic':
            o = o_attn
        elif self.mode == 'hybrid' or self.mode == 'random':
            o = (1 - s) * o + s * o_attn  # [VARI] maybe the inverse?
        else:
            assert self.mode == 'linear'
        # o = o_attn

        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        return o, s


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        Args:
            q: Query tensor of shape [batch_size, n_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, n_heads, seq_len, head_dim]
        Returns:
            q_rot: Rotated query tensor
            k_rot: Rotated key tensor
        """
        seq_len = q.shape[-2]
        device = q.device

        # 生成位置索引
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]

        # 生成旋转矩阵
        cos = emb.cos()  # [seq_len, dim]
        sin = emb.sin()  # [seq_len, dim]

        # 应用旋转
        q_rot = self.apply_rotary_pos_emb(q, cos, sin)
        k_rot = self.apply_rotary_pos_emb(k, cos, sin)

        return q_rot, k_rot

    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Apply rotary position embedding to input tensor
        Args:
            x: Input tensor of shape [batch_size, n_heads, seq_len, head_dim]
            cos: Cosine values of shape [seq_len, rotary_emb_dim]
            sin: Sine values of shape [seq_len, rotary_emb_dim]
        Returns:
            Rotated tensor of same shape as input
        """
        # 获取输入张量的形状
        batch_size, n_heads, seq_len, head_dim = x.shape
        rotary_emb_dim = cos.shape[-1]

        # 确保cos和sin在与输入相同的设备上
        cos = cos.to(x.device)
        sin = sin.to(x.device)

        # 如果 rotary_emb_dim 等于 head_dim，则对整个张量应用 RoPE
        if rotary_emb_dim == head_dim:
            # 将输入分为两部分
            x1, x2 = x.chunk(2, dim=-1)  # Each of shape [batch_size, n_heads, seq_len, head_dim//2]

            # 应用旋转公式
            cos = cos[None, None, :, :].to(x.device)  # [1, 1, seq_len, head_dim]
            sin = sin[None, None, :, :].to(x.device)  # [1, 1, seq_len, head_dim]

            # 旋转
            rotated_x = torch.cat((-x2, x1), dim=-1)  # [batch_size, n_heads, seq_len, head_dim]
            x_rot = (x * cos) + (rotated_x * sin)  # [batch_size, n_heads, seq_len, head_dim]
        else:
            # 如果 rotary_emb_dim 小于 head_dim，则只对前 rotary_emb_dim 维度应用 RoPE
            # 分割张量：旋转部分和非旋转部分
            x_rotary = x[..., :rotary_emb_dim]  # [batch_size, n_heads, seq_len, rotary_emb_dim]
            x_pass = x[..., rotary_emb_dim:]  # [batch_size, n_heads, seq_len, head_dim - rotary_emb_dim]

            # 对旋转部分应用 RoPE
            x1, x2 = x_rotary.chunk(2, dim=-1)  # Each of shape [batch_size, n_heads, seq_len, rotary_emb_dim//2]

            # 应用旋转公式
            cos = cos[None, None, :, :].to(x.device)  # [1, 1, seq_len, rotary_emb_dim]
            sin = sin[None, None, :, :].to(x.device)  # [1, 1, seq_len, rotary_emb_dim]

            # 旋转
            rotated_x = torch.cat((-x2, x1), dim=-1)  # [batch_size, n_heads, seq_len, rotary_emb_dim]
            x_rotary_rot = (x_rotary * cos) + (rotated_x * sin)  # [batch_size, n_heads, seq_len, rotary_emb_dim]

            # 合并旋转部分和非旋转部分
            x_rot = torch.cat((x_rotary_rot, x_pass), dim=-1)  # [batch_size, n_heads, seq_len, head_dim]

        return x_rot
