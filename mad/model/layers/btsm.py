from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from torch.func import vjp


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / (x.norm(dim=-1, keepdim=True) + 1e-6)).to(x)

def _sep_fwd_func(x, weight, bias):
    u = torch.einsum('bthd,HDd->bthD', x, weight) + bias
    # u = F.linear(x, weight, bias)
    s = F.sigmoid(u)
    return s

def _sep_cont_loss_func(s_unnormed, s_p_normed):
    # s_unnormed = F.sigmoid(F.linear(x, weight, bias))
    s_normed = s_unnormed / (s_unnormed.norm(dim=-1, keepdim=True) + 1e-6)
    loss = (s_normed * s_p_normed).sum(dim=-1).mean()
    return loss    
    
class Separator(nn.Module):
    def __init__(self, num_heads: int, in_dim: int, out_dim: int, grad_scale: float = 0.25):
        super().__init__()
        self.num_heads = num_heads    # type: ignore
        self.in_dim = in_dim    # type: ignore
        self.out_dim = out_dim    # type: ignore
        self.grad_scale = grad_scale    # type: ignore
        self.proj = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.cont_weight_grad = None    # type: ignore
        self.cont_bias_grad = None    # type: ignore
        self.proj.weight.register_hook(self._weight_hook)
        self.proj.bias.register_hook(self._bias_hook)

    def _weight_hook(self, grad):
        if self.cont_weight_grad is not None:
            new_grad = self.grad_scale * self.cont_weight_grad
            self.cont_weight_grad = None
            return new_grad
        else:
            return grad

    def _bias_hook(self, grad):
        if self.cont_bias_grad is not None:
            new_grad = self.grad_scale * self.cont_bias_grad
            self.cont_bias_grad = None
            return new_grad
        else:
            return grad

    def forward(self, x, use_cont_grad: bool = True):
        B, T, H, d = x.shape
        weight = self.proj.weight.view(self.num_heads, -1, self.in_dim)
        bias = self.proj.bias.view(self.num_heads, -1)
        # s, vjp_fn = vjp(_sep_fwd_func, x, weight, bias)
        s = _sep_fwd_func(x, weight, bias)
        if use_cont_grad:
            with torch.no_grad():
                _, vjp_fn = vjp(_sep_fwd_func, x, weight, bias)
                x_p = x[:, torch.randperm(T), ...]
                s_p = _sep_fwd_func(x_p, weight, bias)
                # s_p, vjp_fn = vjp(_sep_fwd_func, x, weight, bias)
                s_p_normed = s_p / (s_p.norm(dim=-1, keepdim=True) + 1e-6)
                _, cont_vjp_fn = vjp(_sep_cont_loss_func, s, s_p_normed)
                s_grad, _ = cont_vjp_fn(torch.tensor(1.0))
                _, cont_weight_grad, cont_bias_grad = vjp_fn(s_grad)
                assert self.cont_bias_grad is None and self.cont_weight_grad is None
                self.cont_bias_grad = cont_bias_grad.view(-1)
                self.cont_weight_grad = cont_weight_grad.reshape(-1, self.in_dim)
        return s
        
        

# Operation: y = sigma(Linear(x))
class SeparatorOp(torch.autograd.Function):

    @ staticmethod
    def forward(ctx, x, weight, bias):
        B, T, H, d = x.shape
        with torch.amp.autocast(enabled=False, device_type='cuda'):
            x = x.to(torch.float32)
            s, vjp_fn = vjp(_sep_fwd_func, x, weight, bias)
        x_p = x[:, torch.randperm(T), ...]
        s_p = _sep_fwd_func(x_p, weight, bias)
        ctx.vjp_fn = vjp_fn
        torch.cuda.empty_cache()
        ctx.save_for_backward(s, s_p)
        return s

    @ staticmethod
    def backward(ctx, grad_output):    # type: ignore
        s, s_p = ctx.saved_tensors
        input_grad, weight_grad, bias_grad = ctx.vjp_fn(grad_output)
        s_p_normed = s_p / s_p.norm(dim=-1, keepdim=True)
        _, cont_vjp_fn = vjp(_sep_cont_loss_func, s, s_p_normed)
        s_grad, _ = cont_vjp_fn(torch.tensor(1.0, dtype=s.dtype))
        _, cont_weight_grad, cont_bias_grad = ctx.vjp_fn(s_grad)
        
        # print(weight_grad, cont_weight_grad, bias_grad, cont_bias_grad)
        
        return input_grad, weight_grad + cont_weight_grad, bias_grad + cont_bias_grad
        
        

    # @staticmethod
    # def forward(ctx, x, weight, bias):
    #     u = F.linear(x, weight, bias)
    #     s = F.sigmoid(u)
    #     ctx.save_for_backward(weight, bias, x, u, s)
    #     return s

    # @staticmethod
    # def backward(ctx, grad_output):    # type: ignore
    #     # x: [B, T, H, d]
    #     # grad_output: [B, T, H, D]
    #     # u = Wx
    #     # s = sigma(u)
    #     # y = norm(s)
    #     # g = norm(sigma(Wx')) (detached)
    #     # l = y^T g
    #     weight, bias, x, u, s = ctx.saved_tensors
    #     B, T, H, d = x.shape
    #     x_p = x[:, torch.randperm(T), ...]
    #     s_p = F.sigmoid(F.linear(x_p, weight, bias))
    #     g = s_p / s_p.norm(dim=-1, keepdim=True)
    #     y = s / s.norm(dim=-1, keepdim=True)
    #     norm_grad = (g - torch.einsum('bthd,bthd->bth', g, y).unsqueeze(-1) * y) / s.norm(dim=-1, keepdim=True)
    #     sig_grad = s * (1 - s)
    #     grad = norm_grad * sig_grad
    #     bias_grad = grad.mean(dim=(0, 1, 2))
    #     weight_grad = torch.einsum('bthD,bthd->Dd', grad, x) / (B * T * H)

        
    #     return grad_output + grad

sepop = SeparatorOp.apply

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
        # self.enable_cont_loss = (self.layer_idx == 0)    # type: ignore
        self.enable_cont_loss = True    # type: ignore
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # self.q_sep_proj = nn.Linear(self.head_dim, dim * expand_k, bias=True)    # n_heads x [head_dim, head_dim x expand]
        # self.sep_proj = Separator(self.num_heads, self.head_dim, dim * expand_k)
        self.sep_proj = nn.Linear(self.head_dim, dim * expand_k, bias=True)
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.b_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_conv1d = ShortConvolution(
            hidden_size=self.dim,
            kernel_size=4,
            conv_bias=False,
            activation=None
            )
        self.k_conv1d = ShortConvolution(
            hidden_size=self.dim,
            kernel_size=4,
            conv_bias=False,
            activation=None,
        )
        self.v_conv1d = ShortConvolution(
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
            B, T, H, d = k.shape
            sep_k_p = k.detach()
            sep_k_p = k[:, torch.randperm(T), ...]
            sep_k_p = _sep_fwd_func(sep_k_p, weight, bias)
            # sep_k_p = sum_norm(sep_k_p).to(sep_k_p)
            aux_loss = (sum_norm(sep_k).detach() * sum_norm(sep_k_p)).sum(dim=-1).abs()
            # print(sep_k_p.norm(dim=-1))
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
        
    
