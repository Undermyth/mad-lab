import torch
import torch.nn as nn
import math


class FlexAttention(nn.Module):

    def __init__(
        self,
        dim: int = 128,
        n_heads: int = 4,
        rotary_emb_dim: int = 8,
        **kwargs
    ):
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
        
        # RoPE 编码器
        self.rope = RotaryEmbedding(rotary_emb_dim)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 线性投影
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, dim]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, dim]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, dim]
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        
        # 应用 RoPE 位置编码（仅应用于前 rotary_emb_dim 维度）
        q, k = self.rope(q, k)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, n_heads, seq_len, seq_len]
        
        # 应用因果掩码（如果需要）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device)).bool()
        # attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        attn_scores = attn_scores.masked_fill(~causal_mask, float(0))
        
        # 计算注意力权重
        # attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, n_heads, seq_len, seq_len]
        attn_weights = torch.sigmoid(attn_scores) * attn_scores
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)  # [batch_size, seq_len, dim]
        
        # 最终线性投影
        output = self.o_proj(attn_output)  # [batch_size, seq_len, dim]
        
        return output


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
            x_pass = x[..., rotary_emb_dim:]    # [batch_size, n_heads, seq_len, head_dim - rotary_emb_dim]
            
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


if __name__ == "__main__":
    # 基本测试
    batch_size = 2
    seq_len = 10
    dim = 128
    n_heads = 4
    
    # 创建模块实例 (rotary_emb_dim < head_dim)
    attention = FlexAttention(dim=dim, n_heads=n_heads, rotary_emb_dim=8)
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = attention(x)
    
    # 验证输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证输出与输入形状一致
    assert output.shape == x.shape, f"Output shape {output.shape} does not match input shape {x.shape}"
    
    # 测试梯度流
    loss = output.sum()
    loss.backward()
    
    print("Basic test passed!")
    print(f"Q proj gradient norm: {attention.q_proj.weight.grad.norm().item():.4f}")
    print(f"K proj gradient norm: {attention.k_proj.weight.grad.norm().item():.4f}")
    print(f"V proj gradient norm: {attention.v_proj.weight.grad.norm().item():.4f}")
    
    # 测试 rotary_emb_dim == head_dim 的情况
    print("\nTesting rotary_emb_dim == head_dim:")
    attention2 = FlexAttention(dim=dim, n_heads=n_heads, rotary_emb_dim=32)  # 32 = head_dim (128/4)
    output2 = attention2(x)
    assert output2.shape == x.shape, f"Output shape {output2.shape} does not match input shape {x.shape}"
    print("Test with rotary_emb_dim == head_dim passed!")