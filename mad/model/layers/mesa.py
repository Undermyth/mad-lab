import torch
import torch.nn as nn
from fla.layers import MesaNet

class MesaAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        num_heads: int = 16,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads    # type: ignore
        self.dim = dim    # type: ignore
        self.expand_k = expand_k    # type: ignore
        self.head_dim = self.dim * self.expand_k // self.num_heads   # type: ignore
        self.mesanet = MesaNet(
            hidden_size=dim,
            num_heads=num_heads,
            head_dim=self.head_dim,
            use_output_gate=True,
        )
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return self.mesanet(hidden_states, *args, **kwargs)[0]
