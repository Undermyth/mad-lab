import torch
import torch.nn as nn
from fla.layers import DeltaNet

class DeltaAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        expand_k: int = 1,
        expand_v: int = 1,
        num_heads: int = 16,
        **kwargs
    ):
        super().__init__()
        self.deltanet = DeltaNet(
            d_model=dim,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_gate=False
        )
    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return self.deltanet(hidden_states, *args, **kwargs)[0]
