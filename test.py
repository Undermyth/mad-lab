# %%
import torch
from fla.modules import ShortConvolution

conv = ShortConvolution(
    hidden_size=3,
    kernel_size=4,
    bias=False,
    activation=None,
)
print(conv.weight.shape)

conv.weight.data = torch.tensor(
    [[1, 2, 3], [4, 5, 6], [-1, -2, -3], [-4, -5, -6]]
).float().unsqueeze(dim=1)

print(conv.weight.shape)

conv = conv.cuda()

# %%
x = torch.ones(1, 6, 3).cuda()
print(conv(x))

# %%
import torch
from fla.modules import ShortConvolution

conv = ShortConvolution(
    hidden_size=3,
    kernel_size=4,
    bias=False,
    activation=None,
    backend='triton' 
)
print(conv.weight.shape)

# fake_weight = torch.tensor([[i + j - 4 for i in range(3)] for j in range(4)]).float()
# fake_weight = torch.tensor([[1, 2, 3], [4, 5, 6], [-1, -2, -3], [-4, -5, 6]]).float()
# conv.weight.data = fake_weight.transpose(0, 1).unsqueeze(dim=1)

conv.weight.data = torch.tensor(
    [[1, 2, 3], [4, 5, 6], [-1, -2, -3], [-4, -5, -6]]
).float().transpose(0, 1).unsqueeze(dim=1)

print(conv.weight.shape)

conv = conv.cuda()

# %%
x = torch.ones(1, 6, 3).cuda()
print(conv(x))
# %%

