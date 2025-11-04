# %%
import torch

# %%
ckpt = torch.load('hybrid-2k-2.ckpt', weights_only=False)
state_dict = ckpt['state_dict']

# %%
state_dict['model.model.0.1.threshold'] = torch.ones(1, 1, 2, 1).cuda() * 0.5
state_dict['model.model.2.1.threshold'] = torch.ones(1, 1, 2, 1).cuda() * 0.5
# state_dict['model.model.0.1.s_proj.weight'] = torch.nn.Linear(64, 2, bias=False).weight
# state_dict['model.model.2.1.s_proj.weight'] = torch.nn.Linear(64, 2, bias=False).weight

# %%
ckpt['state_dict'] = state_dict
torch.save(ckpt, 'hybrid-2k-th.ckpt')
# %%
