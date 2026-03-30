# %%
import torch

from mad.configs import MADConfig, MADModelConfig
from mad.model import PLModelWrap

mad_config = MADConfig()
mad_model_config = MADModelConfig()
args = {
    'layers': ['sep', 'swiglu', 'sep', 'swiglu'],
    'dim': 64,
    'vocab_size': 64,
}
mad_model_config.update_from_kwargs(args)
model = mad_model_config.build_model_from_registry()
checkpoint_path = 'checkpoints/sep-fuzzy3-relu-base-64.ckpt'
model = PLModelWrap.load_from_checkpoint(checkpoint_path, model=model, mad_config=mad_config)

data = torch.tensor([[
    62, 62, 62, 62, 13, 20, 22, 55,  5, 12, 27, 35,  5, 10,  7, 55, 60, 60,
    60, 58, 17, 23,  0, 45, 11, 19, 16, 48, 17, 16, 21, 42, 14, 19, 18, 42,
    21, 22,  9, 36, 13, 15, 22, 51, 23, 22, 15, 47, 60, 60, 60, 57,  7,  9,
    11, 30, 23, 25,  4, 50, 63, 13, 20, 22
]]).cuda()
# data = torch.tensor([[
#     62, 62, 62, 62, 13, 20, 22, 55,  5, 12, 27, 35,  5, 10,  7, 55, 60, 60,
#     60, 58, 17, 23,  0, 45, 11, 19, 16, 48, 17, 16, 21, 42, 14, 19, 18, 42,
#     21, 22,  9, 36, 14, 27, 21, 51, 23, 22, 15, 47, 60, 60, 60, 57,  7,  9,
#     11, 30, 23, 25,  4, 50, 63, 13, 20, 22
# ]]).cuda()
output = model(data)
if isinstance(output, tuple):
    output = output[0]
output = output.view(-1, output.size(-1))
print(output.argmax(axis=-1))

# %%
import matplotlib.pyplot as plt

embedding = model.model.token_embeds.weight

# Convert embedding weights to numpy
embedding_np = embedding.detach().cpu().numpy()

# Create a heatmap visualization
plt.figure(figsize=(12, 8))
plt.imshow(embedding_np, cmap='RdBu', aspect='auto')
plt.colorbar(label='Embedding Value')
plt.xlabel('Embedding Dimension')
plt.ylabel('Token Index')
plt.title('Token Embedding Weights')
plt.tight_layout()
plt.show()

# %%
from torchinspect import RegisterHandler
layer = 0
head = 0

handler = RegisterHandler(model.model.model[layer * 2], module_index=1, reusable=True)
handler.register_onetime_record('sep_q', 'lin_attn_fwd')
handler.register_onetime_record('sep_k', 'lin_attn_fwd')
# handler.register_onetime_record('q', '_sep_fwd_func(q, weight, bias)')
# handler.register_onetime_record('k', '_sep_fwd_func(q, weight, bias)')
handler.apply()
model(data)
record = handler.get_record()
handler.remove()
q = record['sep_q'].transpose(1, 2)[0, head]
k = record['sep_k'].transpose(1, 2)[0, head]

print(q.max(), k.max())
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

axes[0].imshow(q.detach().cpu().numpy(), cmap='binary', aspect='auto')
axes[0].set_title('Query Attention Map')
axes[0].set_xlabel('Embedding Dimension')
axes[0].set_ylabel('Token Index')
# fig.colorbar(axes[0].collections[0], ax=axes[0], label='Embedding Value')

axes[1].imshow(k.detach().cpu().numpy(), cmap='binary', aspect='auto')
axes[1].set_title('Key Attention Map')
axes[1].set_xlabel('Embedding Dimension')
axes[1].set_ylabel('Token Index')
# fig.colorbar(axes[1].collections[0], ax=axes[1], label='Embedding Value')

plt.tight_layout()
plt.show()

# %%
attn_map = torch.matmul(q, k.t())
plt.imshow(attn_map.detach().cpu().numpy(), cmap='gray', aspect='auto')
plt.colorbar(label='Value')
plt.ylabel('Query Token Index')
plt.xlabel('Key Token Index')
plt.title(f'Attention Map (Layer {layer}, Head {head})')
plt.show()

# %%
k_corr = torch.matmul(k, k.t())
plt.imshow(k_corr.detach().cpu().numpy(), cmap='gray', aspect='auto')
plt.ylabel('Key Token Index')
plt.xlabel('Key Token Index')
plt.title(f'Key Self Correlation (Layer {layer}, Head {head})')
plt.show()

# %%
shuffle_k = k[torch.randperm(64), :]
normed_k = k / (k.norm(dim=-1, keepdim=True) + 1e-5)
normed_shuffle_k = shuffle_k / (shuffle_k.norm(dim=-1, keepdim=True) + 1e-5)
sim = normed_k * normed_shuffle_k
sim = sim.sum(dim=-1).mean()
print('average similarity: ', sim)
print('k norms: ', k.norm(dim=-1))
plt.hist(k.norm(dim=-1).detach().cpu().numpy(), bins=30)
plt.title(f'Distribution of K Norm in Layer {layer}, Head {head}')
plt.show()

# %%
layer = 1
head = 1
failed_data = torch.tensor([[
    62, 62, 62, 62, 12, 23, 19, 33, 22, 13, 18, 40, 13, 15, 18, 31,  6, 18,
    19, 38,  3, 15, 21, 28, 19, 21, 23, 41, 10,  7, 24, 39, 25, 16, 26, 37,
    20, 27, 13, 54,  9, 27,  4, 54,  3, 23, 21, 52, 15, 20, 13, 34, 15, 26,
    17, 50, 12, 14, 15, 42, 63,  3, 15, 21
]]).cuda()
success_data = torch.tensor([[
    62, 62, 62, 62, 12, 23, 19, 33, 22, 13, 18, 40, 13, 15, 18, 31,  6, 18,
    19, 38,  3, 15, 21, 28, 19, 21, 23, 41, 10,  7, 24, 39, 25, 16, 26, 37,
    20, 27, 13, 54,  9, 27,  4, 54,  4, 23, 12, 52, 15, 20, 13, 34, 15, 26,
    17, 50, 12, 14, 15, 42, 63,  3, 15, 21
]]).cuda()
origin_attn = model.model.model[layer * 2][1]
detector = LinearAttentionCapture(origin_attn)
model.model.model[layer * 2][1] = detector
model(success_data)
success_q = detector.q_capture[0][head].clone()
success_k = detector.k_capture[0][head].clone()
model(failed_data)
failed_q = detector.q_capture[0][head].clone()
failed_k = detector.k_capture[0][head].clone()
model.model.model[layer * 2][1] = origin_attn
diff_in_attn_map = torch.matmul(failed_q, failed_k.t()) - torch.matmul(success_q, success_k.t())
v_abs_max = max(abs(diff_in_attn_map.min()), abs(diff_in_attn_map.max()))
plt.imshow(diff_in_attn_map.detach().cpu().numpy(), cmap='bwr', vmin=-v_abs_max, vmax=v_abs_max)
plt.title(f'Diff in Attention Map (Layer {layer}, Head {head})')
plt.colorbar(label='Value')
plt.xlabel('Key Token Index')
plt.ylabel('Query Token Index')
plt.show()
# %%
import numpy as np

x = np.array(
    [[1, 3, 2], [0, 2, 0]]
)
print(x.shape)
plt.imshow(x, aspect='auto')
plt.colorbar()

from torch.optim import Adam