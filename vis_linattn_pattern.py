# %%
import torch

from mad.configs import MADConfig, MADModelConfig
from mad.model import PLModelWrap

mad_config = MADConfig()
mad_model_config = MADModelConfig()
args = {
    'layers': ['linear-attention', 'swiglu', 'linear-attention', 'swiglu'],
    'dim': 64,
    'vocab_size': 64,
}
mad_model_config.update_from_kwargs(args)
model = mad_model_config.build_model_from_registry()
checkpoint_path = 'checkpoints/lin-fuzzy3-base-64.ckpt'
model = PLModelWrap.load_from_checkpoint(checkpoint_path, model=model, mad_config=mad_config)

data = torch.tensor([[
    62, 62, 62, 62, 13, 20, 22, 55,  5, 12, 27, 35,  5, 10,  7, 55, 60, 60,
    60, 58, 17, 23,  0, 45, 11, 19, 16, 48, 17, 16, 21, 42, 14, 19, 18, 42,
    21, 22,  9, 36, 13, 15, 22, 51, 23, 22, 15, 47, 60, 60, 60, 57,  7,  9,
    11, 30, 23, 25,  4, 50, 63, 13, 20, 22
]]).cuda()
data = torch.tensor([[
    62, 62, 62, 62, 13, 20, 22, 55,  5, 12, 27, 35,  5, 10,  7, 55, 60, 60,
    60, 58, 17, 23,  0, 45, 11, 19, 16, 48, 17, 16, 21, 42, 14, 19, 18, 42,
    21, 22,  9, 36, 14, 27, 21, 51, 23, 22, 15, 47, 60, 60, 60, 57,  7,  9,
    11, 30, 23, 25,  4, 50, 63, 13, 20, 22
]]).cuda()

# data = torch.tensor([[
#     62, 62, 62, 62, 22,  1,  8, 28,  8, 15, 27, 37, 22, 24,  2, 37, 59, 61,
#     60, 57, 16,  7,  6, 35, 57, 57, 60, 60,  8, 24, 19, 56, 15,  0, 23, 47,
#     13,  8, 19, 31, 25, 10,  5, 52,  9, 11, 23, 39, 15, 13, 23, 35, 11, 22,
#     24, 50,  3,  8, 24, 44, 63, 15,  0, 23
# ]]).cuda()
# data = torch.tensor([[
#     62, 62, 62, 62, 22,  1,  8, 28,  8, 15, 27, 37, 22, 24,  2, 37, 59, 61,
#     60, 57, 16,  7,  6, 35, 57, 57, 60, 60,  8, 24, 19, 56, 15,  0, 23, 47,
#     13,  8, 19, 31, 25, 10,  5, 52,  9, 11, 23, 39, 14, 37, 27, 22, 11, 22,
#     24, 50,  3,  8, 24, 44, 63, 15,  0, 23
# ]]).cuda()

output = model(data)
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
head = 1

handler = RegisterHandler(model.model.model[layer * 2], module_index=1, reusable=True)
handler.register_onetime_record('q', 'parallel_forward')
handler.register_onetime_record('k', 'parallel_forward')
handler.apply()
model(data)
record = handler.get_record()
handler.remove()
q = record['q'][0, head]
k = record['k'][0, head]

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
plt.xlabel('Key Token Index')
plt.ylabel('Query Token Index')
plt.title(f'Attention Map (Layer {layer}, Head {head})')
plt.show()

# %%
import matplotlib.pyplot as plt
from torchinspect import RegisterHandler

layer = 1
head = 1
failed_data = torch.tensor([[
    62, 62, 62, 62, 13, 20, 22, 55,  5, 12, 27, 35,  5, 10,  7, 55, 60, 60,
    60, 58, 17, 23,  0, 45, 11, 19, 16, 48, 17, 16, 21, 42, 14, 19, 18, 42,
    21, 22,  9, 36, 13, 15, 22, 51, 23, 22, 15, 47, 60, 60, 60, 57,  7,  9,
    11, 30, 23, 25,  4, 50, 63, 13, 20, 22
]]).cuda()
success_data = torch.tensor([[
    62, 62, 62, 62, 13, 20, 22, 55,  5, 12, 27, 35,  5, 10,  7, 55, 60, 60,
    60, 58, 17, 23,  0, 45, 11, 19, 16, 48, 17, 16, 21, 42, 14, 19, 18, 42,
    21, 22,  9, 36, 14, 27, 21, 51, 23, 22, 15, 47, 60, 60, 60, 57,  7,  9,
    11, 30, 23, 25,  4, 50, 63, 13, 20, 22
]]).cuda()
handler = RegisterHandler(model.model.model[layer * 2], module_index=1, reusable=True)
handler.register_onetime_record('q', 'parallel_forward')
handler.register_onetime_record('k', 'parallel_forward')
handler.apply()
model(success_data)
record = handler.get_record()
success_q = record['q'][0, head]
success_k = record['k'][0, head]
model(failed_data)
record = handler.get_record()
failed_q = record['q'][0, head]
failed_k = record['k'][0, head]
handler.remove()
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