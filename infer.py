# %%
from mad.data.instances import generate_vocab_permutations, exists
import numpy as np
import typing as tp
from mad.model.language_model import LanguageModel
from mad.configs import MADModelConfig
from mad.model.layers import Switcher

import torch
import torchshow as ts

def generate_multiquery_fuzzy_in_context_recall_instance(
    vocab_size: int = 16,
    seq_len: int = 128,
    k_motif_size: int = 3,
    v_motif_size: int = 3,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    frac_query: float = 0.1,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:

    if not exists(rng):
        rng = np.random.default_rng()

    silence_prefix = vocab_size - 1
    active_prefix = vocab_size - 2
    pad_token = vocab_size - 3
    non_special_vocab_size = vocab_size - 3
    non_special_vocab_size -= noise_vocab_size
    key_vocab = np.arange(non_special_vocab_size//2)
    value_vocab = np.arange(non_special_vocab_size//2, non_special_vocab_size)

    if is_training:
        # generate keys and values of variable motif sizes
        keys = {}
        for motif_size in range(1, k_motif_size+1):
            keys_size = generate_vocab_permutations(key_vocab, token_motif_size=motif_size, rng=rng)
            keys[motif_size] = keys_size
        values = {}
        for motif_size in range(1, v_motif_size+1):
            values_size = generate_vocab_permutations(value_vocab, token_motif_size=motif_size, rng=rng)
            values[motif_size] = values_size
    else:
        # we always prompt at the maximum key motif size
        keys = {k_motif_size: generate_vocab_permutations(key_vocab, token_motif_size=k_motif_size, rng=rng)}
        values = {}
        for motif_size in range(1, v_motif_size+1):
            values_size = generate_vocab_permutations(value_vocab, token_motif_size=motif_size, rng=rng)
            values[motif_size] = values_size

    # generate noise vocab, if needed:
    assert frac_noise >= 0 and frac_noise < 1, "frac_noise must be 0 =< frac_noise < 1"
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise >0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)

    kv_map = {s: {} for s in range(1, k_motif_size+1)}
    inputs, targets = [], []
    keys_presented = []

    inputs.extend([silence_prefix])
    targets.extend([target_ignore_idx])
    silence_mode = True

    special_prob = frac_noise + frac_query  # noise: [0, frac_noise], query: [frac_noise, special_prob]

    while len(inputs) < seq_len - 2 * (k_motif_size + v_motif_size + 1):

        # determine key-value motif sizes:
        k_size = rng.choice(list(keys.keys())) if is_training else k_motif_size
        v_size = rng.choice(list(values.keys()))

        # determine if noise or key-value pair collected:
        sample_prob = rng.random()
        is_noise = sample_prob<frac_noise if frac_noise>0 else False
        is_query = frac_noise<sample_prob<special_prob if frac_query>0 else False

        if is_query and len(keys_presented) > 0:
            if silence_mode:
                silence_mode = False
                inputs.extend([active_prefix])
                targets.extend([target_ignore_idx])
            # randomly select a key-value pair from keys_presented
            # key_list = list(keys_presented.keys())
            # k_query = key_list[rng.integers(len(key_list))]
            # v_query = keys_presented[k_query]
            k_query, v_query = keys_presented[-5 if len(keys_presented) >= 5 else 0]
            
            # extend inputs with the key
            inputs.extend(k_query)
            inputs.extend(v_query)
            
            # extend targets with ignore tokens for key and the value
            targets.extend([target_ignore_idx] * len(k_query))
            targets.extend(v_query)

        else:
            if not silence_mode:
                silence_mode = True
                inputs.extend([silence_prefix])
                targets.extend([target_ignore_idx])

            if is_noise:
                noise_size = k_size + v_size
                noise = rng.choice(noise_vocab, size=noise_size, replace=True)
                inputs.extend(noise)
                targets.extend(tuple([target_ignore_idx]*noise_size))
        
            # collect key-value pair:
            else:
                # key:
                k = tuple(rng.choice(keys[k_size]))
                inputs.extend(k)
                if k not in kv_map[k_size]:
                    v = tuple(rng.choice(values[v_size]))
                    kv_map[k_size][k] = v
                else:
                    v = kv_map[k_size][k]
                inputs.extend(v)
                targets.extend(tuple([target_ignore_idx]*k_size))
                targets.extend(tuple([target_ignore_idx]*len(v)))
                # keys_presented[k] = v
                keys_presented.append((k, v))

    # print(key_vocab, value_vocab, noise_vocab)
    # print(keys_presented)

    if silence_mode:
        inputs.extend([active_prefix])
        targets.extend([target_ignore_idx])

    # key_list = list(keys_presented.keys())
    # key = key_list[rng.integers(len(key_list))]
    # value = keys_presented[key]
    key, value = keys_presented[-5 if len(keys_presented) >= 5 else 0]
    inputs.extend(key)
    inputs.extend(value)
    targets.extend(tuple([target_ignore_idx]*len(key)))
    targets.extend(value)

    inputs = np.array(inputs).astype(int)
    targets = np.array(targets).astype(int)

    # pad inputs/targets to seq_len:
    if len(inputs)<(seq_len+1): # add one to account for autoregressive shift
        n_pad = seq_len+1-len(inputs)
        inputs = np.concatenate([np.array([pad_token]*n_pad), inputs])
        targets = np.concatenate([np.array([target_ignore_idx]*n_pad), targets])

    if is_training:
        # autoregressive shift
        # return inputs[:-1], inputs[1:] # use shifted inputs as targets for training
        return inputs[:-1], targets[1:] # use shifted inputs as targets for training
    else:
        return inputs[:-1], targets[1:]

instance = generate_multiquery_fuzzy_in_context_recall_instance(
    vocab_size=128,
    seq_len=2048,
    k_motif_size=2,
    v_motif_size=1,
    is_training=True,
    noise_vocab_size=10,
    frac_noise=0.1,
    frac_query=0.2
)

sample, target = instance
sample = torch.LongTensor(sample).unsqueeze(0).cuda()
target = torch.LongTensor(target).unsqueeze(0).cuda()

config = MADModelConfig(
    layers=['switcher', 'swiglu', 'switcher', 'swiglu'],
    dim=64,
    vocab_size=128
)

model = config.build_model_from_registry().cuda()

# %%

ckpt = torch.load('select-2k-2.ckpt', weights_only=False)
state_dict = ckpt['state_dict']
new_state_dict = {}
for key, weight in state_dict.items():
    if key.startswith('model.'):
        newkey = key[6:]
    else:
        newkey = key
    new_state_dict[newkey] = weight
model.load_state_dict(new_state_dict)

# %%
for name, module in model.named_modules():
    if isinstance(module, Switcher):
        module.mode = 'quadratic'

# %%
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(sample)

predict, extend = output

# %%
extend = extend.squeeze(-1).mean(dim=-1).squeeze(dim=0)
mask = (target != -100).squeeze(dim=0)
print(extend[mask].mean())
predict = torch.softmax(predict, dim=-1)
predict = torch.argmax(predict, dim=-1)
mask_target = target.squeeze(0)[mask]
predict = predict.squeeze(0)[mask]
print((predict == mask_target).float().mean())

# %%
