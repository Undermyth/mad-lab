import numpy as np
from mad.data.instances import generate_distract_fuzzy_in_context_recall_instance

instance = generate_distract_fuzzy_in_context_recall_instance(
    vocab_size=32,
    seq_len=64,
    k_motif_size=4,
    v_motif_size=1,
    is_training=False,
    noise_vocab_size=10,
    frac_noise=0.1,
    # rng=np.random.default_rng(seed=1044)
)
print(instance)

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
