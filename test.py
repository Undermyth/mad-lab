from mad.data.instances import generate_in_context_recall_instance 

res = generate_in_context_recall_instance(
    vocab_size=16,
    seq_len=32,
    noise_vocab_size=8,
    frac_noise=0.6,
    is_training=True
)
print(res)