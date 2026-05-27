import torch

from working_gpt.config import GPTConfig
from working_gpt.models import TinyGPT


def test_gpt_forward_shape_and_loss():
    config = GPTConfig(vocab_size=12, block_size=8, emb_dim=16, num_heads=4, num_layers=2)
    model = TinyGPT(config)
    x = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is not None

