from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    emb_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_steps: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    seed: int = 1337

