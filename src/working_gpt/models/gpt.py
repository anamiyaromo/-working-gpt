import torch
import torch.nn as nn
import torch.nn.functional as F

from working_gpt.config import GPTConfig
from working_gpt.models.blocks import Block


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position_embedding_table = nn.Embedding(config.block_size, config.emb_dim)
        self.blocks = nn.Sequential(
            *[
                Block(config.emb_dim, config.num_heads, config.block_size, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.config.block_size}")

        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(t, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b * t, c), targets.view(b * t))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

