from __future__ import annotations

from pathlib import Path

import torch

from working_gpt.models import TinyGPT
from working_gpt.tokenizer import CharTokenizer
from working_gpt.utils import get_device


def load_model(checkpoint_path: str | Path) -> TinyGPT:
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TinyGPT(checkpoint["gpt_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def generate_text(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 1.0,
) -> str:
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(out[0].tolist())
