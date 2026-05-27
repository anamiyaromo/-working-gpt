# Working GPT

Course project for **Artificial Intelligence and Financial Engineering**.

This repository builds a small GPT-style autoregressive language model from scratch in PyTorch. The code follows the progression of the course notebooks:

1. Bigram language model on names
2. MLP character model on names
3. MLP language model on Tiny Shakespeare
4. GPT-style next-token dataset and sequence loss
5. Masked self-attention
6. Multi-head attention, transformer blocks, and TinyGPT

The final goal is a working GPT-style model that can be trained and sampled from this repository.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --input data/raw/tinyshakespeare.txt
python scripts/train_gpt.py --max-steps 1000
python scripts/sample.py --checkpoint checkpoints/gpt_latest.pt --prompt "The market"
```

If `data/raw/tinyshakespeare.txt` is missing, `prepare_data.py` creates a tiny fallback corpus so the pipeline can still run.

## Project Layout

```text
src/working_gpt/
  tokenizer.py        character-level tokenizer
  data.py             next-token dataset and dataloaders
  models/gpt.py       TinyGPT model
  train.py            training loop
  generate.py         text generation utilities
scripts/
  prepare_data.py     build vocab and train/val split
  train_gpt.py        train the final GPT-style model
  sample.py           generate text from a checkpoint
docs/
  architecture.md     model explanation
  training_notes.md   training checklist and experiments
```

## References

- Andrej Karpathy, `nn-zero-to-hero`: https://github.com/karpathy/nn-zero-to-hero
- Andrej Karpathy, `makemore`: https://github.com/karpathy/makemore
- Vaswani et al., "Attention Is All You Need"

