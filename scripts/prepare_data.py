import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from working_gpt.tokenizer import CharTokenizer


FALLBACK_TEXT = """The market opened with cautious optimism.
Risk, return, volatility, and time are the language of finance.
This tiny corpus exists only to verify that the GPT pipeline runs.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/tinyshakespeare.txt")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        text = input_path.read_text(encoding="utf-8")
    else:
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(FALLBACK_TEXT, encoding="utf-8")
        text = FALLBACK_TEXT

    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int((1 - args.val_ratio) * len(data))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data[:n], out_dir / "train.pt")
    torch.save(data[n:], out_dir / "val.pt")
    tokenizer.save(out_dir / "vocab.json")
    print(f"tokens: {len(data)}, vocab: {tokenizer.vocab_size}, train: {n}, val: {len(data) - n}")


if __name__ == "__main__":
    main()
