import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from working_gpt.generate import generate_text, load_model
from working_gpt.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/gpt_latest.pt")
    parser.add_argument("--vocab", default="data/processed/vocab.json")
    parser.add_argument("--prompt", default="The market")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    tokenizer = CharTokenizer.load(args.vocab)
    model = load_model(args.checkpoint)
    print(generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature))


if __name__ == "__main__":
    main()

