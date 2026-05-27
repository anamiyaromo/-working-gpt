import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from working_gpt.config import GPTConfig, TrainConfig
from working_gpt.data import load_split
from working_gpt.tokenizer import CharTokenizer
from working_gpt.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer = CharTokenizer.load(data_dir / "vocab.json")
    train_data, val_data = load_split(data_dir)

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        learning_rate=args.learning_rate,
    )
    train_model(train_data, val_data, gpt_config, train_config)


if __name__ == "__main__":
    main()

