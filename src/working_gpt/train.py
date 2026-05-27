from pathlib import Path

import torch
from tqdm import tqdm

from working_gpt.config import GPTConfig, TrainConfig
from working_gpt.data import make_loader
from working_gpt.models import TinyGPT
from working_gpt.utils import get_device, set_seed


@torch.no_grad()
def estimate_loss(model: TinyGPT, train_loader, val_loader, device: torch.device, eval_iters: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split, loader in {"train": train_loader, "val": val_loader}.items():
        losses = []
        for i, (xb, yb) in enumerate(loader):
            if i >= eval_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / max(1, len(losses))
    model.train()
    return out


def train_model(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    gpt_config: GPTConfig,
    train_config: TrainConfig,
    checkpoint_dir: str | Path = "checkpoints",
) -> TinyGPT:
    set_seed(train_config.seed)
    device = get_device()
    model = TinyGPT(gpt_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_loader = make_loader(train_data, gpt_config.block_size, train_config.batch_size)
    val_loader = make_loader(val_data, gpt_config.block_size, train_config.batch_size, shuffle=False)
    train_iter = iter(train_loader)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in tqdm(range(train_config.max_steps), desc="training"):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:
            losses = estimate_loss(model, train_loader, val_loader, device, train_config.eval_iters)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save(
                {"model": model.state_dict(), "gpt_config": gpt_config, "train_config": train_config},
                checkpoint_dir / "gpt_latest.pt",
            )

    return model

