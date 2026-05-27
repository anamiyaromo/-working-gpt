from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data.long()
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_split(data_dir: str | Path = "data/processed") -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = Path(data_dir)
    train = torch.load(data_dir / "train.pt", map_location="cpu")
    val = torch.load(data_dir / "val.pt", map_location="cpu")
    return train, val


def make_loader(data: torch.Tensor, block_size: int, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = NextTokenDataset(data, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

