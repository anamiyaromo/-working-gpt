import torch

from working_gpt.data import NextTokenDataset


def test_next_token_dataset_shapes():
    dataset = NextTokenDataset(torch.arange(10), block_size=4)
    x, y = dataset[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]

