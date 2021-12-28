import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from typing import List, Optional, Tuple, Union

dataset_key = {"cifar10": ds.CIFAR10, "mnist": ds.MNIST, "caltech": ds.Caltech101}


def training_data(
    batch_size: Optional[int] = 8,
    dataset: Optional[str] = "cifar10",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    data = dataset_key[dataset](
        root="datasets/" + dataset, transform=transform, download=True
    )
    loader = DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return loader


def noise(
    dim: int,
    batch_size: Optional[int] = 8,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    vec = torch.randn((batch_size, dim))

    return vec.to(device)


def save_images(batch: torch.Tensor, epoch: int, mosaic: bool = False):
    pass


if __name__ == "__main__":
    data = training_data(1)
    assert isinstance(data, DataLoader)
    rand_noise = noise(10, batch_size=1)
    assert isinstance(rand_noise, torch.Tensor)
    assert rand_noise.shape == (1, 10)
