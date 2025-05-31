from pathlib import Path
from typing import Literal, Sequence, Tuple, Type

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

DatasetName = Literal["CIFAR10", "CIFAR100"]


# For CIFAR10 from `chenyaofo/pytorch-cifar-models`
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# For CIFAR100 from `chenyaofo/pytorch-cifar-models`
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_cifar_dataset_class(dataset_name: DatasetName) -> Type[Dataset]:
    """Returns the appropriate CIFAR dataset class."""
    if dataset_name == "CIFAR10":
        return CIFAR10
    elif dataset_name == "CIFAR100":
        return CIFAR100
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. Must be 'CIFAR10' or 'CIFAR100'."
        )


def get_loaders(
    dataset_name: DatasetName,
    data_dir: str | Path,
    batch_size: int,
    mean: Sequence[float],
    std: Sequence[float],
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns training and testing DataLoaders for CIFAR10 or CIFAR100.

    Args:
        dataset_name: Name of the dataset ("CIFAR10" or "CIFAR100").
        data_dir: Directory to store/load the dataset.
        batch_size: Number of samples per batch.
        mean: Sequence of mean values for normalization (R, G, B).
        std: Sequence of standard deviation values for normalization (R, G, B).
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: If True, a_dir_loader will copy Tensors into pinned memory
                    before returning them.

    Returns:
        A tuple containing the training DataLoader and testing DataLoader.
    """
    data_dir = Path(data_dir)

    transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    DatasetClass: Type[Dataset] = get_cifar_dataset_class(dataset_name)

    train_ds = DatasetClass(
        root=str(data_dir), train=True, download=True, transform=transform_train
    )
    test_ds = DatasetClass(
        root=str(data_dir), train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Often useful for training, esp. with BatchNorm
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,  # Can be larger for testing if memory allows
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
