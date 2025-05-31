from pathlib import Path
import torch

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
import bitbybit as bb
from bitbybit.config.resnet20 import resnet20_full_patch_config

OUTPUT_DIR = Path(__file__).parent / "submission_checkpoints"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cifar_10_train_loader, cifar_10_test_loader = get_loaders(
        dataset_name="CIFAR10",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_workers=2,
        pin_memory=True,
    )

    cifar_100_train_loader, cifar_100_test_loader = get_loaders(
        dataset_name="CIFAR100",
        data_dir=Path(__file__).parent / "data",
        batch_size=128,     
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        num_workers=2,
        pin_memory=True,
    )

    models = [
        ("cifar10_resnet20", get_backbone("cifar10_resnet20"), cifar_10_train_loader, cifar_10_test_loader),
        ("cifar100_resnet20", get_backbone("cifar100_resnet20"), cifar_100_train_loader, cifar_100_test_loader),
    ]

    for model_name, model, train_loader, test_loader in models:
        # <train model>

        # <evaluate model>

        # Store model
        hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
        torch.save(hashed_model.state_dict(), OUTPUT_DIR / f"{model_name}.pth")


if __name__ == "__main__":
    main()