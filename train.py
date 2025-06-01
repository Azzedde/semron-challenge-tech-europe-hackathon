import argparse
import os
import time
import math
from pathlib import Path
import torchvision.transforms as T

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import (
    get_loaders,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
import bitbybit as bb
from bitbybit.config.resnet20 import (
    submission_config_cifar10,
    submission_config_cifar100,
)


def build_param_groups(hashed_model: nn.Module, kernel_type: str):
    """
    Return two lists for AdamW:
      - weight_params: includes all layer.weight, layer.bias, BN.weight, BN.bias
      - proj_params: includes all projection_matrix parameters (if learned_projection)
    """
    weight_params = []
    proj_params = []

    for module in hashed_model.modules():
        # 1) Collect conv/linear weights and biases, BN weights & biases
        if hasattr(module, "weight") and module.weight is not None and module.weight.requires_grad:
            weight_params.append(module.weight)
        if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
            weight_params.append(module.bias)

        if isinstance(module, nn.BatchNorm2d):
            if module.weight is not None and module.weight.requires_grad:
                weight_params.append(module.weight)
            if module.bias is not None and module.bias.requires_grad:
                weight_params.append(module.bias)

        # 2) If learned_projection, collect projection_matrix
        if kernel_type == "learned_projection":
            if hasattr(module, "projection_matrix") and isinstance(module.projection_matrix, nn.Parameter):
                proj_params.append(module.projection_matrix)

    return weight_params, proj_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a hashed ResNet-20 on CIFAR-10 or CIFAR-100 "
                    "with strong augmentations, label smoothing, AdamW + cosine schedule."
    )
    parser.add_argument(
        "--dataset", choices=["CIFAR10", "CIFAR100"],
        default="CIFAR10", help="Which dataset to train on"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=256,
        help="Test batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=120,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--weight-lr", type=float, default=1e-3,
        help="Learning rate for backbone weights (AdamW)"
    )
    parser.add_argument(
        "--proj-lr", type=float, default=1e-4,
        help="Learning rate for projection matrices (AdamW)"
    )
    parser.add_argument(
        "--kernel", choices=["random_projection", "learned_projection"],
        default="learned_projection",
        help="Which projection kernel to use"
    )
    parser.add_argument(
        "--save-dir", type=Path, default=Path("submission_checkpoints"),
        help="Directory to save best checkpoints"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--no-mps", action="store_true", help="Disable MPS (Apple) even if available"
    )
    return parser.parse_args()


def get_transforms(dataset: str):
    """
    Return (train_transform, test_transform) for CIFAR-10 / CIFAR-100, with:
      - RandomCrop + RandomHorizontalFlip
      - AutoAugment(CIFAR10) or AutoAugment(CIFAR100)
      - RandomErasing
      - Normalize
    """
    if dataset == "CIFAR10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        aa_policy = T.AutoAugmentPolicy.CIFAR10
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        aa_policy = T.AutoAugmentPolicy.CIFAR10  # CIFAR policy works well for CIFAR100 too

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(policy=aa_policy),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_transform, test_transform


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_mps = torch.backends.mps.is_available() and not args.no_mps

    if use_cuda:
        device = torch.device("cuda")

    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create save directory
    OUTPUT_DIR = args.save_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Choose dataset, transforms, loaders, and patch_config
    if args.dataset == "CIFAR10":
        patch_config = submission_config_cifar10.copy()
        train_transform, test_transform = get_transforms("CIFAR10")
        train_dataset = CIFAR10(root=Path(__file__).parent / "data",
                                train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=Path(__file__).parent / "data",
                               train=False, download=True, transform=test_transform)
        num_classes = 10
    else:
        patch_config = submission_config_cifar100.copy()
        train_transform, test_transform = get_transforms("CIFAR100")
        train_dataset = CIFAR100(root=Path(__file__).parent / "data",
                                 train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=Path(__file__).parent / "data",
                                train=False, download=True, transform=test_transform)
        num_classes = 100

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"))

    # 2) Overwrite each layer’s hash_kernel_type if needed
    if args.kernel == "learned_projection":
        for layer_name, layer_cfg in patch_config.items():
            if isinstance(layer_cfg, dict) and "hash_kernel_type" in layer_cfg:
                layer_cfg["hash_kernel_type"] = "learned_projection"

    # 3) Build and patch the ResNet-20 backbone
    model = get_backbone(f"{args.dataset.lower()}_resnet20").to(device)
    hashed_model = bb.patch_model(model, config=patch_config).to(device)

    # 4) Adjust BatchNorm momentum (slower running‐stats updates)
    for m in hashed_model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.05

    # 5) Build parameter groups
    weight_params, proj_params = build_param_groups(hashed_model, args.kernel)

    # 6) Define AdamW optimizers
    weight_optim = AdamW(
        weight_params,
        lr=args.weight_lr,
        betas=(0.9, 0.999),
        weight_decay=5e-4,
    )
    if args.kernel == "learned_projection":
        # use a smaller weight_decay on projection matrices
        proj_optim = AdamW(
            proj_params,
            lr=args.proj_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
    else:
        proj_optim = None

    # 7) LR schedulers: linear warmup (10% steps) + cosine‐anneal to 0
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    weight_scheduler = LambdaLR(weight_optim, lr_lambda)
    proj_scheduler = LambdaLR(proj_optim, lr_lambda) if proj_optim else None

    # 8) Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    global_step = 0
    start_time = time.time()

    print(f"\n=== Starting training for {args.epochs} epochs ===\n")
    for epoch in range(1, args.epochs + 1):
        hashed_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            weight_optim.zero_grad()
            if proj_optim:
                proj_optim.zero_grad()

            # forward
            outputs = hashed_model(inputs)
            loss = criterion(outputs, targets)
            # backward
            loss.backward()

            # gradient clipping on projections
            if args.kernel == "learned_projection" and proj_params:
                torch.nn.utils.clip_grad_norm_(proj_params, max_norm=1.0)

            # optimizer steps
            if args.kernel == "learned_projection" and proj_optim:
                proj_optim.step()
            weight_optim.step()

            # scheduler steps
            global_step += 1
            weight_scheduler.step()
            if proj_scheduler:
                proj_scheduler.step()

            # track stats
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                avg_loss = running_loss / total
                acc = 100.0 * correct / total
                lr_w = weight_scheduler.get_last_lr()[0]
                lr_p = proj_scheduler.get_last_lr()[0] if proj_scheduler else 0.0
                elapsed = time.time() - epoch_start
                print(
                    f"Batch {batch_idx:03d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Acc: {acc:.2f}% | "
                    f"LR_w: {lr_w:.2e} | LR_p: {lr_p:.2e} | "
                    f"Elapsed: {elapsed:.2f}s"
                )

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        print(
            f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Time: {epoch_time:.2f}s"
        )

        # 9) Validation pass
        hashed_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = hashed_model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total
        print(f"[Epoch {epoch}] Test Acc: {test_acc:.2f}%")

        # 10) Save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_name = f"{args.dataset.lower()}_resnet20_best.pth"
            save_path = OUTPUT_DIR / ckpt_name
            torch.save(hashed_model.state_dict(), save_path)
            print(f"*** New best model saved (Test Acc: {test_acc:.2f}%) → {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best Test Acc across all epochs: {best_acc:.2f}%\n")


if __name__ == "__main__":
    main()