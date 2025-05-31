#python evaluate_hash.py --model cifar10_resnet20 --kernel random_projection --max-batches 50

"""
evaluate_hash.py

Measure test‐set accuracy of both the original (float32) model and its
“hashed” (patched) counterpart, then compute a hackathon‐style score.

Usage examples:
  # CIFAR-10, Random‐projection hash, full test set:
  python evaluate_hash.py \
      --model cifar10_resnet20 \
      --kernel random_projection

  # CIFAR-100, Learned hash, only first 50 batches of test set:
  python evaluate_hash.py \
      --model cifar100_resnet20 \
      --kernel learned_projection \
      --max-batches 50

  # To force CPU (even if you have a GPU):
  python evaluate_hash.py \
      --model cifar10_resnet20 \
      --kernel random_projection \
      --no-cuda
"""

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

import bitbybit as bb
from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import (
    get_loaders,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from bitbybit.utils.score import calculate_submission_score

# Import patch‐config aliases for ResNet-20
from bitbybit.config.resnet20 import (
    submission_config_cifar10,
    submission_config_cifar100,
)

from pathlib import Path


# If you have a similar config file for VGG11 (e.g. `bitbybit/config/vgg11.py`),
# import those here. For now, we only demonstrate ResNet‐20.
# from bitbybit.config.vgg11 import submission_config_vgg11_cifar10, submission_config_vgg11_cifar100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate original vs. hashed model accuracy and compute hackathon score"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cifar10_resnet20", "cifar100_resnet20", "cifar10_vgg11_bn", "cifar100_vgg11_bn"],
        help="Which backbone to evaluate"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["random_projection", "learned_projection"],
        default="random_projection",
        help="Which hash kernel to use when patching"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for test DataLoader"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Only evaluate this many batches (None = full test split)"
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    return parser.parse_args()


def evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> float:
    """
    Evaluate model accuracy on the test dataset.

    Args:
        model:       PyTorch model (should already be on `device`)
        test_loader: DataLoader for the test set
        device:      CPU/GPU device
        max_batches: Max number of batches to run (None = entire dataset)

    Returns:
        Accuracy as a percentage (0.0 to 100.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    if total == 0:
        return 0.0
    accuracy = 100.0 * correct / total
    return accuracy


def main():
    args = parse_args()

    # 1) Device selection
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}\n")

    # 2) Determine dataset (CIFAR10 vs. CIFAR100) and get test_loader
    dataset_name = None
    if args.model.startswith("cifar10_"):
        dataset_name = "CIFAR10"
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        patch_config = submission_config_cifar10
    elif args.model.startswith("cifar100_"):
        dataset_name = "CIFAR100"
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        patch_config = submission_config_cifar100
    else:
        raise ValueError(f"Unsupported model: {args.model}. Must start with 'cifar10_' or 'cifar100_'.")

    # We only need the test loader for evaluation
    _, test_loader = get_loaders(
        dataset_name=dataset_name,
        data_dir=Path(__file__).parent / "data",
        batch_size=args.batch_size,
        mean=mean,
        std=std,
        num_workers=2,
        pin_memory=use_cuda,
    )

    # 3) Load the original backbone and move to device
    print(f"Loading backbone: {args.model}")
    original_model = get_backbone(args.model)
    original_model = original_model.to(device)

    # 4) Evaluate original model accuracy
    print("Evaluating original model (float32) accuracy...")
    orig_acc = evaluate_accuracy(original_model, test_loader, device, max_batches=args.max_batches)
    print(f"Original {args.model} accuracy: {orig_acc:.2f}%\n")

    # 5) Patch the model (hashed) with the chosen kernel and config
    print(f"Patching model with {args.kernel} …")
    hashed_model = bb.patch_model(original_model, config=patch_config)
    # If your patch‐factory requires specifying kernel type at patch time, you might need:
    # hashed_model = bb.patch_model(original_model, config=patch_config, kernel_type=args.kernel)
    hashed_model = hashed_model.to(device)

    # 6) Evaluate hashed model accuracy
    print("Evaluating hashed (patched) model accuracy …")
    hashed_acc = evaluate_accuracy(hashed_model, test_loader, device, max_batches=args.max_batches)
    print(f"Hashed {args.model} accuracy: {hashed_acc:.2f}%\n")

    # 7) Compute accuracy drop (as a fraction, not percentage)
    acc_drop = (orig_acc - hashed_acc) / 100.0
    print(f"Accuracy drop fraction: {acc_drop:.4f}")

    # 8) Compute hackathon score
    # The scoring function typically expects (hashed_model, acc_drop_fraction).
    score = calculate_submission_score(hashed_model, acc_drop=acc_drop)
    print(f"Submission score: {score:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
