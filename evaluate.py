import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path  # <-- import Path here

from src.bitbybit.kernel import _HashKernel
from src.bitbybit.utils.score import calculate_submission_score, _hash_kernels
from src.bitbybit.utils.models import get_backbone
from src.bitbybit.utils.data import (
    get_loaders,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
import src.bitbybit as bb
from src.bitbybit.config.resnet20 import submission_config_cifar10, submission_config_cifar100


def evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    max_batches: int = None,
) -> float:
    """
    Evaluate model accuracy on the test dataset.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for the test set
        device: Device to run evaluation on
        max_batches: Max number of batches to evaluate (None = full dataset)

    Returns:
        Accuracy as a percentage (0-100)
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
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def analyze_model(model_path: str, device: torch.device, max_batches: int = None):
    """
    1) Determine CIFAR10 vs CIFAR100 by filename.
    2) Load baseline pretrained ResNet-20, evaluate it.
    3) Patch and load hashed model, evaluate it.
    4) Compute acc_drop, HDOPs/FLOPs, and final score.
    """

    # 1) Decide dataset & config
    lower = model_path.lower()
    if "cifar100" in lower:
        dataset = "CIFAR100"
        backbone_name = "cifar100_resnet20"
        base_config = submission_config_cifar100.copy()
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        dataset = "CIFAR10"
        backbone_name = "cifar10_resnet20"
        base_config = submission_config_cifar10.copy()
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    print(f"Dataset detected: {dataset}")
    print(f"Loading baseline backbone: {backbone_name}")

    # 2) Build test DataLoader (use pathlib.Path, not torch.pathlib)
    _, test_loader = get_loaders(
        dataset_name=dataset,
        data_dir=Path(__file__).parent / "data",  # <--- fixed here
        batch_size=256,      # you can adjust batch size for faster eval
        mean=mean,
        std=std,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # 3) Baseline model (pretrained)
    baseline_model = get_backbone(backbone_name).to(device)
    baseline_acc = evaluate_accuracy(baseline_model, test_loader, device, max_batches)
    print(f"Baseline {backbone_name} accuracy: {baseline_acc:.2f}%\n")

    # 4) Build hashed model exactly as when you trained it
    print("Building hashed model and loading checkpoint …")
    plain_model = get_backbone(backbone_name).to(device)

    # Force every layer's "hash_kernel_type" → "learned_projection"
    for layer_name, layer_cfg in base_config.items():
        if isinstance(layer_cfg, dict) and "hash_kernel_type" in layer_cfg:
            layer_cfg["hash_kernel_type"] = "learned_projection"

    hashed_model = bb.patch_model(plain_model, config=base_config).to(device)

    # 5) Load hashed checkpoint
    state_dict = torch.load(model_path, map_location=device)
    hashed_model.load_state_dict(state_dict)

    # 6) Evaluate hashed model accuracy
    hashed_acc = evaluate_accuracy(hashed_model, test_loader, device, max_batches)
    print(f"Hashed model accuracy: {hashed_acc:.2f}%\n")

    # 7) Compute accuracy drop (as a fraction)
    acc_drop = (baseline_acc - hashed_acc) / 100.0
    print(
        f"Accuracy drop (fraction): {acc_drop:.4f}  "
        f"(baseline {baseline_acc:.2f}% → hashed {hashed_acc:.2f}%)\n"
    )

    # 8) Compute HDOPs vs. FLOPs
    total_hdops = total_flops = 0.0
    kernels = _hash_kernels(hashed_model)

    print(f"Found {len(kernels)} _HashKernel layers. HDOPs / FLOPs per layer:")
    for i, kernel in enumerate(kernels, 1):
        hdops = 2.0 * kernel.hash_length * kernel.in_features
        flops = 2.0 * kernel.in_features * kernel.out_features
        total_hdops += hdops
        total_flops += flops
        ratio = flops / hdops if hdops > 0 else float("nan")
        print(
            f"  Layer {i:2d}: in={kernel.in_features:4d}, out={kernel.out_features:4d}, "
            f"K={kernel.hash_length:5d}  →  HDOPs={hdops:8.0f}, FLOPs={flops:7.0f}, "
            f"ratio={ratio:.3f}"
        )

    print(f"\nTotal HDOPs: {total_hdops:,.0f}")
    print(f"Total FLOPs: {total_flops:,.0f}")
    overall_ratio = total_flops / total_hdops if total_hdops > 0 else float("nan")
    print(f"Overall (FLOPs/HDOPs): {overall_ratio:.3f}\n")

    # 9) Compute final score
    score = calculate_submission_score(hashed_model, acc_drop=acc_drop)
    print(f"Final submission score: {score:.4f}  (δ=0.15, γ=0.5)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and score a hashed ResNet-20 submission."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to your hashed-model checkpoint (e.g. cifar10_resnet20.pth)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        help="Disable MPS (Apple) even if available",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="If set, only evaluate this many test batches (for quick sanity check)",
    )
    args = parser.parse_args()

    # Choose device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_mps = torch.backends.mps.is_available() and not args.no_mps
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    analyze_model(args.checkpoint, device, max_batches=args.max_batches)