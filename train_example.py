# python train_example.py  --dataset CIFAR10 --batch-size 32  --epochs 1  --kernel learned_projection

# python train_example.py --no-cuda --epochs 2 --kernel random_projection    #for cpu



import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import (
    get_loaders,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
import bitbybit as bb
# from bitbybit.config.resnet20 import resnet20_full_patch_config
from bitbybit.config.resnet20 import (
    submission_config_cifar10,
    submission_config_cifar100,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a hashed ResNet-20 on CIFAR-10 or CIFAR-100 with separate optimizers for weights and projections"
    )
    parser.add_argument(
        "--dataset", choices=["CIFAR10", "CIFAR100"],
        default="CIFAR10",
        help="Which dataset to train on"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=128,
        help="Test batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--weight-lr", type=float, default=0.1,
        help="Learning rate for backbone weights"
    )
    parser.add_argument(
        "--proj-lr", type=float, default=1e-5,
        help="Learning rate for projection matrices (learned kernel)"
    )
    parser.add_argument(
        "--kernel", choices=["random_projection", "learned_projection"],
        default="random_projection",
        help="Which projection kernel to use"
    )
    parser.add_argument(
        "--save-dir", type=Path, default=Path("submission_checkpoints"),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--no-cuda", action="store_true",
        help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--no-mps", action="store_true",
        help="Disable MPS even if available"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
        use_gpu = True
    elif use_mps:
        device = torch.device("mps")
        use_gpu = True
    else:
        device = torch.device("cpu")
        use_gpu = False

    print(f"Using device: {device}")

    # Create save directory
    OUTPUT_DIR = args.save_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # print(f"Saving checkpoints to: {OUTPUT_DIR}")
    


    # Choose dataset-specific loaders and normalization
    if args.dataset == "CIFAR10":
        patch_config = submission_config_cifar10
        train_loader, test_loader = get_loaders(
            dataset_name="CIFAR10",
            data_dir=Path(__file__).parent / "data",
            batch_size=args.batch_size,
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
            num_workers=2,
            pin_memory=use_gpu,
        )
        num_classes = 10
    else:
        patch_config = submission_config_cifar100
        train_loader, test_loader = get_loaders(
            dataset_name="CIFAR100",
            data_dir=Path(__file__).parent / "data",
            batch_size=args.batch_size,
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
            num_workers=2,
            pin_memory=use_gpu,
        )
        num_classes = 100
   
    #overwrite the resnet parameter
    if args.kernel == "learned_projection":
    
        for layer_name, layer_cfg in patch_config.items():
            # layer_cfg should be a dict with a "hash_kernel_type" key
            if isinstance(layer_cfg, dict) and "hash_kernel_type" in layer_cfg:
                layer_cfg["hash_kernel_type"] = "learned_projection"


    # Build backbone and patch with hashed layers
    model = get_backbone(f"{args.dataset.lower()}_resnet20")
    model = model.to(device)
    # hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
    hashed_model = bb.patch_model(model, config=patch_config)
    hashed_model = hashed_model.to(device)

    # Separate parameters: backbone weights (if any require_grad) and projection matrices
    weight_params = []
    proj_params = []

    for module in hashed_model.modules():
        # Collect any weight parameters that require gradients
        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.requires_grad:
                weight_params.append(module.weight)
        # Handle projection parameters differently for random vs. learned
        if args.kernel == "learned_projection":
        
            # LearnedProjKernel has attribute `projection_matrix` as nn.Parameter
            if hasattr(module, "projection_matrix") and isinstance(module.projection_matrix, nn.Parameter):
                # print(f"Found learned projection matrix in {module.__class__.__name__}")
                proj_params.append(module.projection_matrix)
        else:
            # For random_projection, the projection is a buffer; we do not train it
            # But if you mistakenly want to optimize random projections, you could:
            # if hasattr(module, "_random_projection_matrix"):
            #     param = module._random_projection_matrix
            #     param.requires_grad = True
            #     proj_params.append(param)
            pass

    # Define optimizers
    weight_optim = torch.optim.AdamW(
        weight_params, lr=args.weight_lr, weight_decay=5e-4
    )

    scheduler = CosineAnnealingLR(weight_optim, T_max=args.epochs, eta_min=1e-5)
    
    
    if args.kernel == "learned_projection":
        proj_optim = torch.optim.AdamW(proj_params, lr=args.proj_lr)
    else:
        proj_optim = None  # No projection optimizer for random_projection
        
    proj_scheduler = torch.optim.lr_scheduler.StepLR(proj_optim, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    

    # Training loop
    best_acc = 0.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        hashed_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            weight_optim.zero_grad()
            if proj_optim is not None:
                proj_optim.zero_grad()

            outputs = hashed_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Apply STE for learned projections:
            if args.kernel == "learned_projection":
                # Gradients on projection_matrix are already computed by STE in kernel implementation
                proj_optim.step()
            # Update backbone weights
            weight_optim.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                batch_time = time.time() - epoch_start
                avg_loss = running_loss / total
                acc = 100.0 * correct / total
                print(
                    f"Batch {batch_idx:03d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Acc: {acc:.2f}% | "
                    f"Elapsed: {batch_time:.2f}s"
                )

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        # Validation
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

        if test_acc > best_acc:
            best_acc = test_acc
            best_ckpt_name = f"{args.dataset.lower()}_resnet20.pth"
            best_save_path = OUTPUT_DIR / best_ckpt_name
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(hashed_model.state_dict(), best_save_path)
            print(f"*** New best model (Test Acc: {test_acc:.2f}%) saved to: {best_save_path} ***") 

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print(f"Best Test Acc across all epochs: {best_acc:.2f}%")

    

    # # Save final checkpoint
    # dataset_short = args.dataset.lower()
    # suffix = "learned" if args.kernel == "learned_projection" else "random"
    # ckpt_name = f"{dataset_short}_resnet20.pth"
    # save_path = OUTPUT_DIR / ckpt_name
    # torch.save(hashed_model.state_dict(), save_path)
    # print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
