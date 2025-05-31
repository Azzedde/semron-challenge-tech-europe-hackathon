from pathlib import Path
import torch
import time

from bitbybit.utils.models import get_backbone
from bitbybit.utils.data import get_loaders, CIFAR10_MEAN, CIFAR10_STD
import bitbybit as bb
from bitbybit.config.resnet20 import resnet20_full_patch_config

OUTPUT_DIR = Path(__file__).parent / "sampler_checkpoints"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Smaller batch size and only CIFAR10
    train_loader, test_loader = get_loaders(
        dataset_name="CIFAR10",
        data_dir=Path(__file__).parent / "data",
        batch_size=32,  # Reduced from 128
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        num_workers=2,
        pin_memory=True,
    )

    model = get_backbone("cifar10_resnet20")
    hashed_model = bb.patch_model(model, config=resnet20_full_patch_config)
    
    # Set up optimizers
    weight_params = []
    proj_params = []
    for module in hashed_model.modules():
        if hasattr(module, 'weight') and module.weight.requires_grad:
            weight_params.append(module.weight)
        if hasattr(module, 'projection_matrix'):
            module._random_projection_matrix.requires_grad = True
            proj_params.append(module._random_projection_matrix)
    
    weight_optim = torch.optim.SGD(weight_params, lr=0.1, momentum=0.9)
    proj_optim = torch.optim.Adam(proj_params, lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Fewer epochs
    num_epochs = 1
    for epoch in range(num_epochs):
        hashed_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start = time.time()
            
            weight_optim.zero_grad()
            proj_optim.zero_grad()
            
            outputs = hashed_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Apply STE for projection gradients
            for module in hashed_model.modules():
                if hasattr(module, 'projection_matrix'):
                    module._random_projection_matrix.grad = module._random_projection_matrix.grad
            
            weight_optim.step()
            proj_optim.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                batch_time = time.time() - batch_start
                print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}s")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

        # Quick test evaluation
        hashed_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = hashed_model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        print(f"Test Acc: {100.*test_correct/test_total:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

    # Save model
    torch.save(hashed_model.state_dict(), OUTPUT_DIR / "cifar10_resnet20_sampler.pth")
    print(f"Model saved to {OUTPUT_DIR/'cifar10_resnet20_sampler.pth'}")

if __name__ == "__main__":
    main()