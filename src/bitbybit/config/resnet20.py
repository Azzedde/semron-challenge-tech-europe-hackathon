# config.py

# For CIFAR-10 / ResNet-20
submission_config_cifar10 = {
    # ─── Top‐level conv1: 3×3×3 → 16 (flattened = 3*3*3 = 27; pad to 256) ───
    "conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,     # was 4096 → now 512 → now 256
    },

    # ─── Layer1 (all 16→16 convs; flattened = 16*3*3 = 144; pad to 256) ───
    # We keep K = 512 throughout Stage 1
    "layer1.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,
    },

    # ─── Layer2 (16→32 conv, then 32→32 convs) ───
    # Stage 2: increase K to 1024
    "layer2.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 16*3*3 = 144 → pad to 256
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer2.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 32*3*3 = 288 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer2.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # 16→32 downsample: flattened = 16*3*3 = 144 → pad 256
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer2.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 32*3*3 = 288 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # again 32*3*3 = 288 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 32*3*3 = 288 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 32*3*3 = 288 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 512,
    },

    # ─── Layer3 (32→64 conv, then 64→64 convs) ───
    # Stage 3: increase K to 2048
    "layer3.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 32*3*3 = 288 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer3.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 64*3*3 = 576 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # 32→64 downsample: flattened = 32*3*3=288 → pad 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 64*3*3 = 576 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 64*3*3 = 576 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 64*3*3 = 576 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 64*3*3 = 576 → pad/truncate to 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },

    # ─── Fully Connected (64→10): flattened = 64 → pad to 256 ───
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 256,     # final FC needs fewer bits—set K=512 → now 256
    },
}


# For CIFAR-100, we do the same stage‐wise hash lengths:
submission_config_cifar100 = {
    # ─── Top‐level conv1: 3×3×3→16 (flattened = 27 → pad 256) ───
    "conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },

    # ─── Layer1 (16→16 convs; flattened = 144 → pad 256) ───
    "layer1.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },

    # ─── Layer2 (16→32 and 32→32) ───
    "layer2.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 144 → pad 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer2.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 288 → pad 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer2.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 144 → pad 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer2.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 288 → pad 256
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },

    # ─── Layer3 (32→64, then 64→64) ───
    "layer3.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 288 → pad 256
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer3.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 576 → pad 256
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 288 → pad 256
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,  # flattened 576 → pad 256
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },

    # ─── Fully Connected (64→100; flattened = 64 → pad 256) ───
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 512,
    },
}