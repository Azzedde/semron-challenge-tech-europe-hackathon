#progressively increase the hash length from 1024 → 2048 → 4096 as channels double.


submission_config_cifar10 = {
    # --- Top‐level conv: 3×3×3→16 (flattened length = 3*3*3 = 27; we pad to 256) ---
    "conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },

    # --- Layer1 (all 16→16 convs; flattened length = 16*3*3 = 144; pad to 256) ---
    "layer1.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 1024,
    },

    # --- Layer2 (16→32 conv, then 32→32 convs) ---
    # First conv in block 2: “layer2.0.conv1” (flattened 16*3*3 = 144; pad to 256)
    "layer2.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    # Second conv in block 2: “layer2.0.conv2” (flattened 32*3*3 = 288; pad to 256)
    # We still pad to 256, ignoring the fact 288 > 256; in practice most implementations do:
    # either tile smaller or pad to the next power of two (512). Here we assume pad to 256 and
    # let “excess” channels get truncated. If that worries you, change tile to 512. For now:
    "layer2.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    # Downsample conv (16→32), same pad/truncate logic as above
    "layer2.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer2.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },

    # --- Layer3 (32→64 conv, then 64→64 convs) ---
    "layer3.0.conv1": {  # (32*3*3 = 288; pad/truncate to 256)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer3.0.conv2": {  # (64*3*3 = 576; pad/truncate to 256—again, in practice you might choose 512)
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 4096,
    },

    # --- Fully Connected (64→num_classes) ---
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flatten 64‐dim vector, pad to 256
        "output_tile_size": 256,
        "hash_length": 1024,      # FC usually needs fewer bits—reduce to 1024
    },
}


submission_config_cifar100 = resnet20_cifar100_patch_config = {
    # ─── Top‐level conv1: 3×3×3→16 (flattened = 27 → pad to 256) ───
    "conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },

    # ─── Layer1 (all 16→16 convs; flattened = 16×3×3 = 144 → pad to 256) ───
    "layer1.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer1.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer1.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer1.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer1.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
    "layer1.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },

    # ─── Layer2 (16→32 convs; 2nd conv flattened = 32×3×3 = 288 → pad to 512) ───
    "layer2.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # actual flattened = 144, but pad to 256
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer2.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 512,   # actual flattened = 288, pad to 512 for exact coverage
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer2.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # 16→32 downsample: flattened = 144 → pad to 256
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer2.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # flattened = 288 → pad to 512? we choose pad-to-512 below
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer2.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 512,
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer2.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,   # again flatten=288, pad to 512
        "output_tile_size": 256,
        "hash_length": 4096,
    },
    "layer2.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 512,
        "output_tile_size": 256,
        "hash_length": 4096,
    },

    # ─── Layer3 (32→64 convs; 2nd conv flattened = 64×3×3 = 576 → pad to 1024) ───
    "layer3.0.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 512,   # flattened = 288 (32×3×3), pad to 512
        "output_tile_size": 256,
        "hash_length": 8192,
    },
    "layer3.0.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 1024,  # flattened = 576 (64×3×3), pad to 1024
        "output_tile_size": 256,
        "hash_length": 8192,
    },
    "layer3.0.downsample.0": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 512,   # 32→64 downsample: flattened = 288 → pad to 512
        "output_tile_size": 256,
        "hash_length": 8192,
    },
    "layer3.1.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 512,   # flattened = 576 → pad to 1024
        "output_tile_size": 256,
        "hash_length": 8192,
    },
    "layer3.1.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 1024,
        "output_tile_size": 256,
        "hash_length": 8192,
    },
    "layer3.2.conv1": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 512,   # flattened = 576 → pad to 1024
        "output_tile_size": 256,
        "hash_length": 8192,
    },
    "layer3.2.conv2": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 1024,
        "output_tile_size": 256,
        "hash_length": 8192,
    },

    # ─── Fully Connected (64→100); flattened = 64 → pad to 256 ───
    "fc": {
        "hash_kernel_type": "learned_projection",
        "input_tile_size": 256,
        "output_tile_size": 256,
        "hash_length": 2048,
    },
}


