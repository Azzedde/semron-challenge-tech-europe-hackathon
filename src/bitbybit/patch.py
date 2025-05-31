from typing import Dict, Any, Optional

import torch.nn as nn

from .nn import HashLinear, HashConv2d, _HashableModule

from .kernel import (
    DEFAULT_KERNEL_TYPE as DEFAULT_GLOBAL_KERNEL_TYPE,
)  # Use from kernel package
from .kernel import DEFAULT_HASH_LENGTH as DEFAULT_GLOBAL_HASH_LENGTH
from .kernel import DEFAULT_INPUT_TILE_SIZE as DEFAULT_GLOBAL_INPUT_TILE_SIZE
from .kernel import DEFAULT_OUTPUT_TILE_SIZE as DEFAULT_GLOBAL_OUTPUT_TILE_SIZE


LayerConfig = Dict[str, Any]  # e.g., {"hash_length": 1024, "hash_kernel_type": "..."}
ModelConfig = Dict[str, LayerConfig]  # e.g., {"conv1": {...}, "layer1.0.conv1": {...}}


def _patch_model_recursive(
    model: nn.Module, config: Optional[ModelConfig] = None, current_path: str = ""
) -> nn.Module:
    """
    Internal recursive helper function to patch the model.
    It tracks the module path for configuration lookup.
    """
    if config is None:
        config = {}

    for name, child_module in model.named_children():
        # Construct the full path for the current module (e.g., "layer1.0.conv1")
        module_path = f"{current_path}.{name}" if current_path else name

        layer_specific_config = config.get(module_path, {})

        hash_length = layer_specific_config.get(
            "hash_length", DEFAULT_GLOBAL_HASH_LENGTH
        )
        hash_kernel_type = layer_specific_config.get(
            "hash_kernel_type", DEFAULT_GLOBAL_KERNEL_TYPE
        )
        input_tile_size = layer_specific_config.get(
            "input_tile_size", DEFAULT_GLOBAL_INPUT_TILE_SIZE
        )
        output_tile_size = layer_specific_config.get(
            "output_tile_size", DEFAULT_GLOBAL_OUTPUT_TILE_SIZE
        )

        if isinstance(child_module, nn.Linear) and not isinstance(
            child_module, _HashableModule
        ):  # Avoid re-patching
            new_linear_module = HashLinear.from_torch_module(
                child_module,
                hash_length=hash_length,
                hash_kernel_type=hash_kernel_type,
                input_tile_size=input_tile_size,
                output_tile_size=output_tile_size,
            )

            setattr(model, name, new_linear_module)

            print(
                f"Patched nn.Linear: {module_path} -> HashLinear("
                f"kernel={hash_kernel_type}, hash_len={hash_length}, "
                f"input_tile_size={input_tile_size}, "
                f"output_tile_size={output_tile_size})"
            )

        elif isinstance(child_module, nn.Conv2d) and not isinstance(
            child_module, _HashableModule
        ):  # Avoid re-patching
            new_conv2d_module = HashConv2d.from_torch_module(
                child_module,
                hash_length=hash_length,
                hash_kernel_type=hash_kernel_type,
                input_tile_size=input_tile_size,
                output_tile_size=output_tile_size,
            )

            setattr(model, name, new_conv2d_module)

            print(
                f"Patched nn.Conv2d: {module_path} -> HashConv2d("
                f"kernel={hash_kernel_type}, hash_len={hash_length}, "
                f"input_tile_size={input_tile_size}, "
                f"output_tile_size={output_tile_size})"
            )

        # Recursively patch child modules if they have children
        elif len(list(child_module.children())) > 0:
            _patch_model_recursive(child_module, config, module_path)
            # Note: No return needed here as modifications are in-place on child_module

    return model


def patch_model(model: nn.Module, config: Optional[ModelConfig] = None) -> nn.Module:
    """
    Patches a PyTorch model by replacing nn.Linear and nn.Conv2d modules
    with their Hash-based counterparts. Configuration can be provided per layer.

    The layer names in the config dictionary should match the names generated
    by `model.named_modules()` or `model.named_children()` recursively.
    For example: "conv1", "layer1.0.conv1", "fc".

    Args:
        model (nn.Module): The PyTorch model to patch.
        config (Optional[ModelConfig]): A dictionary mapping layer names (e.g.,
            "conv1", "layer1.0.conv2") to their specific hash kernel configurations.
            Each layer's configuration is a dictionary that can specify:
            - "hash_length" (int)
            - "hash_kernel_type" (str)
            - "input_tile_size" (int)
            - "output_tile_size" (int)
            If a parameter is not specified for a layer in its config entry,
            or if a layer is not present in the `config` dictionary,
            global default values (DEFAULT_GLOBAL_...) will be used for those parameters.

    Returns:
        nn.Module: The patched model (modified in-place).
    """
    return _patch_model_recursive(model, config, current_path="")


# Alias for convenience
patch = patch_model
