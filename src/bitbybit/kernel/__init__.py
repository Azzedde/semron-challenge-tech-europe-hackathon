import torch

from ._base import _HashKernel
from .learned import LearnedProjKernel
from .random import RandomProjKernel

DEFAULT_KERNEL_TYPE: str | None = "random_projection"
DEFAULT_HASH_LENGTH: int = 2**12

DEFAULT_INPUT_TILE_SIZE: int = 128
DEFAULT_OUTPUT_TILE_SIZE: int = 128


def kernel_factory(
    kernel_type_str: str | None,
    kernel_in_features: int,
    kernel_out_features: int,
    hash_length: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    **kernel_specific_args,
) -> _HashKernel | None:

    if kernel_type_str is None or kernel_type_str.lower() == "none":
        return None  # Explicitly no kernel requested

    if kernel_type_str.lower() == "random_projection":
        return RandomProjKernel(
            in_features=kernel_in_features,
            out_features=kernel_out_features,
            hash_length=hash_length,
            **kernel_specific_args,
        )
    elif kernel_type_str.lower() == "learned_projection":
        return LearnedProjKernel(
            in_features=kernel_in_features,
            out_features=kernel_out_features,
            hash_length=hash_length,
            **kernel_specific_args,
        )
    else:
        print(
            f"Warning: Unknown kernel type '{kernel_type_str}'. "
            "No kernel will be instantiated for this configuration."
        )
        return None


__all__ = [
    "_HashKernel",
    "RandomProjKernel",
    "LearnedProjKernel",
    "kernel_factory",
    "DEFAULT_KERNEL_TYPE",
    "DEFAULT_HASH_LENGTH",
]
