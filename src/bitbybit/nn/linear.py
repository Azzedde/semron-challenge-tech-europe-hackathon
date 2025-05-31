import math
from typing import Self

import torch
import torch.nn.functional as F
import torch.nn as nn

from ._base import _HashableModule
from bitbybit.kernel import _HashKernel, kernel_factory

DEFAULT_KERNEL_TYPE = "random_projection"
DEFAULT_HASH_LENGTH = 4096


class HashLinear(_HashableModule):

    in_features: int
    out_features: int
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_tile_size: int = 128,
        output_tile_size: int = 128,
        hash_kernel_type: str | None = None,
        hash_length: int = DEFAULT_HASH_LENGTH,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_tile_size = input_tile_size
        self.output_tile_size = output_tile_size
        self.hash_kernel_type = hash_kernel_type

        # Initialize hash kernels for tiled computation
        # Only create kernels if hash_kernel_type is specified
        self.hash_kernels: nn.ModuleList = nn.ModuleList()

        if hash_kernel_type is not None:
            for in_start in range(0, in_features, input_tile_size):
                row_kernels = nn.ModuleList()
                in_tile_size = min(input_tile_size, in_features - in_start)

                for out_start in range(0, out_features, output_tile_size):
                    out_tile_size = min(output_tile_size, out_features - out_start)

                    kernel = kernel_factory(
                        hash_kernel_type, in_tile_size, out_tile_size, hash_length
                    )
                    row_kernels.append(kernel)
                self.hash_kernels.append(row_kernels)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initializes weight and bias parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using either hash kernels or standard linear operation.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (Batch, out_features).
        """
        if self.hash_kernel_type is None:
            # Fallback to standard linear operation
            return F.linear(x, self.weight, self.bias)

        # Initialize output tensor
        output = torch.zeros(
            (x.shape[0], self.out_features), device=x.device, dtype=x.dtype
        )

        # Apply tiled hash kernel computation
        for i, in_start in enumerate(range(0, self.in_features, self.input_tile_size)):
            in_end = min(in_start + self.input_tile_size, self.in_features)

            for j, out_start in enumerate(
                range(0, self.out_features, self.output_tile_size)
            ):
                out_end = min(out_start + self.output_tile_size, self.out_features)

                x_tile = x[:, in_start:in_end]
                w_tile = self.weight[out_start:out_end, in_start:in_end]

                # Use the hash kernel for matrix multiplication
                result = self.hash_kernels[i][j](x_tile, w_tile)

                # Accumulate results
                output[:, out_start:out_end] += result

        if self.bias is not None:
            output = output + self.bias

        return output

    @classmethod
    def from_torch_module(
        cls: type[Self],
        module: nn.Linear,
        input_tile_size: int = 64,
        output_tile_size: int = 64,
        hash_kernel_type: str | None = None,
        hash_length: int = DEFAULT_HASH_LENGTH,
        **kwargs,
    ) -> Self:
        """
        Creates a HashLinear layer from an nn.Linear module.
        Args:
            module (nn.Linear): The original nn.Linear module.
            input_tile_size (int): Size of input tiles for hash kernel computation.
            output_tile_size (int): Size of output tiles for hash kernel computation.
            hash_kernel_type (str | None): Type of hash kernel to use.
            hash_length (int): Length of hash for the kernels.
        Returns:
            HashLinear: The new HashLinear layer with copied weights and bias.
        """
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear, but got {type(module).__name__}")

        new_module = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=(module.bias is not None),
            input_tile_size=input_tile_size,
            output_tile_size=output_tile_size,
            hash_kernel_type=hash_kernel_type,
            hash_length=hash_length,
            device=module.weight.device,
            dtype=module.weight.dtype,
            **kwargs,
        )

        # Copy weight and bias from the original module
        with torch.no_grad():
            new_module.weight.copy_(module.weight)
            if module.bias is not None and new_module.bias is not None:
                new_module.bias.copy_(module.bias)

        return new_module

    def extra_repr(self) -> str:
        repr_str = f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        repr_str += f", input_tile_size={self.input_tile_size}, output_tile_size={self.output_tile_size}"
        if self.hash_kernel_type:
            repr_str += f", hash_kernel_type={self.hash_kernel_type}"
            if (
                self.hash_kernels
                and len(self.hash_kernels) > 0
                and len(self.hash_kernels[0]) > 0
            ):
                repr_str += f', hash_length={getattr(self.hash_kernels[0][0], "hash_length", "unknown")}'
        return repr_str
