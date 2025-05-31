import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _pair
import math
from typing import Self

from ._base import _HashableModule
from bitbybit.kernel import kernel_factory, _HashKernel

DEFAULT_KERNEL_TYPE = "random_projection"
DEFAULT_HASH_LENGTH = 4096


class HashConv2d(_HashableModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        input_tile_size: int = 128,
        output_tile_size: int = 128,
        hash_kernel_type: str | None = None,
        hash_length: int = 128,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Convert scalar values to pairs
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.input_tile_size = input_tile_size
        self.output_tile_size = output_tile_size
        self.hash_kernel_type = hash_kernel_type

        # Calculate the flattened input size after im2col
        # Each output position sees kernel_size[0] * kernel_size[1] * in_channels values
        self.im2col_input_size = (
            self.kernel_size[0] * self.kernel_size[1] * in_channels // groups
        )

        # Initialize hash kernels for tiled computation
        # Only create kernels if hash_kernel_type is specified
        self.hash_kernels = nn.ModuleList()

        if hash_kernel_type is not None:
            for in_start in range(0, self.im2col_input_size, input_tile_size):
                row_kernels = nn.ModuleList()
                in_tile_size = min(input_tile_size, self.im2col_input_size - in_start)

                for out_start in range(0, out_channels, output_tile_size):
                    out_tile_size = min(output_tile_size, out_channels - out_start)

                    kernel = kernel_factory(
                        hash_kernel_type, in_tile_size, out_tile_size, hash_length
                    )
                    row_kernels.append(kernel)
                self.hash_kernels.append(row_kernels)

        factory_kwargs = {"device": device, "dtype": dtype}

        # Weight tensor: (out_channels, in_channels // groups, kernel_height, kernel_width)
        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *self.kernel_size),
                **factory_kwargs,
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Initializes weight and bias parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels // self.groups
            for k in self.kernel_size:
                fan_in *= k
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using either hash kernels or standard convolution.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_channels, Height, Width).
        Returns:
            torch.Tensor: Output tensor of shape (Batch, out_channels, out_Height, out_Width).
        """
        if self.hash_kernel_type is None:
            # Fallback to standard convolution
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        batch_size, in_channels, height, width = x.shape

        # Calculate output dimensions
        out_height = (
            height
            + 2 * self.padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        out_width = (
            width
            + 2 * self.padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1

        # Use im2col to unfold the input
        # This converts the convolution into a matrix multiplication
        x_unfolded = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )  # Shape: (batch_size, in_channels * kernel_h * kernel_w, out_height * out_width)

        # Transpose to get (batch_size, out_height * out_width, in_channels * kernel_h * kernel_w)
        x_unfolded = x_unfolded.transpose(1, 2)

        # Flatten weight to (out_channels, in_channels * kernel_h * kernel_w)
        weight_flat = self.weight.view(self.out_channels, -1)

        # Initialize output
        output_flat = torch.zeros(
            (batch_size, out_height * out_width, self.out_channels),
            device=x.device,
            dtype=x.dtype,
        )

        # Apply tiled hash kernel computation
        for i, in_start in enumerate(
            range(0, self.im2col_input_size, self.input_tile_size)
        ):
            in_end = min(in_start + self.input_tile_size, self.im2col_input_size)

            for j, out_start in enumerate(
                range(0, self.out_channels, self.output_tile_size)
            ):
                out_end = min(out_start + self.output_tile_size, self.out_channels)

                x_tile = x_unfolded[
                    :, :, in_start:in_end
                ]  # (batch, out_positions, in_tile_size)
                w_tile = weight_flat[
                    out_start:out_end, in_start:in_end
                ]  # (out_tile_size, in_tile_size)

                # Apply hash kernel: x_tile @ w_tile.T
                # Reshape for hash kernel: (batch * out_positions, in_tile_size)
                x_tile_flat = x_tile.reshape(-1, x_tile.size(-1))

                if i < len(self.hash_kernels) and j < len(self.hash_kernels[i]):
                    # Use hash kernel
                    result = self.hash_kernels[i][j](x_tile_flat, w_tile)
                    result = result.reshape(batch_size, out_height * out_width, -1)
                else:
                    # Fallback to standard matrix multiplication
                    result = torch.matmul(x_tile, w_tile.T)

                # Accumulate results
                output_flat[:, :, out_start:out_end] += result

        # Add bias if present
        if self.bias is not None:
            output_flat = output_flat + self.bias

        # Reshape back to (batch_size, out_channels, out_height, out_width)
        output = output_flat.transpose(1, 2).reshape(
            batch_size, self.out_channels, out_height, out_width
        )

        return output

    @classmethod
    def from_torch_module(
        cls: type[Self],
        module: nn.Conv2d,
        input_tile_size: int = 128,
        output_tile_size: int = 128,
        hash_kernel_type: str | None = None,
        hash_length: int = 128,
        **kwargs,
    ) -> Self:
        """
        Creates a HashConv2d layer from an nn.Conv2d module.
        Args:
            module (nn.Conv2d): The original nn.Conv2d module.
            input_tile_size (int): Size of input tiles for hash kernel computation.
            output_tile_size (int): Size of output tiles for hash kernel computation.
            hash_kernel_type (str | None): Type of hash kernel to use.
            hash_length (int): Length of hash for the kernels.
        Returns:
            HashConv2d: The new HashConv2d layer with copied weights and bias.
        """
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"Expected nn.Conv2d, but got {type(module).__name__}")

        new_module = cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
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
        repr_str = (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}, "
            f"bias={self.bias is not None}"
        )
        if self.hash_kernel_type:
            repr_str += f", hash_kernel_type={self.hash_kernel_type}"
            if (
                self.hash_kernels
                and len(self.hash_kernels) > 0
                and len(self.hash_kernels[0]) > 0
            ):
                repr_str += f', hash_length={getattr(self.hash_kernels[0][0], "hash_length", "unknown")}'
        return repr_str
