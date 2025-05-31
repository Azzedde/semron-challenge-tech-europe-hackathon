import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from bitbybit.kernel import _HashKernel


T_HashableModule = TypeVar("T_HashableModule", bound="_HashableModule")


class _HashableModule(nn.Module, ABC):
    """Base class for PyTorch modules that use Hash Kernels for computation."""

    # Subclasses are responsible for initializing this appropriately.
    # The `| None` part allows it to be uninitialized or explicitly set to None.
    hash_kernel: "_HashKernel | None"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, "hash_kernel"):  # Check if subclass already set it
            self.hash_kernel = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the module."""
        pass

    @classmethod
    @abstractmethod
    def from_torch_module(
        cls: type[T_HashableModule], module: nn.Module, **kwargs
    ) -> T_HashableModule:
        """
        Creates a hashable module from an existing torch.nn module.
        The `**kwargs` is expected to contain `hash_kernel: _HashKernel`
        which will be instantiated by a factory before calling this method.
        """
        pass
