import torch
import torch.nn as nn
import math

from ._base import _HashKernel


class LearnedProjKernel(_HashKernel):

    def __init__(
        self, in_features: int, out_features: int, hash_length: int, **kwargs
    ) -> None:
        super().__init__(in_features, out_features, hash_length)

        # LSH projection matrix (learnable)
        initial_proj_mat = torch.randn(hash_length, self.in_features)
        self._learnable_projection_matrix = nn.Parameter(initial_proj_mat)

    @property
    def projection_matrix(self) -> torch.Tensor:
        return self._learnable_projection_matrix

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Expected to be implemented by Challenge Participants")

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Expected to be implemented by Challenge Participants")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=learned_projection (STE TODO)"
        )
