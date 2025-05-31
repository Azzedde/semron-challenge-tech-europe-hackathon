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
        """
        unit_vectors: shape (..., in_features)
        Returns binary codes ∈ {−1, 0, +1} of shape (..., hash_length),
        with STE so that gradients flow through the real-valued z.
        """
        # 1) Multiply by the learnable projection matrix: z = unit_vectors @ proj.T
        proj = self.projection_matrix  # shape [hash_length, in_features]
        z = torch.matmul(unit_vectors, proj.t())  # shape (..., hash_length)

        # 2) Binarize via sign(): gives −1, 0, or +1
        codes = torch.sign(z)

        # 3) Apply Straight-Through Estimator (STE):
        #    Forward: use codes.detach()  (no gradient through sign)
        #    Backward: propagate gradient wrt z (identity)
        return codes.detach() + (z - z.detach())

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        """
        codes_1:           shape (B,  hash_length)
        codes_2_matmuled:  shape (hash_length, N_out)
        Returns: estimated cosine similarity ∈ [−1, +1], shape (B, N_out).
        """
        # Exactly the same as in RandomProjKernel:
        dot = torch.matmul(codes_1, codes_2_matmuled)  # shape (B, N_out)
        cos_est = dot.to(torch.float32) / float(self.hash_length)
        return cos_est


    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"hash_length={self.hash_length}, "
            "type=learned_projection (with STE)"
        )
