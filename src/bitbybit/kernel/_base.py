import torch
import torch.nn as nn

from abc import ABC, abstractmethod

EPS = 1e-9


class _HashKernel(nn.Module, ABC):
    """
    Base class for Hash Kernels that replace standard matrix multiplication (GEMM/GEMV)
    with a hash-based approximation. These kernels store the primary layer weights
    and the projection matrix for hashing.
    """

    def __init__(self, in_features: int, out_features: int, hash_length: int) -> None:
        super().__init__()
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError("in_features must be a positive integer.")

        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError("out_features must be a positive integer.")

        if not isinstance(hash_length, int) or hash_length <= 0:
            raise ValueError("hash_length must be a positive integer.")

        self.in_features = in_features
        self.out_features = out_features
        self.hash_length = hash_length

    @property
    @abstractmethod
    def projection_matrix(self) -> torch.Tensor:
        """The LSH projection matrix (K, N_feat_of_vector_to_hash)."""
        pass

    @abstractmethod
    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        """
        Computes binary hash codes for unit input vectors using self.projection_matrix.
        Args:
            vectors (torch.Tensor): Input tensor of shape (..., features_dim).
                                            features_dim must match self.projection_matrix's second dim.
        Returns:
            torch.Tensor: Binary codes {-1, 1} of shape (..., self.hash_length).
        """
        pass

    @abstractmethod
    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimates cosine similarity based on hash codes.
        Args:
            codes_1 (torch.Tensor): Hash codes for first set of vectors (e.g., B, K).
            codes_2_matmuled (torch.Tensor): Hash codes for second set, transposed for matmul (e.g., K, M).
        Returns:
            torch.Tensor: Estimated cosine similarities (e.g., B, M).
        """
        pass

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Performs the hash-based matrix multiplication: y ~ x @ W.T + b
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (Batch, out_features).
        """
        current_eps = torch.finfo(x.dtype).eps if x.dtype.is_floating_point else EPS

        # x shape: (B, N_in)
        x_norms = x.norm(dim=-1, keepdim=True) + current_eps
        x_unit = x / x_norms  # (B, N_in)

        # weights shape: (N_out, N_in)
        w_norms = weights.norm(dim=-1, keepdim=True) + current_eps
        w_unit = weights / w_norms  # (N_out, N_in)

        # x_unit @ self.projection_matrix.T should give (B, K)
        codes_x = self._compute_codes_internal(x_unit)  # (B, K)

        # w_unit @ self.projection_matrix.T should give (N_out, K)
        codes_w_prime = self._compute_codes_internal(w_unit)  # (N_out, K)
        codes_w_matmuled = codes_w_prime.transpose(-2, -1)  # (K, N_out)

        # (B,K) @ (K, N_out) -> (B, N_out)
        cos_est = self._estimate_cosine_internal(codes_x, codes_w_matmuled)

        # x_norms (B,1), w_norms.T (1, N_out)
        return (x_norms * w_norms.transpose(-2, -1)) * cos_est  # (B, N_out)
