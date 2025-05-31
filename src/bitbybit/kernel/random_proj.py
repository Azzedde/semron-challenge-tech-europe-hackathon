import torch
import math

from _base import _HashKernel


class RandomProjKernel(_HashKernel):
    """
    Random‐Projection Hash Kernel:
    - Uses a fixed, random projection matrix (Gaussian entries) to map
      each input or weight vector to a hash of length `hash_length`.
    - Approximates cosine similarity via Hamming‐distance on {−1,+1} codes.
    """

    def __init__(self, in_features: int, out_features: int, hash_length: int, **kwargs) -> None:
        super().__init__(in_features, out_features, hash_length)

        # 1. Create a random projection matrix C of shape (K, in_features),
        #    where K = hash_length. We draw each entry ~ N(0,1).
        initial_proj_mat = torch.randn(hash_length, self.in_features)

        # 2. We register it as a buffer (not a parameter) so it stays fixed
        #    and moves with the module (to CPU/GPU) but is never updated by optimizer.
        self.register_buffer("_random_projection_matrix", initial_proj_mat)

    @property
    def projection_matrix(self) -> torch.Tensor:
        # Return the buffer that we called "_random_projection_matrix"
        return self._random_projection_matrix

    def _compute_codes_internal(self, unit_vectors: torch.Tensor) -> torch.Tensor:
        """
        Maps each row of `unit_vectors` (shape (..., in_features)) to a {−1,+1}^K code:
          1) proj = unit_vectors @ C.T      # shape (..., K)
          2) codes = sign(proj), replaced 0 → +1 so codes are strictly {−1,+1}.
        Returns:
          Tensor of shape (..., K) with values in {−1, +1}.
        """

        # 1. Pull out C (shape: K × in_features)
        C = self.projection_matrix  # (K, in_features)

        # 2. Multiply: if unit_vectors is (B, in_features) or (out_features, in_features),
        #    then result is (… , K). In batch use-case, we expect shape (B, K).
        #    Note: `@` on the last two dims does an inner product on in_features.
        proj = unit_vectors @ C.T  # shape (..., K)

        # 3. Take sign. PyTorch's torch.sign(x) gives {−1, 0, +1}. We want no zeros.
        codes = torch.sign(proj)

        # 4. Wherever proj == 0, sign() yields 0. Replace those zeros with +1:
        #    (We assume zero is a measure‐zero event when sampling from Gaussians,
        #    but numeric cancellation can produce exact zero in rare cases.)
        codes[codes == 0] = 1

        return codes  # shape (..., K), values in {−1, +1}

    def _estimate_cosine_internal(
        self, codes_1: torch.Tensor, codes_2_matmuled: torch.Tensor
    ) -> torch.Tensor:
        """
        Given:
          codes_1           of shape (B, K)         in {−1,+1}
          codes_2_matmuled  of shape (K, out_features)  in {−1,+1}

        1) Compute signed dot‐product: dot = codes_1 @ codes_2_matmuled  # shape (B, out_features)
           Each entry is an integer in [−K, +K].
        2) Compute Hamming distance: 
             H = (K − dot) / 2                                      # shape (B, out_features)
        3) The fraction of differing bits is H/K.  
           Estimate angle θ ≈ π * (H/K), so estimate cos(θ) = cos(π * (H/K)).
        4) Return cos_est of shape (B, out_features).
        """

        # 1. Extract K (hash_length)
        K = self.hash_length

        # 2. Signed dot‐product: 
        #    codes_1 (B, K) @ codes_2_matmuled (K, out_features) → (B, out_features)
        dot = codes_1 @ codes_2_matmuled  # dtype: same as codes, e.g. float32, but values are integers in [−K, +K]

        # 3. Compute Hamming distance: H = (K − dot) / 2
        #    Since dot ∈ [−K, +K], (K − dot) ∈ [0, 2K], so H ∈ [0, K].
        hamming = (K - dot) * 0.5  # shape (B, out_features)

        # 4. Fraction of bits that differ = H / K
        ratio = hamming / K  # shape (B, out_features)

        # 5. cos_est = cos(pi * ratio)
        cos_est = torch.cos(math.pi * ratio)  # shape (B, out_features)

        return cos_est

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"hash_length={self.hash_length}, type=random_projection"
        )
