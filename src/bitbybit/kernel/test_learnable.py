try:
    from learned import LearnedProjKernel
except ImportError:
    # For direct script execution
    from learned import LearnedProjKernel

import torch

# Hash‐length (number of bits), input feature size, and output feature count
K = 32    # hash_length
in_f = 16
out_f = 10

# Instantiate the learnable projection kernel
kernel = LearnedProjKernel(in_features=in_f, out_features=out_f, hash_length=K)

# Create a batch of 4 random input vectors (shape: [4, in_f])
x = torch.randn(4, in_f)

# Create 10 random weight vectors (shape: [out_f, in_f])
W = torch.randn(out_f, in_f)

# Compute the approximate output via binary hashing + cosine estimation
# Assuming kernel(x, W) returns shape [4, 10]
y_approx = kernel(x, W)

# Compute the exact floating‐point dot products for comparison
# True output has shape [4, 10]
y_true = x @ W.t()

# Print the first 5 values of the first example (row 0) for both true vs. approx
print("True[0, :5]   = ", y_true[0, :5])
print("Approx[0, :5] = ", y_approx[0, :5])
