try:
    from random_proj import RandomProjKernel
except ImportError:
    # For direct script execution
    from random_proj import RandomProjKernel
import torch
K = 32    # hash_length
in_f = 16
out_f = 10

kernel = RandomProjKernel(in_f, out_f, K)

x = torch.randn(4, in_f)       # batch of 4 inputs
W = torch.randn(out_f, in_f)   # 10 weight vectors
y_approx = kernel(x, W)  # shape (4, 10)

# Exact (floating‐point) dot products:
y_true = x @ W.T  # shape (4, 10)

# Print a small slice to see approximate vs. true:
print("True[0, :5]      = ", y_true[0, :5])
print("Approx[0, :5]    = ", y_approx[0, :5])

