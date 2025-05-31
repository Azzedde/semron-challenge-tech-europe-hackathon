from collections import deque
from typing import Set

import torch.nn as nn

from bitbybit.kernel import _HashKernel


def _score(
    acc_drop: float,
    hdops: float,
    flops: float,
    *,
    delta: float = 0.125,
    gamma: float = 0.85,
) -> float:
    """
    Parameters
    ----------
    acc_drop : float   Accuracy drop of the compressed model.
    hdops    : float   Hashed-OPs (summed).
    flops    : float   Reference FLOPs (summed).
    delta    : float   Allowed accuracy drop (hyperparameter).
    gamma    : float   Controls how sharply the energy term penalises imbalance.
                       • gamma = 0.5 .. 1 → forgiving
                       • gamma > 1        → stricter
    """
    A = max(0.0, 1.0 - acc_drop / delta)

    if hdops == 0 or flops == 0:  # degenerate: no compute
        return 0.0

    if hdops < flops:
        raise ValueError("hdops must be greater than or equal to flops")

    ratio = flops / hdops
    r_balanced = ratio if ratio <= 1 else 1 / ratio
    r_pow = r_balanced**gamma
    E = 2.0 * r_pow / (1.0 + r_pow)  # in (0, 1]

    return (2 * A * E) / (A + E) if (A + E) > 0 else 0.0


def _hash_kernels(root: nn.Module) -> Set["_HashKernel"]:
    """
    Return every unique `_HashKernel` instance reachable from *root*,
    whether or not it’s a registered `nn.Module`.
    """
    seen_ids: Set[int] = set()
    kernels: Set["_HashKernel"] = set()
    q = deque([root])

    while q:
        obj = q.popleft()

        # 1. kernel?
        if isinstance(obj, _HashKernel) and id(obj) not in seen_ids:
            kernels.add(obj)
            seen_ids.add(id(obj))
            # no need to traverse inside a kernel

        # 2. nn.Module: traverse children and its __dict__
        elif isinstance(obj, nn.Module):
            q.extend(obj._modules.values())  # registered children
            q.extend(obj.__dict__.values())  # everything else

        # 3. containers
        elif isinstance(obj, (list, tuple, set)):
            q.extend(obj)
        elif isinstance(obj, dict):
            q.extend(obj.values())

    return kernels


def calculate_submission_score(
    model: nn.Module, acc_drop: float = 0.0, *, delta: float = 0.15, gamma: float = 0.5
) -> float:
    """
    Compute the global score for any model that
    contains `_HashKernel` instances.

    Parameters
    ----------
    model       : nn.Module   Model to analyse.
    acc_drop    : float       Accuracy drop of the compressed (hashed) model.
    delta, gamma: float       hyper-parameters (keep at default).
    """
    total_hdops = total_flops = 0.0

    for k in _hash_kernels(model):
        total_hdops += 2.0 * k.hash_length * k.in_features
        total_flops += 2.0 * k.in_features * k.out_features

    return _score(acc_drop, total_hdops, total_flops, delta=delta, gamma=gamma)
