from typing import Literal

import torch
import torch.nn as nn


ModelHubName = Literal[
    "cifar10_resnet20", "cifar100_resnet20", "cifar10_vgg11_bn", "cifar100_vgg11_bn"
]


def get_backbone(
    name: ModelHubName,
    pretrained: bool = True,
) -> nn.Module:
    """
    Loads a specific model from chenyaofo/pytorch-cifar-models via torch.hub.

    Args:
        name: Model name.
        pretrained: If True, loads pretrained weights.

    Returns:
        A PyTorch nn.Module model.
    """
    model_kwargs = {"pretrained": pretrained}

    model: nn.Module = torch.hub.load(
        repo_or_dir="chenyaofo/pytorch-cifar-models",
        model=name,
        source="github",
        **model_kwargs,
    )

    return model
