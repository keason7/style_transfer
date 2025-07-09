"""Pytorch models helpers."""

import torch.nn as nn
from torchvision.models import vgg16, vgg19

_available_models = {
    "vgg16": vgg16,
    "vgg19": vgg19,
}


def get_vgg_model(model_name):
    """Get a VGG model features extractor with pretrained ImageNet weights.

    Args:
        model_name (str): Model name.

    Returns:
        torch.nn.Module: Pytorch model.
    """
    # NOTE with other models in available models, selecting features extractor might differ
    return _available_models[model_name](weights="DEFAULT").features


def get_model(model_name):
    """Get model features extractor within available model.

    Args:
        model_name (str): Model name.

    Raises:
        ValueError: Unknown model name.

    Returns:
        torch.nn.Module: Pytorch model.
    """
    keys = _available_models.keys()
    if model_name not in keys:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_name in ["vgg16", "vgg19"]:
        return get_vgg_model(model_name)


def set_pooling(model, pooling, pool_layers, kernel, stride):
    """Set pooling layers.

    Args:
        model (torch.nn.Module): Input model.
        pooling (str): Pooling type.
        pool_layers (list): Pooling layers keys.
        kernel (int): Kernel size of pooling.
        stride (int): Stride of pooling.

    Returns:
        torch.nn.Module: Model.
    """
    # iterate over model layers
    # NOTE with other models in available models, selecting pooling might differ
    for i, content in enumerate(model._modules.items()):
        name, _ = content
        if name in pool_layers:
            if pooling == "average":
                model[i] = nn.AvgPool2d(kernel_size=kernel, stride=stride)

            if pooling == "max":
                model[i] = nn.MaxPool2d(kernel_size=kernel, stride=stride)
    return model


def load_model(model_name, pooling_layers, pooling=None):
    """Load a pytorch model within available models. Load model, select feature extractor
        and replace pooling layers if needed.

    Args:
        model_name (str): Model name.
        pooling_layers (list): List of str indexes that indicate where are the pooling layers.
        pooling (str or None, optional): Pooling type. Defaults to None.

    Raises:
        ValueError: Invalid pooling type.

    Returns:
        torch.nn.Module: Model.
    """
    __available_poolings = ["average", "max"]

    model = get_model(model_name)

    if pooling is not None:
        if pooling not in __available_poolings:
            raise ValueError(f"Invalid pooling type. Valid types are: {__available_poolings}.")

        if pooling == "average":
            model = set_pooling(model, pooling, pooling_layers, 2, 2)

        if pooling == "max":
            model = set_pooling(model, pooling, pooling_layers, 2, 2)

    # freezing parameters
    for param in model.parameters():
        param.requires_grad_(False)

    return model
