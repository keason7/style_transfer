"""Image transforms functions."""

import numpy as np
import torchvision.transforms as tf
from PIL import Image


def image2tensor(image, normalization, shape=None, max_size=512):
    """Convert input image to tensor.

    Args:
        image (PIL.Image): Input image.
        normalization (dict): Normalization coefficients.
        shape (tuple or None, optional): Imposed output size. Defaults to None.
        max_size (int, optional): Max size. Defaults to 512.

    Returns:
        torch.Tensor: Output tensor.
    """
    # imposed output size
    if shape is not None:
        size = shape

    # resize based on max_size
    else:
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

    transform = tf.Compose(
        [
            tf.Resize(size),
            tf.ToTensor(),
            tf.Normalize(normalization["mean"], normalization["std"]),
        ]
    )

    # discard alpha channel and add batch dim
    return transform(image)[:3, :, :].unsqueeze(0)


def tensor2image(tensor, size, normalization):
    """Convert input tensor to image.

    Args:
        tensor (torch.Tensor): Input tensor.
        size (tuple): Output size.
        normalization (dict): Normalization coefficients.

    Returns:
        np.array: Output image.
    """
    transform = tf.Compose([tf.Resize(min(size))])

    # detach from device, remove batch dim, and resize
    image = tensor.to("cpu").clone().detach()
    image = transform(image)

    # convert to numpy array and change dimension from (C, H, W) to (H, W, C)
    image = image.numpy().squeeze().transpose(1, 2, 0)

    # denormalize and clip values to [0, 1] range
    image = image * np.array(normalization["std"]) + np.array(normalization["mean"])
    image = image.clip(0, 1)

    return image


def prepare_images(paths, config):
    """Load content and style images, convert them to tensors, and store original sizes.

    Args:
        paths (dict): Dictionary with keys "content_paths" and "style_paths",
            each containing a list of file paths.
        config (dict): Configuration dictionary.

    Returns:
        dict: Dictionary with "content" and "style" keys containing tensors.
        dict: Original sizes of images under "content" and "style" keys.
    """
    images = {"content": [], "style": []}
    sizes = {"content": [], "style": []}

    for idx, path_content in enumerate(paths["content"]):
        image_content = Image.open(path_content).convert("RGB")
        image_style = Image.open(paths["style"][idx]).convert("RGB")

        sizes["content"].append(np.array(image_content).shape[0:2])
        sizes["style"].append(np.array(image_style).shape[0:2])

        images["content"].append(
            image2tensor(
                image_content,
                normalization=config["normalize"],
            )
        )
        images["style"].append(
            image2tensor(
                image_style,
                normalization=config["normalize"],
                shape=images["content"][-1].shape[-2:],
            )
        )

    return images, sizes
