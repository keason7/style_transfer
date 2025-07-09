"""Utility functions."""

import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.transforms import tensor2image


def read_yml(path):
    """Read YAML file.

    Args:
        path (str): Input path.

    Returns:
        dict: YAML dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yml(path, data):
    """Write YAML file.

    Args:
        path (str): Output path.
        data (dict): Data dictionary.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def get_timestamp(keyword):
    """Get current timestamp.

    Args:
        keyword (str): Timestamp beginning keyword.

    Returns:
        str: Current timestamp.
    """
    now = datetime.datetime.now()
    return f"{keyword}_{now.year:04d}_{now.month:02d}_{now.day:02d}-{now.hour:02d}_{now.minute:02d}_{now.second:02d}"


def get_paths(path_dataset, dataset_experiments):
    """Return absolute path of files for dataset experiments.

    Args:
        path_dataset (dict): Content and style dataset paths.
        dataset_experiments (list): List of dictionaries with content and style such as:
            [{'content': 'file1.png', 'style': ['A.png', 'B.jpg']}, {'content': 'file2.png', 'style': ['B.jpg']}, ...]

    Raises:
        ValueError: Image is not a file or does not exists.

    Returns:
        dict: File paths dictionary such as (where each path is a PosixPath):
            {
                'content': ['path/to/file1.png', 'path/to/file1.png', 'path/to/file2.png'],
                'style': ['path/to/A.png', 'path/to/B.jpg', 'path/to/B.jpg']
            }
    """
    paths = {"content": [], "style": []}
    for experiment in dataset_experiments:
        path_content_image = (Path(path_dataset["content"]) / experiment["content"]).resolve()

        for image_style in experiment["style"]:
            path_style_image = (Path(path_dataset["style"]) / image_style).resolve()

            for path_image in [path_content_image, path_style_image]:
                if not path_image.exists() or not path_image.is_file():
                    raise ValueError(f"Image is not a file or does not exists: {path_image}")

            paths["content"].append(path_content_image)
            paths["style"].append(path_style_image)

    return paths


def save_image(path_image, image, size, normalization):
    """Save an image.

    Args:
        path_image (str): Image path
        image (torch.tensor): Image tensor.
        size (tuple): Image target size.
        normalization (dict): Normalization coefficients.
    """
    # convert tensor back to np.array in [0, 255] range
    image = tensor2image(image, size, normalization) * 255
    image = image.astype(np.uint8)

    # convert to PIL image and save
    image = Image.fromarray(image)
    image.save(path_image)


def save_images(path_output, images, sizes, normalization):
    """Save images (content, style, target).

    Args:
        path_output (str): Output directory.
        images (dict): Images dictionary.
        sizes (dict): Sizes dictionary.
        normalization (dict): Normalization coefficients.
    """
    for key in images:
        path_image = Path(path_output) / f"{key}.png"
        save_image(str(path_image), images[key], sizes[key], normalization)


def save_run(path_run, current_images, current_sizes, config_parameters, tunning_parameters, normalization):
    """Save run data.

    Args:
        path_run (str): Output directory path.
        current_images (dict): Images dictionary (content, style, target).
        current_sizes (dict): Sizes dictionary (content, style, target).
        config_parameters (dict): Config dictionary.
        tunning_parameters (dict): Tunning parameters dictionary.
        normalization (dict): Normalization coefficients.
    """
    write_yml(str(path_run / "config.yml"), config_parameters)
    write_yml(str(path_run / "parameters.yml"), tunning_parameters)

    save_images(path_run, current_images, current_sizes, normalization)
