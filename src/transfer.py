"""Style transfer functions."""

from itertools import product
from math import prod
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models import load_model
from src.utils import get_timestamp, save_image, save_run


def get_device():
    """Get pytorch device (CPU or GPU).

    Returns:
        torch.device: Pytorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def map_layer(layer_names, layer_mapping):
    """Map layers human-readable names to layer names in model.

    Args:
        layer_names (list): List of layers to select such as: [layer_1, layer_4, ...].
        layer_mapping (dict): Layer mapping dictionary such as: {"layer_1": "0", "layer_2": "1", ...}.

    Returns:
        list: Layer list such as: ["0", "3", ...].
    """
    mapped_layers = []
    for layer_name in layer_names:
        mapped_layers.append(layer_mapping[layer_name])

    return mapped_layers


def gram_matrix(tensor):
    """Compute Gram matrix.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Gram matrix.
    """
    # reshape from [B, C, H, W] to [C, H * W]
    _, c, h, w = tensor.size()
    tensor = tensor.reshape(c, h * w)

    # return dot product
    return tensor @ tensor.t()


def get_features(image, model, layer_names, layer_mapping):
    """Extracts feature maps from specific layers of a neural network model.

    Args:
        image (torch.Tensor): The input image tensor of shape [B, C, H, W].
        model (torch.nn.Module): Input model.
        layer_names (list): List of layers to select such as: [layer_1, layer_4, ...].
        layer_mapping (dict): Layer mapping dictionary such as: {"layer_1": "0", "layer_2": "1", ...}.

    Returns:
        dict: Dictionary where keys are feature labels and values are feature maps such as:
            {"0": torch.Tensor, "4": torch.Tensor, ...} where torch.Tensor are of shape [B, C, H, W]
            and keys are the model layers indexes.
    """
    features = {}
    x = image

    # map layer names
    idxs_layers = map_layer(layer_names, layer_mapping)

    # iterate over model layers
    for idx_layer, layer in model._modules.items():
        # apply layer transform
        x = layer(x)
        # save current feature maps if current layer is a layer to select
        if idx_layer in idxs_layers:
            features[idx_layer] = x

    return features


def get_content_loss(target_features, content_features):
    """Compute the content loss between target and original content features.

    Args:
        target_features (torch.Tensor): Features extracted from the target image.
        content_features (torch.Tensor): Features extracted from the content image.

    Returns:
        torch.Tensor: Scalar tensor representing the content loss.
    """
    squared_diff = (target_features - content_features) ** 2
    return squared_diff.mean() / 2


def get_style_loss(style_gram, target_gram, layer_weight):
    """Compute the style loss between target and style gram matrices.

    Args:
        style_gram (torch.Tensor): Gram matrix of the style image.
        target_gram (torch.Tensor): Gram matrix of the target image.
        layer_weights (float): Weight assigned to this particular layer's
            contribution to style loss.

    Returns:
        torch.Tensor: Scalar tensor representing the style loss for one layer.
    """
    # style normalization coefficient is changed here as Adam is used
    # instead of LBFGS in original implementation
    # https://medium.com/analytics-vidhya/styletransfer-3a74c2cb1202
    squared_diff = (target_gram - style_gram) ** 2
    weighted_loss = layer_weight * squared_diff.mean()
    return weighted_loss / (4.0 * (3**2) * (512**2))


def style_transfer(
    epochs,
    path_run,
    current_images,
    current_sizes,
    model,
    layer_mapping,
    content_layers,
    style_layers,
    target_layers,
    style_weights,
    normalization,
    learning_rate,
    alpha,
    beta,
):
    """Perform the style transfer on content image and style image.

    Args:
        epochs (int): Number of training epochs.
        path_run(pathlib.Path): Run directory path.
        current_images (dict): Dictionary of images (content, style and target).
        current_sizes (dict): Dictionary of sizes (content, style and target).
        model (torch.nn.Module): Input model.
        layer_mapping (dict): Layer mapping dictionary such as: {"layer_1": "0", "layer_2": "1", ...}.
        content_layers (list): List of layers to select.
        style_layers (list): List of layers to select.
        target_layers (list): List of layers to select.
        style_weights (dict): Dictionary mapping layers to style weight values.
        normalization (dict): Normalization coefficients.
        learning_rate (float): Learning rate for target optimization.
        alpha (float): Weight for content loss.
        beta (float): Weight for style loss.
    """
    current_images["target"] = current_images["target"].detach().clone().requires_grad_(True)

    # optimize target image
    optimizer = optim.Adam([current_images["target"]], lr=learning_rate)

    content_features = get_features(
        current_images["content"],
        model,
        content_layers,
        layer_mapping,
    )
    style_features = get_features(
        current_images["style"],
        model,
        style_layers,
        layer_mapping,
    )

    # compute gram matrices for style features
    style_grams = {layer: gram_matrix(style_features[layer]) for layer, _ in style_features.items()}

    for _ in tqdm(range(1, epochs + 1)):
        # extract features from target image at specific target layers
        target_features = get_features(current_images["target"], model, target_layers, layer_mapping)

        # compute loss between target and content features at specific deep layer
        content_loss = get_content_loss(
            target_features[layer_mapping["conv_4_2"]],
            content_features[layer_mapping["conv_4_2"]],
        )

        # clear gradients and initialize style loss
        optimizer.zero_grad()
        style_loss = 0

        # iterate on specific style layers defined in style_weights
        for layer_name in style_weights:
            # extract target features and compute gram matrix of current layer
            target_feature = target_features[layer_mapping[layer_name]]
            target_gram = gram_matrix(target_feature)

            # compute weighted loss between style and target features of current layer
            layer_style_loss = get_style_loss(
                style_grams[layer_mapping[layer_name]],
                target_gram,
                style_weights[layer_name],
            )

            style_loss += layer_style_loss

        # compute weighted (alpha, beta) total loss
        total_loss = alpha * content_loss + beta * style_loss

        # backpropagation
        total_loss.backward()

        # optimization step on target image
        optimizer.step()

        path_target_on_training = Path(path_run) / "target_on_training.png"
        save_image(
            path_target_on_training,
            current_images["target"].data[0].detach().clone().cpu(),
            current_sizes["target"],
            normalization,
        )


def style_transfer_loop(device, params, images, sizes):
    """Run a style transfer experiment with run(s) trying all combinations of hyperparameters.

    Args:
        device (torch.device): Pytorch device (CPU or GPU).
        params (dict): Config dictionary.
        images (dict): Dictionary of input images (content/style) as tensors.
        sizes (dict): Dictionary of original image sizes.
    """
    iterations_count = 1

    # create timestamped experiment directory
    timestamp = get_timestamp(keyword="experiment")
    path_experiment = Path("./results/") / timestamp
    path_experiment.mkdir(mode=0o777, parents=True, exist_ok=True)

    # extract tunable hyperparameter lists
    tunning_parameters = [
        params["epochs"],
        params["alpha"],
        params["beta"],
        params["learning_rate"],
        params["pooling_type"],
    ]

    # compute total number of runs
    combinations_total = prod(len(param_list) for param_list in tunning_parameters)
    iterations_total = len(images["style"]) * combinations_total

    # iterate over each content/style pair
    for i, _ in enumerate(images["style"]):
        # iterate over all combinations of tuning parameters
        for epochs, alpha, beta, learning_rate, pooling_type in product(*tunning_parameters):
            print(f"Config: {iterations_count} / {iterations_total}. Running for {epochs} epochs.")
            iterations_count += 1

            # create timestamped run directory
            timestamp = get_timestamp(keyword="run")
            path_run = Path(path_experiment) / timestamp
            path_run.mkdir(mode=0o777, parents=True, exist_ok=True)

            current_images = {
                "content": images["content"][i].to(device),
                "style": images["style"][i].to(device),
                "target": images["content"][i].clone().to(device),
            }

            current_sizes = {
                "content": sizes["content"][i],
                "style": sizes["style"][i],
                "target": sizes["content"][i],
            }

            # map pooling layers names to select to the indexes in model
            idxs_pooling_layers = map_layer(params["pooling_layers"], params["layer_mapping"])

            model = load_model(params["model_name"], idxs_pooling_layers, pooling_type)
            model.to(device)

            # optimize target image
            style_transfer(
                epochs,
                path_run,
                current_images,
                current_sizes,
                model,
                params["layer_mapping"],
                params["content_layers"],
                params["style_layers"],
                params["target_layers"],
                params["style_weights"],
                params["normalize"],
                learning_rate,
                alpha,
                beta,
            )

            save_run(
                path_run,
                current_images,
                current_sizes,
                params,
                {
                    "epochs": epochs,
                    "alpha": alpha,
                    "beta": beta,
                    "learning_rate": learning_rate,
                    "pooling_type": pooling_type,
                },
                params["normalize"],
            )
