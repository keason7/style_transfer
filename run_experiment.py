"""Main script to run a style transfer experiment."""

import argparse
from pprint import pprint

from src.config import Config
from src.transfer import get_device, style_transfer_loop
from src.transforms import prepare_images
from src.utils import get_paths


def run_experiment(path_config):
    """Run a style transfer experiment.

    Args:
        path_config (str): Input config path.
    """
    config = Config(path_config)
    pprint(config.params)

    device = get_device()

    # get content and style images paths
    paths = get_paths(config.params["path_dataset"], config.params["dataset_experiments"])

    # prepare data for style transfer
    images, sizes = prepare_images(paths, config.params)

    # launch experiments
    style_transfer_loop(device, config.params, images, sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a style transfer experiment.")
    parser.add_argument("-pc", "--path_config", type=str, default="./config/root.yml", help="Config path.")
    args = parser.parse_args()

    run_experiment(args.path_config)
