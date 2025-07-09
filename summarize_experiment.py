"""Main script to summarize a style transfer experiment within a single folder."""

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image


def summarize_experiment(path_experiment):
    """Summarize a style transfer experiment within a single folder.

    Args:
        path_experiment (str): Experiment path.
    """
    path_experiment = Path(path_experiment)

    # get all directory paths within experiment directory
    runs = [path for path in path_experiment.iterdir() if path.is_dir()]

    # filter directories that do not start by "run"
    runs = [path for path in runs if path.name.startswith("run")]

    # sort runs by timestamp
    runs = sorted(runs, key=lambda p: datetime.strptime(p.name.replace("run_", ""), "%Y_%m_%d-%H_%M_%S"))

    # create summary directory within experiment directory
    path_summary = path_experiment / "summary"
    path_summary.mkdir(mode=0o777, parents=True, exist_ok=True)

    for i, path_run in enumerate(runs):
        # open run target image
        image_target = Image.open(path_run / "target.png")

        # save it in summary directory
        image_target.save(path_summary / f"{i+1}_{path_run.name}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy target image(s) of experiment run(s) in a summary folder.")
    parser.add_argument("-pe", "--path_experiment", type=str, required=True, help="Experiment directory path.")
    args = parser.parse_args()

    summarize_experiment(args.path_experiment)
