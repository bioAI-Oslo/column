import argparse
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from main import evaluate_nca
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
    pixel_wise_L2_and_CE,
    scale_loss,
)
from src.mnist_processing import get_MNIST_data
from src.moving_nca import MovingNCA
from src.utils import get_config
from tqdm import tqdm

sns.set()


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="Main", description="This program runs an optimization.", epilog="Text at the bottom of help"
    )
    parser.add_argument("-c", "--config", type=str, help="The config file to use", default="config")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="The path to the run to analyze",
        default=None,
    )

    args = parser.parse_args()
    return args


def plot_average_out_circular():
    N_neo, M_neo = 26, 26

    def get_num_out(radius):
        out_num = 0
        iter = 100
        for _ in range(iter):
            out_this_time = 0
            x = np.random.randint(N_neo)
            y = np.random.randint(M_neo)
            for i in range(N_neo):
                for j in range(M_neo):
                    if np.sqrt((i - x) ** 2 + (j - y) ** 2) < radius:
                        out_this_time += 1
            out_num += out_this_time
        return out_num / iter

    out_avg = []
    for radius in range(int(np.sqrt(26**2 + 26**2)) + 1):
        out_avg.append(get_num_out(radius))

    plt.plot(out_avg)
    plt.xlabel("Radius")
    plt.ylabel("Average number of silenced cells")
    plt.show()


if __name__ == "__main__":
    # Moved to its own function because of size.
    args = parse_args()

    # Read config
    config.read(args.config)
    mnist_digits = eval(config.dataset.mnist_digits)
    loss_function = eval(config.training.loss)
    predicting_method = eval(config.training.predicting_method)

    # This parameter dictionary will be used for all instances of the network
    moving_nca_kwargs = {
        "size_image": (28, 28),  # Changed below if we need to
        "num_classes": len(mnist_digits),
        "num_hidden": config.network.hidden_channels,
        "hidden_neurons": config.network.hidden_neurons,
        "iterations": config.network.iterations,
        "position": config.network.position,
        "moving": config.network.moving,
        "mnist_digits": mnist_digits,
    }

    # Data function and kwargs
    data_func = eval(config.dataset.data_func)
    kwargs = {
        "MNIST_DIGITS": mnist_digits,
        "SAMPLES_PER_DIGIT": 100,
        "verbose": False,
        "test": True,
    }
    # Taking specific care with the data functions
    if config.dataset.data_func == "get_MNIST_data_resized":
        kwargs["size"] = config.dataset.size
        moving_nca_kwargs["size_image"] = (config.dataset.size, config.dataset.size)
    elif config.dataset.data_func == "get_MNIST_data_translated":
        # Size of translated data "get_MNIST_data_translated" is 70x70, specified in the function
        moving_nca_kwargs["size_image"] = (70, 70)
    elif config.dataset.data_func == "get_MNIST_data_padded":
        # Size of translated data "get_MNIST_data_translated" is 70x70, specified in the function
        moving_nca_kwargs["size_image"] = (56, 56)

    winner_flat = Logger.load_checkpoint(args.path)

    # Get test data for new evaluation
    training_data, target_data = data_func(**kwargs)

    silenced_test_sizes = list(range(0, int(np.sqrt(26**2 + 26**2)) + 1, 1))
    print(silenced_test_sizes)

    scores = []
    for silenced in tqdm(silenced_test_sizes):
        loss, acc = evaluate_nca(
            winner_flat,
            training_data,
            target_data,
            moving_nca_kwargs,
            loss_function,
            predicting_method,
            verbose=False,
            visualize=False,
            return_accuracy=True,
            N_neo=config.scale.test_n_neo,
            M_neo=config.scale.test_m_neo,
            return_confusion=True,
            silenced=silenced,
        )
        print("Score", loss, "acc:", acc)
        scores.append(acc)

    plt.plot(silenced_test_sizes, scores)
    plt.vlines(13, linestyles="dashed", label="ca 340/676 cells out on avg", ymin=0, ymax=1)
    plt.vlines(20, linestyles="dashed", label="ca 550/676 cells out on avg", ymin=0, ymax=1)
    plt.vlines(30, linestyles="dashed", label="ca all cells out on avg", ymin=0, ymax=1)
    plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, 120, 20))
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Radius of randomly placed circle")
    plt.legend()
    plt.show()
