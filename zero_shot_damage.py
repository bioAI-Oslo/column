import argparse
import os
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

NUM_DATA = 100
N_neo = 26
M_neo = 26
test_sizes = [0, 50, 100, 150, 200, 250, 300, 400, 500, 600, N_neo * M_neo]
# test_sizes = [0, 5, 10, 15, 20, 25, 30, 35]


### Altering methods


def set_to_zero(pixel_list):
    return 0.0


def flip_values(pixel_list):
    return -pixel_list


def set_to_random(pixel_list):
    return np.random.rand(*pixel_list.shape) * 2 - 1


### Getting indexes methods


def sample_randomly(test_size):
    x, y = np.meshgrid(list(range(N_neo)), list(range(M_neo)))
    xy = [x.ravel(), y.ravel()]
    indices = np.array(xy).T

    random_indices = np.random.choice(range(len(indices)), size=test_size, replace=False)

    random_x, random_y = indices[random_indices].T + 1

    return random_x, random_y


def sample_rectangular(test_size):
    if test_size == 0:
        return [], []
    random_width = np.random.choice(
        [i for i in range(1, N_neo + 1) if test_size % i == 0 and test_size / i <= 26 and test_size / i >= 1]
    )
    random_height = test_size // random_width

    y_start = 0 if M_neo - random_width == 0 else np.random.randint(M_neo - random_width)
    x_start = 0 if N_neo - random_height == 0 else np.random.randint(N_neo - random_height)

    random_x, random_y = [], []
    for i in range(random_height):
        for j in range(random_width):
            random_x.append(x_start + i)
            random_y.append(y_start + j)

    random_x = np.array(random_x) + 1
    random_y = np.array(random_y) + 1

    return random_x, random_y


def sample_circular(test_size):
    x = np.random.randint(N_neo)
    y = np.random.randint(M_neo)

    radius = test_size
    random_x, random_y = [], []
    for i in range(N_neo):
        for j in range(M_neo):
            if np.sqrt((i - x) ** 2 + (j - y) ** 2) < radius:
                random_x.append(i)
                random_y.append(j)

    random_x = np.array(random_x, dtype=int) + 1
    random_y = np.array(random_y, dtype=int) + 1

    return random_x, random_y


### Getting scores methods


def get_scores_for_all_subfolders(path, silencing_method_get_indexes, pixel_altering_method):

    test_data, target_data = None, None

    scores_all_subpaths = []

    for number, sub_folder in enumerate(os.listdir(path)):  # For each subfolder in folder path
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):  # If it is a folder
            # Load the saved network for run "sub_path"
            winner_flat = Logger.load_checkpoint(sub_path)

            # Also load its config
            config = get_config(sub_path)

            # Fetch info from config and enable environment for testing
            mnist_digits = eval(config.dataset.mnist_digits)

            moving_nca_kwargs = {
                "size_image": (28, 28),
                "num_classes": len(mnist_digits),
                "num_hidden": config.network.hidden_channels,
                "hidden_neurons": config.network.hidden_neurons,
                "iterations": config.network.iterations,
                "position": str(config.network.position),
                "moving": config.network.moving,
                "mnist_digits": mnist_digits,
            }

            predicting_method = eval(config.training.predicting_method)

            # Get the data to use for all the tests on this network
            if test_data is None:
                print("Fetching data")
                data_func = get_MNIST_data
                kwargs = {
                    "MNIST_DIGITS": mnist_digits,
                    "SAMPLES_PER_DIGIT": NUM_DATA,
                    "verbose": False,
                    "test": True,
                }
                test_data, target_data = data_func(**kwargs)
                test_data = test_data.reshape(*test_data.shape, 1)
            else:
                print("Data already loaded, continuing")

            # Load network
            network = MovingNCA.get_instance_with(
                winner_flat, size_neo=(config.scale.test_n_neo, config.scale.test_m_neo), **moving_nca_kwargs
            )

            # Get score for this network for all test sizes
            scores_all_subpaths.append(
                get_score_for_damage_sizes(
                    network,
                    config,
                    test_data,
                    target_data,
                    predicting_method,
                    test_sizes,
                    silencing_method_get_indexes,
                    pixel_altering_method,
                )
            )

    return scores_all_subpaths


def get_score_for_damage_sizes(
    network, config, x_data, y_data, predicting_method, test_sizes, silencing_method_get_indexes, pixel_altering_method
):
    scores = []
    for test_size in tqdm(test_sizes):
        # Get network's altered score under test size
        score = get_networks_altered_score(
            test_size,
            network,
            config,
            x_data,
            y_data,
            silencing_method_get_indexes,
            predicting_method,
            pixel_altering_method,
        )
        scores.append(score)

    return scores


def get_networks_altered_score(
    test_size, network, config, x_data, y_data, silencing_method_get_indexes, predicting_method, pixel_altering_method
):
    batch_size = len(x_data)

    # By setting iterations to 1, we can exit the loop to manipulate state, and then (by not resetting) continue the loop
    network.iterations = 1

    # For each image in the batch, alter a random spot by the current altering method
    score = 0
    for i in range(batch_size):
        # Silence and get performance
        to_alter_indexes = silencing_method_get_indexes(test_size)
        class_predictions = predict_altered(
            network,
            config,
            to_alter_indexes,
            x_data[i],
            pixel_altering_method,
            visualize=True if test_size == 200 and i == 0 else False,
        )

        # Record Accuracy
        believed = predicting_method(class_predictions)
        actual = np.argmax(y_data[i])
        score += int(believed == actual)

    return score / batch_size


def predict_altered(network, config, to_alter_indexes, x_data_i, pixel_altering_method, visualize=False):
    alter_index_x, alter_index_y = to_alter_indexes

    states = []  # Just for plotting

    network.reset()
    for _ in range(config.network.iterations):
        class_predictions, _ = network.classify(x_data_i)
        network.state[alter_index_x, alter_index_y, :] = pixel_altering_method(
            network.state[alter_index_x, alter_index_y, :]
        )

        states.append(deepcopy(network.state))  # Just for plotting

    states = np.array(states)  # Just for plotting
    B, _, _, O = states.shape

    # Plot the states
    """if visualize:
        for b in range(B):
            plt.figure()
            for o in range(O):
                plt.subplot(1, O, o + 1)
                plt.imshow(states[b, :, :, o], cmap="RdBu", vmin=-1, vmax=1)
        plt.show()"""

    # Set altered values to 0 so that it doesn't mess up prediction
    network.state[alter_index_x, alter_index_y, :] = set_to_zero(network.state[alter_index_x, alter_index_y, :])

    return class_predictions


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


def plot_scores(all_scores, title=None):
    cmap = plt.cm.plasma

    _, ax = plt.subplots(1)

    # Plot a baseline to show how bad you could possibly do with a random policy
    ax.plot(test_sizes, [0.2 for _ in test_sizes], label="Random accuracy", color="black")

    # Plot the scores
    ax.plot(test_sizes, np.mean(all_scores, 0), label="Mean", color=cmap(0.2))
    ax.fill_between(
        test_sizes,
        np.mean(all_scores, axis=0) - np.std(all_scores, axis=0),
        np.mean(all_scores, axis=0) + np.std(all_scores, axis=0),
        color=cmap(0.2),
        alpha=0.5,
    )

    # Plot best network's accuracy
    best_network = np.argmax(np.max(all_scores, axis=1))  # Get best network
    ax.plot(test_sizes, all_scores[best_network], label="Best network's accuracy", color=cmap(0.5))

    ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    ax.set_xticks(test_sizes, test_sizes)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Nr. randomly silenced cells")
    if title is not None:
        ax.set_title(title)
    plt.show()


def show_sampling_effect(sampling_method, pixel_altering_method):
    for size in test_sizes:
        for _ in range(5):
            x, y = sampling_method(size)
            img = np.ones((28, 28, 1))
            img[x, y, :] = pixel_altering_method(img[x, y, :])

            plt.imshow(img, cmap="RdBu", vmin=-1, vmax=1)
            plt.show()


if __name__ == "__main__":
    path = "experiments/current_pos_winners"
    sampling_method = sample_rectangular
    pixel_altering_method = set_to_random

    show_sampling_effect(sampling_method, pixel_altering_method)

    all_scores = get_scores_for_all_subfolders(path, sampling_method, pixel_altering_method)
    plot_scores(all_scores, title="Rectangular silencing")
