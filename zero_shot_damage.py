from functools import partial
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from main import get_from_config
from src.active_nca import ActiveNCA
from src.data_processing import get_MNIST_data, get_simple_object
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
    pixel_wise_L2_and_CE,
)
from src.utils import get_config
from tqdm import tqdm


def alter_divisible(size_list, N_neo, M_neo):
    new_size_list = []
    for size in size_list:
        if size == 0:
            new_size_list.append(size)
        else:
            divisible = False
            while divisible == False:
                for divisor in range(2, M_neo + 1):
                    if size % divisor == 0 and size / divisor <= N_neo:
                        divisible = True
                if not divisible:
                    size += 1
            new_size_list.append(size)

    return np.array(new_size_list)


### Altering methods


def set_to_zero(pixel_list):
    return 0.0


def flip_values(pixel_list):
    return -pixel_list


def set_to_random(pixel_list):
    return np.random.rand(*pixel_list.shape) * 2 - 1


### Getting indexes methods


def sample_randomly(test_size, n_neo=None, m_neo=None):
    if n_neo is None:
        n_neo = N_neo
    if m_neo is None:
        m_neo = M_neo

    x, y = np.meshgrid(list(range(n_neo)), list(range(m_neo)))
    xy = [x.ravel(), y.ravel()]
    indices = np.array(xy).T

    random_indices = np.random.choice(range(len(indices)), size=test_size, replace=False)

    random_x, random_y = indices[random_indices].T + 1

    return random_x, random_y


def sample_rectangular(test_size, n_neo=None, m_neo=None):
    if test_size == 0:
        return [], []

    if n_neo is None:
        n_neo = N_neo
    if m_neo is None:
        m_neo = M_neo

    random_width = np.random.choice(
        [i for i in range(1, n_neo + 1) if test_size % i == 0 and test_size / i <= m_neo and test_size / i >= 1]
    )
    random_height = test_size // random_width

    y_start = 0 if m_neo - random_width == 0 else np.random.randint(m_neo - random_width)
    x_start = 0 if n_neo - random_height == 0 else np.random.randint(n_neo - random_height)

    random_x, random_y = [], []
    for i in range(random_height):
        for j in range(random_width):
            random_x.append(x_start + i)
            random_y.append(y_start + j)

    random_x = np.array(random_x) + 1
    random_y = np.array(random_y) + 1

    return random_x, random_y


def sample_squarely(test_size, n_neo=None, m_neo=None):
    if test_size == 0:
        return [], []

    # As long as test_size is less than 5, width can be 1
    # As long as test_size is less than 10, width can be 2
    # As long as test_size is less than 15, width can be 3

    # which means width must start at test_size // 5 + 1

    possible_least_columns = (test_size // m_neo) + int(test_size % m_neo != 0)

    random_width = np.random.choice(np.arange(possible_least_columns, m_neo + 1))
    height = int(np.ceil(test_size / random_width))

    y_start = 0 if m_neo - random_width == 0 else np.random.randint(m_neo - random_width)
    x_start = 0 if n_neo - height == 0 else np.random.randint(n_neo - height)

    left_of_test_size = test_size
    random_x, random_y = [], []
    for j in range(random_width):
        for i in range(height):
            if left_of_test_size > 0:
                random_x.append(x_start + i)
                random_y.append(y_start + j)
                left_of_test_size -= 1

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


def get_scores_for_all_subfolders_variable_train_size(
    path, silencing_method_get_indexes, pixel_altering_method, predict_altered_method, test_sizes
):

    test_data, target_data = None, None

    scores_all_subpaths = {}

    for number, sub_folder in enumerate(os.listdir(path)):  # For each subfolder in folder path
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):  # If it is a folder
            # Load the saved network for run "sub_path"
            winner_flat = Logger.load_checkpoint(sub_path)

            # Also load its config
            config = get_config(sub_path)
            if config.scale.train_n_neo * config.scale.train_m_neo < len(test_sizes) - 1:
                print(config.scale.train_n_neo * config.scale.train_m_neo)
                continue
            moving_nca_kwargs, loss_function, predicting_method, data_func, kwargs = get_from_config(config)

            # Get the data to use for all the tests on this network
            if test_data is None:
                kwargs["SAMPLES_PER_CLASS"] = NUM_DATA
                kwargs["test"] = True
                test_data, target_data = data_func(**kwargs)
            else:
                print("Data already loaded, continuing")

            # Load network
            network = ActiveNCA.get_instance_with(
                winner_flat, size_neo=(config.scale.train_n_neo, config.scale.train_m_neo), **moving_nca_kwargs
            )

            test_sizes_this = np.round(test_sizes * (config.scale.train_n_neo * config.scale.train_m_neo)).astype(int)

            print("Test sizes", test_sizes_this)

            # Get score for this network for all test sizes

            scores_all_subpaths[sub_path] = get_score_for_damage_sizes(
                network,
                config,
                test_data,
                target_data,
                predicting_method,
                test_sizes_this,
                silencing_method_get_indexes,
                pixel_altering_method,
                predict_altered_method,
                neo_size=(config.scale.train_n_neo, config.scale.train_m_neo),
            )

    return scores_all_subpaths


def get_scores_for_all_subfolders(path, silencing_method_get_indexes, pixel_altering_method):

    test_data, target_data = None, None

    scores_all_subpaths = {}

    for number, sub_folder in enumerate(os.listdir(path)):  # For each subfolder in folder path
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):  # If it is a folder
            # Load the saved network for run "sub_path"
            winner_flat = Logger.load_checkpoint(sub_path)

            # Also load its config
            config = get_config(sub_path)
            moving_nca_kwargs, loss_function, predicting_method, data_func, kwargs = get_from_config(config)

            # Get the data to use for all the tests on this network
            if test_data is None:
                kwargs["SAMPLES_PER_CLASS"] = NUM_DATA
                kwargs["test"] = True
                test_data, target_data = data_func(**kwargs)
            else:
                print("Data already loaded, continuing")

            # Load network
            network = ActiveNCA.get_instance_with(winner_flat, size_neo=(N_neo, M_neo), **moving_nca_kwargs)

            # Get score for this network for all test sizes

            scores_all_subpaths[sub_path] = get_score_for_damage_sizes(
                network,
                config,
                test_data,
                target_data,
                predicting_method,
                test_sizes,
                silencing_method_get_indexes,
                pixel_altering_method,
                neo_size=(N_neo, M_neo),
            )

    return scores_all_subpaths


def get_score_for_damage_sizes(
    network,
    config,
    x_data,
    y_data,
    predicting_method,
    test_sizes,
    silencing_method_get_indexes,
    pixel_altering_method,
    predict_altered_method,
    neo_size,
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
            predict_altered_method,
            neo_size,
        )
        scores.append(score)

    return scores


def get_networks_altered_score(
    test_size,
    network,
    config,
    x_data,
    y_data,
    silencing_method_get_indexes,
    predicting_method,
    pixel_altering_method,
    predict_altered_method,
    neo_size,
):
    batch_size = len(x_data)

    # For each image in the batch, alter a random spot by the current altering method
    score = 0
    for i in range(batch_size):
        # Silence and get performance
        to_alter_indexes = silencing_method_get_indexes(test_size, neo_size[0], neo_size[1])
        class_predictions = predict_altered_method(
            network,
            config,
            to_alter_indexes,
            x_data[i],
            pixel_altering_method,
            visualize=False,  # True if test_size == 473 and i == 0 else False,
        )

        # Record Accuracy
        believed = predicting_method(class_predictions)
        actual = np.argmax(y_data[i])
        score += int(believed == actual)

    return score / batch_size


def predict_altered_silencing(
    network, config, to_alter_indexes, x_data_i, pixel_altering_method, visualize=False, aggregated=False
):
    alter_index_x, alter_index_y = to_alter_indexes

    network.reset()
    for step in range(config.network.iterations):
        class_predictions, _ = network.classify(x_data_i, step=step, visualize=visualize)
        network.state[alter_index_x, alter_index_y, :] = pixel_altering_method(
            network.state[alter_index_x, alter_index_y, :]
        )

    # Set altered values to 0 so that it doesn't mess up prediction
    network.state[alter_index_x, alter_index_y, :] = set_to_zero(network.state[alter_index_x, alter_index_y, :])

    if aggregated:
        class_predictions = network.aggregate()

    return class_predictions


def predict_altered_no_silencing(
    network, config, to_alter_indexes, x_data_i, pixel_altering_method, visualize=False, aggregated=False
):
    alter_index_x, alter_index_y = to_alter_indexes

    network.reset()

    alter_index_x_perc = [(int(x) - 1) for x in alter_index_x]
    alter_index_y_perc = [(int(y) - 1) for y in alter_index_y]

    for step in range(config.network.iterations):
        class_predictions, _ = network.classify(
            x_data_i, step=step, visualize=False, silencing_indexes=[alter_index_x_perc, alter_index_y_perc]
        )

    if aggregated:
        class_predictions = network.aggregate()

    return class_predictions


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


def plot_scores_variable_size(all_scores, title=None):
    cmap = plt.cm.plasma

    _, ax = plt.subplots(1)

    # Plot the scores
    for i, (path, score) in enumerate(all_scores.items()):
        ax.plot(test_sizes, score, color=cmap(i / (len(all_scores) - 1)), label=path)

    ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    ax.set_xticks(test_sizes, np.round(test_sizes * 100).astype(int))
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Randomly silenced cells (%)")
    if title is not None:
        ax.set_title(title)

    plt.legend()

    plt.show()


def plot_scores(all_scores, title=None):
    cmap = plt.cm.plasma

    _, ax = plt.subplots(1)

    # Plot a baseline to show how bad you could possibly do with a random policy
    ax.plot(test_sizes, [0.2 for _ in test_sizes], label="Random accuracy", color="black")

    # Plot the scores
    for i, (path, score) in enumerate(all_scores.items()):
        ax.plot(test_sizes, score, color=cmap(i / (len(all_scores) - 1)), label=path)

    ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    ax.set_xticks(test_sizes, np.round(test_sizes * 100 / (N_neo * M_neo)).astype(int))
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Randomly silenced cells (%)")
    if title is not None:
        ax.set_title(title)

    plt.legend()

    ### ABOSULUTE FIGURE ###

    _, ax = plt.subplots(1)

    for i, (path, score) in enumerate(all_scores.items()):
        plt.plot(
            test_sizes, (np.array(score) - 0.2) / (score[0] - 0.2), color=cmap(i / (len(all_scores) - 1)), label=path
        )

    ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    ax.set_xticks(test_sizes, np.round(test_sizes * 100 / (N_neo * M_neo)).astype(int))
    ax.set_ylabel("Retained accuracy (%)")
    ax.set_xlabel("Randomly silenced cells (%)")
    if title is not None:
        ax.set_title(title)

    plt.legend()

    plt.show()


def show_sampling_effect(sampling_method, pixel_altering_method, test_sizes):
    test_sizes_this = np.round(test_sizes * (26 * 26)).astype(int)

    for size in test_sizes_this:
        for _ in range(5):
            x, y = sampling_method(size, 26, 26)
            img = np.ones((28, 28, 1))
            img[x, y, :] = pixel_altering_method(img[x, y, :])

            plt.imshow(img, cmap="RdBu", vmin=-1, vmax=1)
            plt.show()


if __name__ == "__main__":
    sns.set()

    NUM_DATA = 40
    path = "experiments/mnist3_robust_selective_aggregated_26"
    aggregated = True
    sampling_method = sample_randomly
    pixel_altering_method = set_to_zero
    predict_altered_method = partial(predict_altered_silencing, aggregated=aggregated)
    filename = "/random_silencing_robustness.json"

    test_sizes = np.array(np.linspace(0, 1, 11), dtype=float)

    """show_sampling_effect(sampling_method, pixel_altering_method, test_sizes)
    assert False"""

    all_scores = get_scores_for_all_subfolders_variable_train_size(
        path, sampling_method, pixel_altering_method, predict_altered_method, test_sizes
    )
    plot_scores_variable_size(all_scores, title="Random silencing")

    import json

    all_scores["test_sizes"] = test_sizes.tolist()
    json.dump(all_scores, open(path + filename, "w"))
