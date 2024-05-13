import argparse
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from localconfig import config
from main import evaluate_nca
from src.data_processing import (
    get_CIFAR_data,
    get_labels,
    get_MNIST_data,
    get_MNIST_fashion_data,
    get_simple_object,
    get_simple_pattern,
)
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
from src.moving_nca import MovingNCA
from src.plotting_utils import get_plotting_ticks
from src.utils import get_config
from tqdm import tqdm

NUM_DATA = 1
# sub_path = "experiments/color_cifar/19-3-24_10:38"
# sub_path = "experiments/fashion/10-3-24_19:30"
# sub_path = "experiments/fashion/12-3-24_3:14"
# sub_path = "experiments/simple_pattern/21-3-24_13:56"  # Moving
# sub_path = "experiments/simple_pattern/21-3-24_15:31"  # Not moving
# sub_path = "experiments/current_pos_winners/3-3-24_15:58"
# sub_path = "experiments/simple_object_moving/25-3-24_18:56"
# sub_path = "experiments/simple_object_nonmoving/26-3-24_10:27"
sub_path = "experiments/mnist_final/21-4-24_0:34_2"
# sub_path = "experiments/fashion_mnist/9-4-24_11:16"


def get_network(sub_path):
    winner_flat = Logger.load_checkpoint(sub_path)

    # Also load its config
    config = get_config(sub_path)

    # Fetch info from config and enable environment for testing
    mnist_digits = eval(config.dataset.mnist_digits)

    predicting_method = eval(config.training.predicting_method)

    moving_nca_kwargs = {
        "size_image": (config.dataset.size, config.dataset.size),
        "num_hidden": config.network.hidden_channels,
        "hidden_neurons": config.network.hidden_neurons,
        "iterations": config.network.iterations,
        "position": str(config.network.position),
        "moving": config.network.moving,
        "mnist_digits": mnist_digits,
        "img_channels": config.network.img_channels,
    }

    data_func = eval(config.dataset.data_func)
    kwargs = {
        "CLASSES": mnist_digits,
        "SAMPLES_PER_CLASS": NUM_DATA,
        "verbose": False,
        "test": True,
        "colors": True if config.network.img_channels == 3 else False,
    }

    labels = get_labels(data_func, kwargs["CLASSES"])

    # Load network
    network = MovingNCA.get_instance_with(
        winner_flat, size_neo=(config.scale.test_n_neo, config.scale.test_m_neo), **moving_nca_kwargs
    )

    return network, labels, data_func, kwargs, predicting_method


def get_frequency(network, x_data_i):
    frequencies = np.zeros((*x_data_i.shape[:2], 1))

    for row in network.perceptions:
        for x, y in row:
            frequencies[x, y] = 1

    return frequencies


def get_frequencies_and_beliefs(network, x_data_i, original_iterations, iterations):
    frequencies_list = []
    individual_beliefs = []

    network.reset()
    for _ in range(int(original_iterations / iterations)):
        # Get visitation frequency of pixels
        frequencies = get_frequency(network, x_data_i)
        frequencies_list.append(deepcopy(frequencies))

        for _ in range(iterations):
            class_predictions, _ = network.classify(x_data_i)

            beliefs = np.zeros((class_predictions.shape[-1]))
            for x in range(class_predictions.shape[0]):
                for y in range(class_predictions.shape[1]):
                    beliefs[np.argmax(class_predictions[x, y])] += 1  # This one for prediction belief
                    """beliefs += (
                        np.exp(class_predictions[x, y]) / tf.reduce_sum(np.exp(class_predictions[x, y])).numpy()
                    )"""  # This one for softmax belief

            individual_beliefs.append(beliefs)

    # Get visitation frequency of pixels
    frequencies = get_frequency(network, x_data_i)
    frequencies_list.append(deepcopy(frequencies))

    return frequencies_list, individual_beliefs


def plot_frequencies_and_beliefs(frequencies_list, individual_beliefs, x_data_i, y_data_i, iterations, labels):
    plt.figure()
    for i, frequencies in enumerate(frequencies_list):
        plt.subplot(int("2" + str(len(frequencies_list)) + str(1 + i)))
        # plt.imshow(frequencies * x_data_i)
        if x_data_i.shape[-1] == 3:
            plt.imshow(x_data_i)
        else:
            plt.imshow(x_data_i, cmap="gray")

        if i == 0:
            xticks, yticks = get_plotting_ticks(x_data_i)
            plt.xticks(xticks[0], xticks[1])
            plt.yticks(yticks[0], yticks[1])
        else:
            plt.xticks([])
            plt.yticks([])

        ax = plt.gca()
        for x in range(frequencies.shape[0]):
            for y in range(frequencies.shape[1]):
                if frequencies[x, y] == 1:
                    rect = plt.Rectangle((y - 0.5, x - 0.5), 3, 3, fill=False, color="mediumvioletred", linewidth=1)
                    ax.add_patch(rect)
        plt.title("Step: " + str(i * iterations))

    plt.subplot(212)
    for line, label_i in zip(np.array(individual_beliefs).T, labels):
        plt.plot(line / (config.scale.test_n_neo * config.scale.test_m_neo), label=label_i)
    plt.title("Correct class is " + labels[np.argmax(y_data_i)])
    plt.title("Correct class is " + labels[np.argmax(y_data_i)])
    plt.yticks(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10))
    plt.ylabel("System beliefs (%)")
    plt.xlabel("Time steps")
    plt.legend()


def plotting_individual_classifications():
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path)

    test_data, target_data = data_func(**kwargs)

    # By setting iterations to 1, we can exit the loop to manipulate state, and then (by not resetting) continue the loop
    original_iterations = network.iterations
    iterations = 10
    network.iterations = 1

    for x_data_i, y_data_i in zip(test_data, target_data):
        frequencies_list, individual_beliefs = get_frequencies_and_beliefs(
            network, x_data_i, original_iterations, iterations
        )
        plot_frequencies_and_beliefs(frequencies_list, individual_beliefs, x_data_i, y_data_i, iterations, labels)

    plt.show()


if __name__ == "__main__":
    plotting_individual_classifications()
