from copy import deepcopy

import numpy as np
from localconfig import config
from matplotlib import pyplot as plt
from src.data_processing import get_MNIST_data
from src.logger import Logger
from src.moving_nca import MovingNCA


def show_weights(network):
    input_to_hidden_weights = np.array(network.weights[0]).T
    input_to_hidden_bias = network.weights[1]
    hidden_to_out_weights = network.weights[2]
    hidden_to_out_bias = network.weights[3]

    for j, hidden_node in enumerate(input_to_hidden_weights):
        hidden_node = hidden_node[:-2]
        reshapen = np.reshape(hidden_node, (3, 3, 8))

        for i in range(8):
            plt.subplot(10, 8, j * 8 + i + 1)
            plt.imshow(reshapen[:, :, i])

    plt.show()


def show_diff_map(network, config, mnist_digits):
    x_data, y_data = get_MNIST_data(CLASSES=mnist_digits, SAMPLES_PER_CLASS=100, verbose=False, test=True)
    x_data = np.reshape(x_data, (len(x_data), 28, 28, 1))

    network.reset()
    nr_channels = network.state.shape[-1]
    network.iterations = 2
    for j in range(10):
        prev_state = deepcopy(network.state)
        class_predictions, _ = network.classify(x_data[0])
        new_state = network.state
        diff_state = new_state - prev_state

        for i in range(nr_channels - len(mnist_digits)):
            plt.subplot(10, (nr_channels - len(mnist_digits) + 1), j * (nr_channels - len(mnist_digits) + 1) + i + 1)
            plt.imshow(diff_state[:, :, i])  # , vmax=np.max(diff_state), vmin=np.min(diff_state))

        plt.subplot(10, (nr_channels - len(mnist_digits) + 1), j * (nr_channels - len(mnist_digits) + 1) + i + 2)
        plt.imshow(diff_state[:, :, -3:] / np.max(diff_state[:, :, -3:]))
    plt.show()


if __name__ == "__main__":
    path = "experiments/tuning_iter_and_sigma/29-1-24_11:41"
    # path = "experiments/tuning_size/3-2-24_17:28"
    winner_flat = Logger.load_checkpoint(path)

    config.read(path + "/config")
    mnist_digits = eval(config.dataset.mnist_digits)

    N_neo, M_neo = 26, 26
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

    network = MovingNCA.get_instance_with(winner_flat, size_neo=(N_neo, M_neo), **moving_nca_kwargs)

    show_diff_map(network, config, mnist_digits)
