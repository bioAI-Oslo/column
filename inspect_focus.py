"""Script to analyze a single network's focus. Do not import from here."""

import argparse
import os
from copy import deepcopy

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from common_funcs import get_network
from localconfig import config
from main import evaluate_nca
from skimage.filters.rank import entropy
from skimage.morphology import cube
from src.plotting_utils import get_plotting_ticks
from tqdm import tqdm

############################### Change here ################################

NUM_DATA = 1  # How many images to classify and visualize

# Network to use, specified by sub_path:
# sub_path = "experiments/color_cifar/19-3-24_10:38"
# sub_path = "experiments/fashion/10-3-24_19:30"
# sub_path = "experiments/fashion/12-3-24_3:14"
# sub_path = "experiments/simple_pattern/21-3-24_13:56"  # Moving
# sub_path = "experiments/simple_pattern/21-3-24_15:31"  # Not moving
# sub_path = "experiments/current_pos_winners/3-3-24_15:58"
# sub_path = "experiments/simple_object_moving/25-3-24_18:56"
# sub_path = "experiments/simple_object_nonmoving/26-3-24_10:27"
# sub_path = "experiments/mnist_final/21-4-24_0:34_2"
# sub_path = "experiments/fashion_mnist/9-4-24_11:16"
sub_path = "experiments/simple_pattern_fixed_loss/8-7-24_21:1_3"
# sub_path = "experiments/fashion_tuning_fixed_loss/6-7-24_0:59"

############################################################################


def get_frequency(network, x_data_i):
    frequencies = np.zeros((*x_data_i.shape[:2], 1))

    # For every field, record position as visited by putting a 1 there
    for row in network.perceptions:
        for x, y in row:
            frequencies[x, y] = 1

    return frequencies


def get_frequency_array(image):
    # p = [[Keys], [Values]] It's just a dumb dictionary
    p = [[], []]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Bin the values of the hidden state
            label = np.round(image[x, y], 1)
            # -0.01 would be rounded to -0.0, causing higher fidelity around 0. There is no reason for there to be, so I set -0 to 0.
            label[label == -0.0] = 0.0
            label = str(label)  # Making the label a string to use as a weird dictionary

            if label in p[0]:
                index = p[0].index(label)
                p[1][index] += 1
            else:
                p[0].append(label)
                p[1].append(1)

    p[1] = np.array(p[1]) / np.sum(p[1])

    return p


def get_entropy(frequency_array):
    H = -np.sum(frequency_array * np.log(frequency_array))
    return H


def get_entropy_of_state(network):
    hidden_channels = network.state[1:-1, 1:-1, : network.num_hidden]

    p = get_frequency_array(hidden_channels)

    H = get_entropy(p[1])

    return H


def get_max_entropy(N, M, O):
    image = np.zeros((N, M, O))

    counter = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                image[x, y, z] = counter
                counter += 1
    p = get_frequency_array(image)
    H = get_entropy(p[1])

    return H


def get_frequencies_and_beliefs(network, x_data_i, original_iterations, iterations):
    frequencies_list = []
    individual_beliefs = []
    entropy_over_time = []
    hidden_states = []

    # Network should always be reset before use on one datapoint to null the internal state
    network.reset()
    for _ in range(int(original_iterations / iterations)):  # At what rate we'll plot, f.ex plot every 10 iterations
        # Get visitation frequency of pixels
        frequencies = get_frequency(network, x_data_i)
        frequencies_list.append(deepcopy(frequencies))  # Don't think I have to deepcopy here, but... it doesn't hurt

        # Collect hidden states
        hidden_states.append(deepcopy(network.state[:, :, : network.num_hidden]))

        # Now run the network while recording the beliefs
        for _ in range(iterations):
            class_predictions, _ = network.classify(x_data_i)

            beliefs = np.zeros((class_predictions.shape[-1]))  # Number of classes
            for x in range(class_predictions.shape[0]):
                for y in range(class_predictions.shape[1]):
                    # Count networks that most believe the class
                    beliefs[np.argmax(class_predictions[x, y])] += 1  # This one for prediction belief
                    """beliefs += (
                        np.exp(class_predictions[x, y]) / tf.reduce_sum(np.exp(class_predictions[x, y])).numpy()
                    )"""  # This one for softmax belief

            individual_beliefs.append(beliefs)

            entropy_over_time.append(get_entropy_of_state(network))

    # Get visitation frequency of pixels
    # Note: These field positions are the 50th timestep, and hasn't been trained for. Consider plotting the 49th instead.
    frequencies = get_frequency(network, x_data_i)
    frequencies_list.append(deepcopy(frequencies))

    # Collect hidden states
    hidden_states.append(deepcopy(network.state[:, :, : network.num_hidden]))

    return frequencies_list, individual_beliefs, entropy_over_time, hidden_states


def plot_frequencies_and_beliefs(
    frequencies_list, individual_beliefs, entropy_over_time, hidden_states, x_data_i, y_data_i, iterations, labels
):
    plt.figure()

    rows = 2  # 3 + len(hidden_states[0][0, 0])

    # We start by plotting the original image with the fields on top for a few selected timesteps

    for i, frequencies in enumerate(frequencies_list):
        # Constructing the appropriate "plt.subplot(121)" string and then turning it to int"
        plt.subplot(int(str(rows) + str(len(frequencies_list)) + str(1 + i)))

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

        # Plotting the pink fields as rectangles
        ax = plt.gca()
        for x in range(frequencies.shape[0]):
            for y in range(frequencies.shape[1]):
                if frequencies[x, y] == 1:
                    # Origin of plt.Rectangle is upper left corner
                    # However, I call first axis the X-axis (it goes down, and is therefore plt's y-axis)
                    # Basically, my way of doing thing is switched from plt.imshow and plt.Rectangle
                    # Therefore, I switch x and y for Rectangle
                    rect = plt.Rectangle((y - 0.5, x - 0.5), 3, 3, fill=False, color="mediumvioletred", linewidth=1)
                    ax.add_patch(rect)
        plt.title("Step: " + str(i * iterations))

    # Now, under, we plot the evolution of system beliefs over time

    sns.set_theme()
    colors = [
        "#2C2463",
        "#DC267F",
        "#EF792A",
        "#D0AE3C",
    ]  # Modified Plasma palette to be more colorfriendly (but idk if I succeeded)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    plt.subplot(int(str(rows) + "12"))
    counter = 0
    for line, label_i in zip(np.array(individual_beliefs).T, labels):
        plt.plot(
            line / (config.scale.test_n_neo * config.scale.test_m_neo),
            label=label_i,
            color=cmap((counter // 2) / max((len(labels) // 2 - 1), 1)),
            linestyle="dashed" if counter % 2 == 0 else "solid",
        )
        counter += 1

    plt.title("Correct class is " + labels[np.argmax(y_data_i)])
    plt.yticks(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10))
    plt.ylabel("System beliefs (%)")
    plt.xlabel("Time steps")
    plt.legend()

    """plt.subplot(int(str(rows) + "13"))
    plt.plot(entropy_over_time, label="Entropy")
    plt.plot(
        [get_max_entropy(hidden_states[0].shape[0], hidden_states[0].shape[1], hidden_states[0].shape[2])]
        * len(entropy_over_time),
        color="gray",
        linestyle="dashed",
        label="Max entropy",
    )

    plt.title("Entropy over time")
    plt.xlabel("Time steps")
    plt.legend()

    maxx = np.max(hidden_states)
    minn = np.min(hidden_states)
    for i, hidden_state in enumerate(hidden_states):
        for j in range(hidden_state.shape[-1]):
            plt.subplot(rows, len(hidden_states), len(hidden_states) * (3 + j) + 1 + i)
            plt.imshow(hidden_state[1:-1, 1:-1, j], vmax=maxx, vmin=minn)"""


def plotting_individual_classifications():
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    test_data, target_data = data_func(**kwargs)

    # By setting iterations to 1, we can exit the loop to manipulate state, and then (by not resetting) continue the loop
    original_iterations = network.iterations
    iterations = 10  # At what rate we'll plot. F.ex. plot every 10 iterations
    network.iterations = 1

    # For each datapoint
    for x_data_i, y_data_i in zip(test_data, target_data):
        # Get frequencies and individual beliefs
        # Frequencies is if the pixel is visited IN THAT timestep or not. So no averaging (this gives a better image than averaging)
        # Individual beliefs is what the system thinks the class is at EVERY timestep. Likewise, no averaging.
        frequencies_list, individual_beliefs, entropy_over_time, hidden_states = get_frequencies_and_beliefs(
            network, x_data_i, original_iterations, iterations
        )
        plot_frequencies_and_beliefs(
            frequencies_list,
            individual_beliefs,
            entropy_over_time,
            hidden_states,
            x_data_i,
            y_data_i,
            iterations,
            labels,
        )

    plt.show()


if __name__ == "__main__":
    plotting_individual_classifications()

    """ents = []
    for size in range(1, 50):
        image = np.zeros((size, size, 3))
        p = get_frequency_array(image)
        H = get_entropy(p[1])

        print("Homogeneous entropy:", H)

        counter = 0
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                for z in range(image.shape[2]):
                    image[x, y, z] = counter
                    counter += 1
        p = get_frequency_array(image)
        H = get_entropy(p[1])
        print(p[1][0])

        ents.append(H)

        print("Noise entropy:", H)

    plt.plot(range(1, 50), ents)
    plt.show()"""
