"""Script to analyze a single network's focus. Do not import from here."""

import random
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common_funcs import get_network
from localconfig import config
from skimage import color
from src.plotting_utils import get_plotting_ticks

############################### Change here ################################

NUM_DATA = 1  # How many images to classify and visualize

# Network to use, specified by sub_path:
# sub_path = "experiments/simple_object_moving/18-7-24_12:4_10" # Random network for simple object
# sub_path = "experiments/simple_pattern/16-7-24_15:33_2" # Random network for simple pattern
# sub_path = "experiments/cifar4_color_tuning3/20-9-24_14:53" # Cool cifar network
# sub_path = "experiments/fashion_final/17-9-24_0:28"  # Best fashion 10 class network
# sub_path = "experiments/fashion_final/17-9-24_15:21"  # Another good fashion 10 class network
# sub_path = "experiments/cifar4_color_tuning4/1-10-24_11:23"

# MNIST 10
sub_path = "experiments/mnist10_final/11-9-24_9:45"  # Best Mnist 10 class network
# sub_path = "experiments/mnist10_final/10-9-24_9:13_2"
# sub_path = "experiments/mnist10_final/6-9-24_10:29"
# sub_path = "experiments/mnist10_final/5-9-24_12:5"
# sub_path = "experiments/mnist10_final/9-9-24_11:22"
# sub_path = "experiments/mnist10_final/10-9-24_9:13"
# sub_path = "experiments/mnist10_final/2-9-24_15:10"

############################################################################


def get_frequency(network, x_data_i, count=False, area=False):
    frequencies = np.zeros((*x_data_i.shape[:2], 1))

    # For every field, record position as visited by putting a 1 there
    for row in network.perceptions:
        for x, y in row:
            for i in range(0, 1 + 2 * int(area)):
                for j in range(0, 1 + 2 * int(area)):
                    if count:
                        frequencies[x + i, y + j] += 1
                    else:
                        frequencies[x + i, y + j] = 1

    return frequencies


def get_movement(network, x_data_i, movement_iterations=1):
    # Network should always be reset before use on one datapoint to null the internal state
    network.reset()

    positions = np.zeros((network.iterations // movement_iterations, *network.perceptions.shape))
    movements = np.zeros((network.iterations // movement_iterations, *network.perceptions.shape))

    for step in range(network.iterations // movement_iterations):
        prev_pos = deepcopy(network.perceptions)

        positions[step, :, :, :] = prev_pos

        for int_step in range(movement_iterations):
            _, _ = network.classify(x_data_i, step=step * movement_iterations + int_step)

        movements[step, :, :, :] = deepcopy(network.perceptions) - prev_pos

    return movements, positions


def get_frequencies_and_beliefs(network, x_data_i, frequency_iterations=10, count=False, area=False):
    frequencies_list = []
    individual_beliefs = []

    # Network should always be reset before use on one datapoint to null the internal state
    network.reset()
    for step in range(network.iterations // frequency_iterations):
        # At what rate we'll plot, f.ex plot every 10 iterations

        # Get visitation frequency of pixels
        frequencies = get_frequency(network, x_data_i, count=count, area=area)
        frequencies_list.append(deepcopy(frequencies))  # Don't think I have to deepcopy here, but... it doesn't hurt

        # Now run the network while recording the beliefs
        for int_step in range(frequency_iterations):
            class_predictions, _ = network.classify(x_data_i, step=step * frequency_iterations + int_step)

            beliefs = np.zeros((class_predictions.shape[-1]))  # Number of classes
            for x in range(class_predictions.shape[0]):
                for y in range(class_predictions.shape[1]):
                    # Count networks that most believe the class
                    beliefs[np.argmax(class_predictions[x, y])] += 1  # This one for prediction belief
                    """beliefs += (
                        np.exp(class_predictions[x, y]) / tf.reduce_sum(np.exp(class_predictions[x, y])).numpy()
                    )"""  # This one for softmax belief

            individual_beliefs.append(beliefs)

    # Get visitation frequency of pixels
    # Note: These field positions are the 50th timestep, and hasn't been trained for. Consider plotting the 49th instead.
    frequencies = get_frequency(network, x_data_i, count=count, area=area)
    frequencies_list.append(deepcopy(frequencies))

    return frequencies_list, individual_beliefs


def plot_frequencies_and_beliefs(frequencies_list, individual_beliefs, x_data_i, y_data_i, iterations, labels):
    plt.figure()

    rows = 2

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
        color = cmap((counter // 2) / max((len(labels) // 2 - 1), 1) / (len(labels) // 2))
        style = "dashed" if counter % 2 == 0 else "solid"
        if len(labels) == 3:
            color = cmap((counter) / max((len(labels)), 1))
            style = "solid"
        plt.plot(
            line / (config.scale.test_n_neo * config.scale.test_m_neo),
            label=label_i,
            color=color,
            linestyle=style,
        )
        counter += 1

    plt.title("Correct class is " + labels[np.argmax(y_data_i)])
    plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, 120, 20))
    plt.ylabel("System beliefs (%)")
    plt.xlabel("Time steps")
    plt.legend()


def plot_individual_classifications():
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    # For each datapoint
    for x_data_i, y_data_i in zip(test_data, target_data):
        # Get frequencies and individual beliefs
        # Frequencies is if the pixel is visited IN THAT timestep or not. So no averaging (this gives a better image than averaging)
        # Individual beliefs is what the system thinks the class is at EVERY timestep. Likewise, no averaging.
        frequencies_list, individual_beliefs = get_frequencies_and_beliefs(network, x_data_i, frequency_iterations=10)
        plot_frequencies_and_beliefs(
            frequencies_list,
            individual_beliefs,
            x_data_i,
            y_data_i,
            network.iterations,
            labels,
        )

    plt.show()


def plot_individual_classifications_beliefs():
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    # For each datapoint
    for x_data_i, y_data_i in zip(test_data, target_data):
        # Get frequencies and individual beliefs
        # Frequencies is if the pixel is visited IN THAT timestep or not. So no averaging (this gives a better image than averaging)
        # Individual beliefs is what the system thinks the class is at EVERY timestep. Likewise, no averaging.
        frequencies_list, individual_beliefs = get_frequencies_and_beliefs(network, x_data_i, frequency_iterations=10)

        sns.set_theme()
        colors = [
            "#2C2463",
            "#DC267F",
            "#EF792A",
            "#D0AE3C",
        ]  # Modified Plasma palette to be more colorfriendly (but idk if I succeeded)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
        plt.figure()
        counter = 0
        for line, label_i in zip(np.array(individual_beliefs).T, labels):
            color = cmap((counter // 2) / max((len(labels) // 2 - 1), 1) / (len(labels) // 2))
            style = "dashed" if counter % 2 == 0 else "solid"
            if len(labels) == 3:
                color = cmap((counter) / max((len(labels)), 1))
                style = "solid"
            plt.plot(
                line / (config.scale.test_n_neo * config.scale.test_m_neo),
                label=label_i,
                color=color,
                linestyle=style,
            )
            counter += 1

        plt.title("Correct class is " + labels[np.argmax(y_data_i)])
        plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, 120, 20))
        plt.ylabel("System beliefs (%)")
        plt.xlabel("Time steps")
        plt.legend()

    plt.show()


def plot_heatmap():
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    # For each datapoint
    for x_data_i, y_data_i in zip(test_data, target_data):
        # Get frequencies and individual beliefs
        # Frequencies is if the pixel is visited IN THAT timestep or not. So no averaging (this gives a better image than averaging)
        # Individual beliefs is what the system thinks the class is at EVERY timestep. Likewise, no averaging.
        frequencies_list, individual_beliefs = get_frequencies_and_beliefs(
            network, x_data_i, frequency_iterations=1, count=False, area=True
        )

        fig = plt.figure()
        fig.suptitle("Believed class is " + labels[np.argmax(individual_beliefs[-1])])

        plt.subplot(131)
        plt.imshow(x_data_i, cmap="gray")
        plt.title("Input image")

        heatmap = np.mean(frequencies_list, axis=(0, -1))

        plt.subplot(132)
        plt.imshow(heatmap)
        plt.title("Field position heatmap")

        if x_data_i.shape[2] <= 1:
            plt.subplot(133)
            image_plus_heatmap = np.zeros((x_data_i.shape[0], x_data_i.shape[1], 3))
            image_plus_heatmap[:, :, 0] = (heatmap) / np.max(heatmap)
            image_plus_heatmap[:, :, 1] = x_data_i[:, :, 0] * 0.7
            image_plus_heatmap[:, :, 2] = (heatmap) / np.max(heatmap)
            plt.imshow(image_plus_heatmap)
            plt.title("image + heatmap")

        else:
            plt.subplot(133)
            image_plus_heatmap = np.zeros((x_data_i.shape[0], x_data_i.shape[1], 3))
            image_plus_heatmap[:, :, 0] = (heatmap) / np.max(heatmap)
            image_plus_heatmap[:, :, 1] = color.rgb2gray(x_data_i)
            image_plus_heatmap[:, :, 2] = (heatmap) / np.max(heatmap)
            plt.imshow(image_plus_heatmap)
            plt.title("image + heatmap")

        plt.show()


def plot_path_map():
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    # For each datapoint
    for x_data_i, y_data_i in zip(test_data, target_data):
        # Get frequencies and individual beliefs
        # Frequencies is if the pixel is visited IN THAT timestep or not. So no averaging (this gives a better image than averaging)
        # Individual beliefs is what the system thinks the class is at EVERY timestep. Likewise, no averaging.
        frequencies_list, individual_beliefs = get_frequencies_and_beliefs(
            network, x_data_i, frequency_iterations=1, count=False, area=True
        )

        movement_iterations = 2
        partitions = 5

        movements, positions = get_movement(network, x_data_i, movement_iterations=movement_iterations)

        plt.figure()

        for i in range(len(movements) // partitions):
            all_positions = {}

            for movement, position in zip(
                movements[i * partitions : (i + 1) * partitions], positions[i * partitions : (i + 1) * partitions]
            ):

                for x in range(movement.shape[0]):
                    for y in range(movement.shape[1]):
                        if (position[x, y, 0], position[x, y, 1]) not in all_positions:
                            all_positions[(position[x, y, 0], position[x, y, 1])] = [
                                [movement[x, y, 0], movement[x, y, 1]]
                            ]
                        else:
                            all_positions[(position[x, y, 0], position[x, y, 1])].append(
                                [movement[x, y, 0], movement[x, y, 1]]
                            )

            plt.subplot(int(str(1) + str(len(movements) // partitions) + str(1 + i)))
            plt.imshow(x_data_i, cmap="gray")
            plt.title(
                str(i * partitions * movement_iterations) + " - " + str((i + 1) * partitions * movement_iterations)
            )

            for pos, movements_at_point in all_positions.items():
                avg_movement = np.mean(movements_at_point, axis=0)
                avg_movement /= np.linalg.norm(avg_movement)
                if avg_movement[0] == 0 and avg_movement[1] == 0:
                    plt.scatter(pos[1] + 1, pos[0] + 1, color="hotpink")
                else:
                    plt.arrow(
                        pos[1] + 1,
                        pos[0] + 1,
                        avg_movement[1],
                        avg_movement[0],
                        width=0.2,
                        alpha=min(len(movements_at_point) / 4, 1),
                        color="hotpink",
                    )
                """for movement in movements_at_point:
                    plt.arrow(
                        pos[1] + 1,
                        pos[0] + 1,
                        movement[1],
                        movement[0],
                        width=0.4,
                        alpha=0.2,
                        color="hotpink",
                    )"""

    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    plot_individual_classifications_beliefs()
