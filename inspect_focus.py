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

# Simple pattern
# sub_path = "experiments/simple_pattern/16-7-24_15:33_2"  # Random network used in article

# Simple object
# sub_path = "experiments/simple_object_moving/18-7-24_12:4_10"  # Random network used in article

# MNIST 5 class
# sub_path = "experiments/mnist_final15/28-9-24_13:53"  # Best performer MNIST final 15x15 and 5 class
# sub_path = "experiments/mnist_final/20-8-24_11:51"  # Best performer MNIST final 7x7 and 5 class
# sub_path = "experiments/mnist_final15_2:40/25-7-24_9:51"  # Best performer MNIST final 15x15 and 5 class 2 hc 40 hn

# MNIST 10 class 7x7
# sub_path = "experiments/mnist10_final/11-9-24_9:45"  # Best Mnist 10 class network
# sub_path = "experiments/mnist10_final/10-9-24_9:13_2"
# sub_path = "experiments/mnist10_final/6-9-24_10:29"
# sub_path = "experiments/mnist10_final/5-9-24_12:5"
# sub_path = "experiments/mnist10_final/9-9-24_11:22"  # 2nd best
# sub_path = "experiments/mnist10_final/10-9-24_9:13"
# sub_path = "experiments/mnist10_final/2-9-24_15:10"

# MNIST 10 class 15x15
# sub_path = "experiments/mnist10_final15/30-9-24_9:26"

# FASHION MNIST 10 class 7x7
# sub_path = "experiments/fashion_final/17-9-24_0:28"  # Best fashion 10 class network

# FASHION MNIST 10 class 15x15
# sub_path = "experiments/fashion_final15/26-9-24_13:46_3"  # Best fashion 10 class network
# sub_path = "experiments/fashion_final15/26-9-24_13:46"  # 2nd best
# sub_path = "experiments/fashion_final15/26-9-24_13:46_2"  # 3rd best
# sub_path = "experiments/fashion_final15/26-9-24_13:46_4"  # 4th best


# CIFAR 4 class
sub_path = "experiments/cifar4_final15/6-10-24_13:23_2"  # Best CIFAR 4 class
# sub_path = "experiments/cifar4_final15/6-10-24_13:24"  # 2nd best
# sub_path = "experiments/cifar4_final15/6-10-24_13:24_2"  # 3rd best

############################################################################


def get_data(network, x_data_i):
    """
    Get the perception positions and network classifications for a single data point/image.

    Args:
        network (ActiveNCA): The network to use.
        x_data_i (np.ndarray): The data point/image to classify.

    Returns:
        np.ndarray: The perception positions after each iteration.
        np.ndarray: The network classifications after each iteration.
    """
    # Network should always be reset before use on one datapoint to null the internal state
    network.reset()

    # Initialize arrays to store perception positions and network classifications
    perception_positions = np.zeros((network.iterations + 1, *network.perceptions.shape))
    network_classifications = np.zeros((network.iterations + 1, *network.state.shape[:-1], network.num_classes))

    # Run a classification loop using steps
    for step in range(network.iterations):
        # Save before to get the initial
        perception_positions[step, :, :, :] = deepcopy(network.perceptions)
        network_classifications[step, :, :, :] = deepcopy(network.state[:, :, -network.num_classes :])

        # Progress episode
        _, _ = network.classify(x_data_i, step=step)

    # Save final (these values are technically never used in an episode)
    perception_positions[-1, :, :, :] = deepcopy(network.perceptions)
    network_classifications[-1, :, :, :] = deepcopy(network.state[:, :, -network.num_classes :])

    # Make sure to only return active part of substrate (so removing the pixels at the edges)
    return perception_positions, network_classifications[:, 1:-1, 1:-1, :]


def get_pixel_occupacy(perceptions, x_data_i, count=False, area=False):
    """
    Calculate the frequency of every pixel in an image being occupied by a perception field.

    Args:
        perceptions (list of lists of tuples): The positions of the perception fields after each iteration.
        x_data_i (np.ndarray): The image to get the pixel frequencies from.
        count (bool): Whether to count how many times each pixel is occupied, or just mark it as occupied.
        area (bool): Whether to count the area of the perception field as occupied, or just the single pixel in the center.

    Returns:
        np.ndarray: The frequency of every pixel in the image being occupied.
    """
    frequencies = np.zeros([*x_data_i.shape[:-1]])

    # For every field, record position as visited by putting a 1 there
    for row in perceptions:
        for x, y in row:
            # For neighborhood
            for i in range(0, 1 + 2 * int(area)):
                for j in range(0, 1 + 2 * int(area)):
                    if count:
                        frequencies[int(x) + i, int(y) + j] += 1
                    else:
                        frequencies[int(x) + i, int(y) + j] = 1

    return frequencies


def get_episode_pixel_occupacy(perception_positions, x_data_i, iterations=1, count=False, area=False):
    """
    Calculate the frequency of every pixel in an image being occupied by a perception field over all iterations of an episode.

    Args:
        perception_positions (list of lists of tuples): The positions of the perception fields after each iteration in an episode.
        x_data_i (np.ndarray): The image to get the pixel frequencies from.
        iterations (int): How many iterations to skip between each calculation of pixel occupacy.
        count (bool): Whether to count how many times each pixel is occupied, or just mark it as occupied.
        area (bool): Whether to count the area of the perception field as occupied, or just the single pixel in the center.

    Returns:
        list of np.ndarrays: The frequency of every pixel in the image being occupied for every iteration in the episode.
    """
    # Initialize array to store pixel occupacy per timestep
    pixel_occupacy = []

    # For every timestep (sampled at every other iteration, specified by "iterations")
    for step in range(0, len(perception_positions), iterations):
        # Get pixel occupacy
        pixel_occupacy.append(
            get_pixel_occupacy(
                perception_positions[step],
                x_data_i,
                count=count,
                area=area,
            )
        )

    return pixel_occupacy


def get_movement(perception_positions, x_data_i, iterations=1):
    """
    Calculate the movement of every perception field between every iteration in an episode.

    Args:
        perception_positions (list of lists of tuples): The positions of the perception fields after each iteration in an episode.
        x_data_i (np.ndarray): The image to get the pixel frequencies from. Not used.
        iterations (int): How many iterations to skip between each calculation of pixel occupacy.

    Returns:
        np.ndarray: The movement of every perception field between every iteration in the episode.
    """
    # Initialize array to store movements per timestep
    movements = np.zeros(((len(perception_positions) - 1) // iterations, *perception_positions.shape[1:]))

    # For every timestep (sampled at every other iteration, specified by "iterations")
    for step in range(0, (len(perception_positions) - 1) // iterations):
        # Movement is position after - position before
        movements[step, :, :, :] = (
            perception_positions[step * iterations + iterations, :, :, :]
            - perception_positions[step * iterations, :, :, :]
        )

    return movements


def get_delta_beliefs(network_classifications, iterations=1):
    """
    Calculate the change in the network's beliefs between every iteration in an episode.

    Args:
        network_classifications (list of lists of np.ndarrays): The network's classifications after each iteration in an episode.
        iterations (int): How many iterations to skip between each calculation of delta beliefs.

    Returns:
        np.ndarray: The change in the network's beliefs between every iteration in the episode.
    """
    # Initialize array to store delta beliefs per timestep
    delta_beliefs = np.zeros(((len(network_classifications) - 1) // iterations, *network_classifications.shape[1:]))

    # For every timestep (sampled at every other iteration, specified by "iterations")
    for step in range(0, (len(network_classifications) - 1) // iterations):
        # Delta belief is belief after - belief before
        delta_beliefs[step, :, :, :] = (
            network_classifications[step * iterations + iterations, :, :, :]
            - network_classifications[step * iterations, :, :, :]
        )

    return delta_beliefs


def plot_heatmap(perception_positions, x_data_i, belief, actual_class):
    """
    Plot the input image, a heatmap of the perception fields' positions, and a superposition of the two.

    Args:
        perception_positions (list of lists of tuples): The positions of the perception fields after each iteration in an episode.
        x_data_i (np.ndarray): The image to get the pixel frequencies from.
        belief (str): The label of the class the network believed it was.
        actual_class (str): The label of the class the network was actually classifying.
    """
    pixel_occupacy = get_episode_pixel_occupacy(perception_positions, x_data_i, iterations=1, count=False, area=True)

    fig = plt.figure(frameon=False)
    fig.suptitle("Believed class is " + belief + ", correct class is " + actual_class)

    # Plotting the input image
    plt.subplot(131)
    plt.imshow(x_data_i, cmap="gray")  # "gray" is just ignored for color images
    plt.title("Input image")
    plt.xticks([])
    plt.yticks([])

    # Plotting the heatmap
    heatmap = np.mean(pixel_occupacy, axis=(0))

    plt.subplot(132)
    # cmap_heatmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#00330000", "#FF00FF99", "#FF99FFFF"])
    cmap = "viridis"
    plt.imshow(heatmap, cmap=cmap)
    plt.title("Field position heatmap")
    plt.xticks([])
    plt.yticks([])

    # Plotting the superposition
    plt.subplot(133)
    plt.imshow(x_data_i * 0.7, vmin=0, vmax=1, cmap="gray")  # "gray" is just ignored for color images
    plt.imshow((heatmap) / np.max(heatmap), alpha=(heatmap) / np.max(heatmap), cmap=cmap)
    plt.title("image + heatmap")
    plt.xticks([])
    plt.yticks([])


def plot_fields(perception_positions, x_data_i, belief, actual_class):
    """
    Plot the progression of perception fields over multiple iterations for a given input image.

    Args:
        perception_positions (list of lists of tuples): The positions of the perception fields after each iteration in an episode.
        x_data_i (np.ndarray): The input image data to plot.
        belief (str): The label of the class the network believed it was.
        actual_class (str): The label of the class the network was actually classifying.

    This function plots the input image along with rectangles indicating the perception fields' positions
    for each iteration step. The pink rectangles represent the perception fields, and the function iterates
    over each step, displaying the progression of these fields. The subplot titles indicate the iteration step.
    """
    iterations = 10

    pixel_occupacy = get_episode_pixel_occupacy(
        perception_positions, x_data_i, iterations=iterations, count=False, area=False
    )

    fig = plt.figure(figsize=(10, 3))
    fig.suptitle("Believed class is " + belief + ", correct class is " + actual_class)

    # Plotting per interval of iterations
    for i in range(len(pixel_occupacy)):
        plt.subplot(1, len(pixel_occupacy), i + 1)

        # Plotting the input image, "gray" is just ignored for color images
        plt.imshow(x_data_i, cmap="gray")

        # Plotting the pink fields as rectangles
        ax = plt.gca()
        for x in range(pixel_occupacy[i].shape[0]):
            for y in range(pixel_occupacy[i].shape[1]):
                if pixel_occupacy[i][x, y] >= 1:
                    # Origin of plt.Rectangle is upper left corner
                    # However, I call first axis the X-axis (it goes down, and is therefore plt's y-axis)
                    # Basically, my way of doing thing is switched from plt.imshow and plt.Rectangle
                    # Therefore, I switch x and y for Rectangle
                    rect = plt.Rectangle((y - 0.5, x - 0.5), 3, 3, fill=False, color="mediumvioletred", linewidth=1)
                    ax.add_patch(rect)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Step: " + str(i * iterations))


def plot_path_map(perception_positions, x_data_i, network_classifications, label_belief, label_actual_class, labels):
    """
    Plot a heatmap of the perception fields' movement over the input image over multiple iterations.

    Args:
        perception_positions (list of lists of tuples): The positions of the perception fields after each iteration in an episode.
        x_data_i (np.ndarray): The input image data to plot.
        network_classifications (list of np.ndarrays): The network's classifications after each iteration in an episode.
        label_belief (str): The label of the class the network believed it was.
        label_actual_class (str): The label of the class the network was actually classifying.
        labels (list of str): A list of the labels of all classes.

    The function plots the input image along with arrows indicating the movement of the perception fields
    for each iteration step. The color of the arrows is determined by the class the network believed it was
    at that iteration step. The subplot titles indicate the iteration step.
    """
    # The number of iterations to skip between each calculation of movement (and delta beliefs)
    movement_iterations = 2

    # Get the movement and delta beliefs
    movements = get_movement(perception_positions, x_data_i, movement_iterations)
    positions = perception_positions[:-1:2]
    delta_beliefs = get_delta_beliefs(network_classifications, iterations=movement_iterations)

    # Clarify parameters
    partitions = 5  # number of partitions. So for movement_iterations = 2, partitions = 5 means 10 iterations (5 images for 50 timesteps)
    B = movements.shape[0] // partitions  # B = 25/5 = 5
    N_neo = movements.shape[1]  # Substrate size
    M_neo = movements.shape[2]
    N_img = x_data_i.shape[0]  # Image size
    M_img = x_data_i.shape[1]
    C = network_classifications.shape[-1]  # Number of classes

    # Initialize arrays
    count = np.zeros((B, N_img, M_img))
    avg_movement = np.zeros((B, N_img, M_img, 2))
    opinions = np.zeros((B, N_img, M_img, C))

    # Calculate counts, average movement, and opinions
    for i in range(B):
        for x in range(N_neo):
            for y in range(M_neo):
                for j in range(partitions):
                    x_p, y_p = positions[i * partitions + j, x, y]
                    x_p, y_p = int(x_p), int(y_p)
                    count[i, x_p, y_p] += 1
                    avg_movement[i, x_p, y_p, 0] += movements[i * partitions + j, x, y, 0]
                    avg_movement[i, x_p, y_p, 1] += movements[i * partitions + j, x, y, 1]

                    belief = np.argmax(delta_beliefs[(i * partitions + j), x, y, :])
                    opinions[i, x_p, y_p, belief] += 1

    # Calculate average movement and opinions
    avg_belief = np.zeros((B, N_img, M_img))
    for i in range(B):
        for x in range(N_img):
            for y in range(M_img):
                if count[i, x, y] > 0:
                    avg_movement[i, x, y] /= np.linalg.norm(avg_movement[i, x, y])
                    avg_belief[i, x, y] = np.argmax(opinions[i, x, y])

    # Time to start plotting
    plt.figure()

    # Not colorblind friendly, but necessary for me to distinguish between at most 10 arrows
    cmap = matplotlib.cm.get_cmap("rainbow", C)

    # Plot each iteration
    for i in range(B):
        plt.subplot(int(str(1) + str(B) + str(1 + i)))
        plt.imshow(x_data_i, cmap="gray")  # "gray" is just ignored for color images
        plt.title(str(i * partitions * movement_iterations) + " - " + str((i + 1) * partitions * movement_iterations))
        plt.xticks([])
        plt.yticks([])

        avg_belief_i = np.zeros(C)  # Just for debugging
        # Plot arrows
        for x in range(N_img):
            for y in range(M_img):
                if count[i, x, y] > 0:
                    plt.arrow(
                        y + 1,
                        x + 1,
                        avg_movement[i, x, y, 1],
                        avg_movement[i, x, y, 0],
                        width=0.2,
                        color=cmap(avg_belief[i, x, y] / C),
                        alpha=min(count[i, x, y] / (4 if N_neo > 7 else 3), 1),
                    )
                    avg_belief_i[int(avg_belief[i, x, y])] += 1

        # Plot labels for debugging
        print("Label is " + label_belief + ", and avg opinion is" + str(avg_belief_i))

    # Plot colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ticks=np.linspace(0, 1, C))
    cbar.ax.set_yticklabels(labels)


def plot_beliefs(network_classifications, label_belief, label_actual_class):
    """
    Plot the average beliefs of the system over time.

    Args:
        network_classifications (numpy array): Array of shape (ep_length, N_neo, M_neo, C) with the network's classifications.
        label_belief (str): Label of the belief of the network.
        label_actual_class (str): Label of the actual class of the input.
    """
    ep_length, N_neo, M_neo, C = network_classifications.shape
    # Initialize array of opinions over time
    counter_per_class = np.zeros((ep_length, C))

    # For every timestep
    for i in range(ep_length):
        # For every cell in the substrate
        for j in range(N_neo):
            for k in range(M_neo):
                # Get the class with the highest belief, and add it to the counter
                belief = np.argmax(network_classifications[i, j, k])
                counter_per_class[i, belief] += 1

    # Normalize
    counter_per_class /= N_neo * M_neo

    plt.figure()
    for c in range(C):
        plt.plot(range(ep_length), counter_per_class[:, c], label=labels[c])

    plt.title("Correct class is " + label_actual_class)
    plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, 120, 20))
    plt.ylabel("System beliefs (%)")
    plt.xlabel("Time steps")
    plt.legend()


def plot_delta_beliefs(network_classifications, label_belief, label_actual_class):
    """
    Plot the average change in beliefs of the system over time.

    Args:
        network_classifications (numpy array): Array of shape (ep_length, N_neo, M_neo, C) with the network's classifications.
        label_belief (str): Label of the belief of the network.
        label_actual_class (str): Label of the actual class of the input.
    """
    # How many iterations to interpolate between
    delta_iterations = 2

    delta_beliefs = get_delta_beliefs(network_classifications, iterations=delta_iterations)

    ep_length, N_neo, M_neo, C = delta_beliefs.shape
    # Initialize array of opinions over time
    counter_per_class = np.zeros((ep_length, C))
    print(counter_per_class.shape)

    # For every timestep
    for i in range(ep_length):
        # For every cell in the substrate
        for j in range(N_neo):
            for k in range(M_neo):
                # Get the class with the highest belief, and add it to the counter
                belief = np.argmax(delta_beliefs[i, j, k])
                counter_per_class[i, belief] += 1

    # Normalize
    counter_per_class /= N_neo * M_neo

    for c in range(C):
        plt.plot(range(ep_length), counter_per_class[:, c], label=labels[c])

    plt.title("Correct class is " + label_actual_class)
    plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, 120, 20))
    plt.ylabel("System beliefs (%)")
    plt.xlabel("Time steps")
    plt.legend()
    plt.show()


def get_labelled_belief(class_channels, labels):
    """
    Calculate the label of the belief from class channels.

    Args:
        class_channels (np.ndarray): Multi-dimensional array representing the class channels.
        labels (list of str): List of labels corresponding to the classes.

    Returns:
        str: The label of the class with the highest belief.
    """
    belief = np.mean(class_channels, axis=(-3, -2))
    label_belief = get_label_from_belief_vector(belief, labels)

    return label_belief


def get_label_from_belief_vector(belief_vector, labels):
    """
    Get the label of the belief from a belief vector.

    Args:
        belief_vector (np.ndarray): Vector representing the class channels.
        labels (list of str): List of labels corresponding to the classes.

    Returns:
        str: The label of the class with the highest belief.
    """
    numerical_belief = np.argmax(belief_vector)
    label_belief = labels[numerical_belief]

    return label_belief


if __name__ == "__main__":
    # Seed for MNIST and CIFAR used in article
    """np.random.seed(42)
    random.seed(42)"""

    # Seed for fashion MNIST used in article
    np.random.seed(24)
    random.seed(24)

    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    # For each datapoint
    for x_data_i, y_data_i in zip(test_data, target_data):
        # Get data
        perception_positions, network_classifications = get_data(network, x_data_i)

        # Get label
        label_belief = get_labelled_belief(network_classifications[-1], labels)
        label_actual_class = get_label_from_belief_vector(y_data_i, labels)

        plot_heatmap(perception_positions, x_data_i, label_belief, label_actual_class)

        # plot_fields(perception_positions, x_data_i, label_belief, label_actual_class)

        # plot_path_map(perception_positions, x_data_i, network_classifications, label_belief, label_actual_class, labels)

        # plot_beliefs(network_classifications, label_belief, label_actual_class)

        # plot_delta_beliefs(network_classifications, label_belief, label_actual_class)

    plt.show()
