"""Script to analyze a single network's group behavior. Do not import from here."""

import random
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from common_funcs import get_network
from localconfig import config
from src.plotting_utils import get_plotting_ticks
from numba import jit
from scipy.spatial.distance import jensenshannon
import ot  # pip install POT

############################### Change here ################################

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
sub_path = "experiments/mnist10_final/11-9-24_9:45"  # Best Mnist 10 class network
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
# sub_path = "experiments/cifar4_final15/6-10-24_13:23_2"  # Best CIFAR 4 class
# sub_path = "experiments/cifar4_final15/6-10-24_13:24"  # 2nd best
# sub_path = "experiments/cifar4_final15/6-10-24_13:24_2"  # 3rd best

############################################################################


def get_perception(network, x_data_i, y_data_i, visualize=False):
    # Network should always be reset before use on one datapoint to null the internal state
    """
    Get the perception positions for a single datapoint/image.

    Args:
        network (ActiveNCA): The network to use.
        x_data_i (np.ndarray): The data point/image to classify.
        y_data_i (np.ndarray): The correct label for the data point/image.
        visualize (bool): Whether to visualize the perception positions after each iteration.

    Returns:
        np.ndarray: The perception positions after each iteration.
    """
    network.reset()

    perception_positions = np.zeros((network.iterations, *network.perceptions.shape))

    # Run a classification loop using steps
    for step in range(network.iterations):
        perception_positions[step, :, :, :] = deepcopy(network.perceptions)

        # Progress episode
        _, _ = network.classify(x_data_i, step=step, correct_label_index=y_data_i.argmax(), visualize=visualize)

    return perception_positions


def get_pos_avg(network, x_data_i, y_data_i):

    perception_positions = get_perception(network, x_data_i, y_data_i)

    avg_pos = np.mean(perception_positions, axis=(1, 2))

    return avg_pos


def show_path_map(network, x_data_i, y_data_i):
    avg_pos = get_pos_avg(network, x_data_i, y_data_i)

    delta_pos = avg_pos[1:] - avg_pos[:-1]

    # Arrows in direction
    plt.figure()
    plt.imshow(x_data_i, cmap="gray")
    plt.quiver(
        avg_pos[:-1, 1],
        avg_pos[:-1, 0],
        delta_pos[:, 1],
        delta_pos[:, 0],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="r",
    )


def plot_similarity():
    # Based on difference between mean of distributions, but doesn't say anything about the distribution. Not ideal.
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    # Calculate the avergae positions of every image class
    pos_avgs = []

    for x_data_i, y_data_i in zip(test_data, target_data):
        avg_pos = get_pos_avg(
            network,
            x_data_i=x_data_i,
            y_data_i=y_data_i,
        )

        pos_avgs.append(avg_pos)

    pos_avgs = np.array(pos_avgs)

    # Calculate similarity between every image. Meaning that for class i, it is compared to every other class and also itself
    similarities = np.zeros((pos_avgs.shape[0], pos_avgs.shape[0]))

    for pos_avg, target in zip(pos_avgs, target_data):
        for pos_avg2, target2 in zip(pos_avgs, target_data):
            similarity = np.mean(((pos_avg - pos_avg2) ** 2))
            similarities[target.argmax(), target2.argmax()] += float(similarity)

    # Reorder the classes to visually show the similarity (follows the same ordering as the confusion matrix in the paper)
    new_ordering = [3, 0, 2, 4, 6, 8, 1, 5, 7, 9]

    new_conf = np.zeros((10, 10))
    for i in range(10):
        class_i = new_ordering[i]
        for j in range(10):
            class_j = new_ordering[j]
            new_conf[i][j] = similarities[class_i][class_j] / NUM_DATA**2

    # Heatmap
    import seaborn as sns

    sns.heatmap(new_conf, annot=True, cmap="plasma_r", cbar=False, square=True)
    plt.xticks(np.arange(10), np.array(labels)[new_ordering], rotation=45)
    plt.yticks(np.arange(10), np.array(labels)[new_ordering], rotation=45)
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.title("Similarity between class behaviors")

    plt.show()


def plot_wasserstein_distance_mnist(interval_start=0, interval_end=50):
    # Plots difference in distributions in terms of earth mover distance. Much better than only comparing mean or clustering :)
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    _, N, M, _ = test_data.shape

    # Gather perceptions
    perceptions = []

    for x_data_i, y_data_i in zip(test_data, target_data):
        pos = get_perception(network, x_data_i, y_data_i)
        perceptions.append(pos)

    perceptions = np.array(perceptions)

    # Calculate similarity between every image class bheavior. Meaning that for class i, it is compared to every other class and also itself
    # We also count the number of comparisons made because I want to be sure I get it right :) Could caluclate it but safer this way
    similarities = np.zeros((perceptions.shape[0], perceptions.shape[0]))
    counts = np.zeros((perceptions.shape[0], perceptions.shape[0]))

    for pos_avg, target in tqdm(zip(perceptions, target_data)):
        for pos_avg2, target2 in zip(perceptions, target_data):
            # Ignore the first 20 steps because start position is the same
            ws_distance = get_wasserstein_distance(
                pos_avg[interval_start:interval_end], pos_avg2[interval_start:interval_end]
            )

            similarities[target.argmax(), target2.argmax()] += float(ws_distance)
            counts[target.argmax(), target2.argmax()] += 1

    # Reorder the classes to visually show the similarity
    new_ordering = [0, 2, 3, 5, 6, 8, 1, 4, 7, 9]

    num_classes = len(labels)

    new_conf = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        class_i = new_ordering[i]
        for j in range(num_classes):
            class_j = new_ordering[j]
            new_conf[i][j] = similarities[class_i][class_j] / counts[class_i][class_j]

    # Printing
    print("The ws distance between a class and each other class is:")
    print(np.round(new_conf, 2))

    print("The ordering is:")
    print(np.array(labels)[new_ordering])

    print("The average ws distance within a class is:")
    within_class = np.mean([new_conf[i][i] for i in range(num_classes)])
    print(within_class)

    print("The average ws distance within the round group is:")
    within_round = np.mean([new_conf[i][j] for i in range(5) for j in range(5) if i != j])
    print(within_round)

    print("The avergae ws distance within the pointy group is:")
    within_pointy = np.mean([new_conf[i][j] for i in range(6, 10) for j in range(6, 10) if i != j])
    print(within_pointy)

    print("The average ws distance between the round and pointy groups is:")
    between = np.mean([new_conf[i][j] for i in range(5) for j in range(6, 10)])
    print(between)

    return within_class, within_round, within_pointy, between


def plot_wasserstein_distance_fashion(interval_start=0, interval_end=50):
    # Plots difference in distributions in terms of earth mover distance. Much better than only comparing mean or clustering :)
    # Get network and data
    network, labels, data_func, kwargs, predicting_method = get_network(sub_path, NUM_DATA)
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    _, N, M, _ = test_data.shape

    # Gather perceptions
    perceptions = []

    for x_data_i, y_data_i in zip(test_data, target_data):
        pos = get_perception(network, x_data_i, y_data_i)
        perceptions.append(pos)

    perceptions = np.array(perceptions)

    # Calculate similarity between every image class bheavior. Meaning that for class i, it is compared to every other class and also itself
    # We also count the number of comparisons made because I want to be sure I get it right :) Could caluclate it but safer this way
    similarities = np.zeros((perceptions.shape[0], perceptions.shape[0]))
    counts = np.zeros((perceptions.shape[0], perceptions.shape[0]))

    for pos_avg, target in tqdm(zip(perceptions, target_data)):
        for pos_avg2, target2 in zip(perceptions, target_data):
            # Ignore the first 20 steps because start position is the same
            ws_distance = get_wasserstein_distance(
                pos_avg[interval_start:interval_end], pos_avg2[interval_start:interval_end]
            )

            similarities[target.argmax(), target2.argmax()] += float(ws_distance)
            counts[target.argmax(), target2.argmax()] += 1

    # Reorder the classes to visually show the similarity (follows the same ordering as the confusion matrix in the paper)
    new_ordering = [3, 0, 2, 4, 6, 8, 1, 5, 7, 9]
    # new_ordering = [0, 2, 3, 5, 6, 8, 1, 4, 7, 9]

    num_classes = len(labels)

    new_conf = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        class_i = new_ordering[i]
        for j in range(num_classes):
            class_j = new_ordering[j]
            new_conf[i][j] = similarities[class_i][class_j] / counts[class_i][class_j]

    # Heatmap
    import seaborn as sns

    sns.heatmap(new_conf, annot=np.round(new_conf, 2), cmap="plasma_r", cbar=False, square=True)
    plt.xticks(np.arange(num_classes), np.array(labels)[new_ordering], rotation=45)
    plt.yticks(np.arange(num_classes), np.array(labels)[new_ordering], rotation=-45)
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.title("Normalized Wasserstein distance between class behaviors")

    plt.show()

    # Printing
    print("The ws distance between a class and each other class is:")
    print(np.round(new_conf, 2))

    print("The ordering is:")
    print(np.array(labels)[new_ordering])

    print("The average ws distance within a class is:")
    within_class = np.mean([new_conf[i][i] for i in range(num_classes)])
    print(within_class)

    print("The average ws distance within the torso garment group is:")
    within_torso = np.mean([new_conf[i][j] for i in range(5) for j in range(5) if i != j])
    print(within_torso)

    print("The avergae ws distance within the shoe group is:")
    within_shoe = np.mean([new_conf[i][j] for i in range(7, 10) for j in range(7, 10) if i != j])
    print(within_shoe)

    print("The average ws distance between the torso and shoe groups is:")
    between = np.mean([new_conf[i][j] for i in range(5) for j in range(7, 10)])
    print(between)

    return within_class, within_torso, within_shoe, between


@jit
def mean_dist_to_all_else(perceptions):
    T, N, M, _ = perceptions.shape
    dist = 0
    counter = 0
    for t in range(T):
        for x in range(N):
            for y in range(M):
                for i in range(N):
                    for j in range(M):
                        if i == x and j == y:
                            continue
                        dist += np.mean(np.abs((perceptions[t, x, y] - perceptions[t, i, j])))
                        counter += 1

    dist /= counter

    return dist


def get_jensen_shannon_divergence(perceptions1, perceptions2):
    total_js_distance = 0

    for perc1, perc2 in zip(perceptions1, perceptions2):
        arr1 = perc1.reshape(-1, 2)
        arr2 = perc2.reshape(-1, 2)

        # Step 1: Create 2D histograms
        bins = (13, 13)
        range_ = [[0, 26], [0, 26]]  # x and y ranges

        H1, _, _ = np.histogram2d(arr1[:, 0], arr1[:, 1], bins=bins, range=range_)
        H2, _, _ = np.histogram2d(arr2[:, 0], arr2[:, 1], bins=bins, range=range_)

        # Step 2: Normalize to get probability distributions
        P = H1 / H1.sum()
        Q = H2 / H2.sum()

        # Step 3: Flatten histograms to 1D vectors
        P_flat = P.ravel()
        Q_flat = Q.ravel()

        # Step 4: Add small epsilon to avoid zeros (optional but good practice)
        eps = 1e-12
        P_flat = np.maximum(P_flat, eps)
        Q_flat = np.maximum(Q_flat, eps)
        P_flat /= P_flat.sum()
        Q_flat /= Q_flat.sum()

        # Step 5: Compute Jensen–Shannon divergence
        js_distance = jensenshannon(P_flat, Q_flat)  # This returns the *square root* of JSD
        total_js_distance += js_distance**2

    return total_js_distance / len(perceptions1)


def get_wasserstein_distance(perceptions1, perceptions2):
    # Bins is less than pixels for speed. Metric is approximately the same if using 26x26, but it's faster this way
    bins = (13, 13)
    range_ = [[0, 26], [0, 26]]

    # Precompute grid coordinates and pairwise distance matrix
    x = np.linspace(0, 25, bins[0])
    y = np.linspace(0, 25, bins[1])
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T  # (676, 2)

    M = ot.dist(coords, coords)  # pairwise Euclidean distances
    M /= M.max()  # normalize so distances are in [0, 1]

    total_ws = 0

    # We're not going to average over all time steps, only every 10.
    # This is because I want to make sure that someone being at pos (x,y) is not counted as
    # the same if it happens at time t1 and if it happens at time t2.
    # However, doing it every timestep yielded asymmetric EMD, which seemed to be because of sparse histograms. When averaging sligthy over time,
    # the asymmetry dissapeared. It seemed more correct this way, because it was was closer to the results I found with the
    # difference-between-distribution-means metric (plot_similarity). And because it was not symmetric. I assume the sparisty yielded slower or unsteady convergence
    # in the optimizer in POT, leading to asymetric results. They had examples of sparser histograms on their website, but they didn't compute the metric twice like I do.
    # Maybe I shouldn't do it twice, it's kind of intensive. But it made me catch that error so that's pretty good :)
    time_grid_size = 10
    time_grid_steps = len(perceptions1) // time_grid_size

    for t in range(0, len(perceptions1), time_grid_size):
        # Unravel 3D array to 2D, keep only the time interval we currently view
        arr1 = perceptions1[t : t + time_grid_size].reshape(-1, 2)
        arr2 = perceptions2[t : t + time_grid_size].reshape(-1, 2)

        # Compute 2D histograms
        H1, _, _ = np.histogram2d(arr1[:, 0], arr1[:, 1], bins=bins, range=range_)
        H2, _, _ = np.histogram2d(arr2[:, 0], arr2[:, 1], bins=bins, range=range_)

        # Normalize to get probability distributions
        P = H1 / H1.sum()
        Q = H2 / H2.sum()

        # Flatten histograms to 1D vectors. 2D relationships are maintained by M
        a = P.ravel()
        b = Q.ravel()

        # Compute 2D Wasserstein (Earth Mover's Distance)
        ws_distance = ot.emd2(a, b, M)  # squared cost, EMD^2
        ws_distance = np.sqrt(ws_distance)  # square root
        total_ws += ws_distance

    # Average over all time steps
    return total_ws / time_grid_steps


if __name__ == "__main__":
    # Seed for fashion MNIST used in article
    np.random.seed(24)
    random.seed(24)

    NUM_DATA = 40

    ws_table = []

    for t in range(0, 50, 10):
        within_class, within_torso, within_shoe, between = plot_wasserstein_distance_fashion(
            interval_start=t, interval_end=t + 10
        )

        ws_table.append([within_class, within_torso, within_shoe, between])

    ws_table = np.array(ws_table).T

    import seaborn as sns

    sns.heatmap(ws_table, annot=np.round(ws_table, 3), cmap="plasma", cbar=False, square=False)
    plt.xticks(np.arange(0, 5) + 0.5, ["0-10", "10-20", "20-30", "30-40", "40-50"])
    plt.yticks(np.arange(0, 4) + 0.5, ["Within class", "Within torso", "Within shoe", "Between"], rotation=0)
    plt.xlabel("Timestep interval")
    plt.title("Normalized EMD distance")

    plt.show()
