"""Script for plotting tuning results from the folder experiments/. Should never be imported from. 
If you want to do that, place those functions in plotting_utils.py"""

import os
import re

import cmcrameri.cm as cmc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from src.plotting_utils import get_plotting_data, get_smoothing_factor, smooth_line
from src.utils import get_config

sns.set()

################################ GLOBALS ################################

path = "./experiments/cifar4_final15"

# Detect difference
feature1, feature2 = "scale.train_n_neo", "training.lambda_weight"

################################ FUNCTIONS ################################


def sort_features(feature_list):
    """Sorts a list of features. If it contains only strings or None, sorts these lexicographically. If it contains other types, converts to numpy array first.

    Args:
        feature_list (list): List of features to sort.

    Returns:
        list or numpy.ndarray: Sorted list of features.
    """
    print(feature_list)
    try:
        feature_list = np.sort(feature_list)
    except TypeError:
        feature_list = np.array(feature_list)
        print(
            "Trying to sort None and str. If one of the features isn't str or None, don't trust the output of this script"
        )

    print(feature_list)
    return feature_list


def get_features(path, feature1, feature2):
    """
    Retrieves two lists of unique features from all the subfolders in path.

    The two features are given by feature1 and feature2. If the same feature is
    used in two different subfolders, it is only added once to the list.

    The lists are then sorted in-place with sort_features for nicer plotting.

    Args:
        path (str): Path to folder containing all subfolders with the features.
        feature1 (str): Feature to retrieve. Should be a valid attribute of config.
        feature2 (str): Feature to retrieve. Should be a valid attribute of config.

    Returns:
        tuple: Two lists of unique features, sorted in-place.
    """
    feature1_list = []
    feature2_list = []

    for sub_folder in os.listdir(path):
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):
            # The IDE might say this isn't used, but it is in the if test below
            # Sorry about the potentially bad code (<.<)
            config = get_config(sub_path)

            # Add non-existent feature to list
            feature1_eval = eval(f"config.{feature1}")
            feature2_eval = eval(f"config.{feature2}")

            if feature1_eval not in feature1_list:
                feature1_list.append(feature1_eval)

            if feature2_eval not in feature2_list:
                feature2_list.append(feature2_eval)

    # Sorting for nicer plotting
    feature1_list = sort_features(feature1_list)
    feature2_list = sort_features(feature2_list)

    return feature1_list, feature2_list


def plot_heatmap(heatmap, title, feature1, feature2, high_is_worse=False):
    """
    Plot a heatmap from a numpy array heatmap. The heatmap is annotated with the
    values from the array. The x and y axes are labeled with the values from the
    lists feature1 and feature2. The title of the plot is set to title.

    Args:
        heatmap (numpy.ndarray): The array to plot as a heatmap.
        title (str): The title of the plot.
        feature1 (list): The values to label the y-axis with.
        feature2 (list): The values to label the x-axis with.
        high_is_worse (bool, optional): If True, the colors of the heatmap will be reversed.
            This is useful for plotting tuning results where higher values are
            worse. Defaults to False.
    """
    plt.title(title)
    # Modified Plasma palette to be more colorfriendly (but idk if I succeeded)
    colors = [
        "#2C2463",
        "#DC267F",
        "#EF792A",
        "#FFD958",
    ]
    if high_is_worse:
        colors.reverse()
    # Making a colormap with those colors
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    # Plotting
    if high_is_worse:
        sns.heatmap(heatmap, annot=True, cmap=cmap, square=True)
    else:
        sns.heatmap(heatmap, annot=True, cmap=cmap, square=True, vmin=0, vmax=1)

    # Lists should be sorted already, so this becomes pretty
    plt.yticks(np.arange(len(feature1_list)) + 0.5, feature1_list)  # +0.5 to center
    plt.xticks(np.arange(len(feature2_list)) + 0.5, feature2_list)

    # Prettier labels made with regex
    ylabel = re.sub("[a-z]+\.", "", feature1).replace("_", " ").title()
    xlabel = re.sub("[a-z]+\.", "", feature2).replace("_", " ").title()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


def plot_convergence_plots(convergence, title, ylabel, yticks=None, smoothing=False):

    # cmap and count_lines will help with color for the lines
    # cmap = plt.cm.plasma
    """
    Plots convergence plots for given features and parameters.

    Args:
        convergence (list): A 4D list containing convergence data for different
            combinations of features. Each element is a list of lists representing
            different runs.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        yticks (tuple, optional): A tuple containing the y-tick positions and labels.
            Defaults to None.
        smoothing (bool, optional): If True, applies smoothing to the convergence lines.
            Defaults to False.

    The plot will display the mean and standard deviation of the convergence data
    across runs, with lines colored based on a colormap. The x-axis represents the
    generation number.
    """
    # Modified IBM design library palette
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#7597FF", "#6A52DC", "#DC267F", "#FFB000"])

    # Counting the number of lines
    count_lines = sum(1 for row in convergence for sublist in row if sublist)

    plt.figure()
    plt.title(title)
    count_plotted = 0  # For keeping track of which color to use

    # Going through the matrix of features
    for i in range(len(feature1_list)):
        for j in range(len(feature2_list)):
            # Skip if there is no data
            if convergence[i][j] == []:
                continue

            # While generating the data, runs might not be completed, and we wait until the last run is completed before plotting the full line
            max_length = np.inf
            for line in convergence[i][j]:
                if len(line) < max_length:
                    max_length = len(line)
            print("Length data", max_length, "Feature 1", feature1_list[i], "Feature 2", feature2_list[j])

            # Get the "shorter" version of each line
            shorter = []
            for h in convergence[i][j]:
                shorter.append(h[:max_length])

            # Get mean and std
            mean = np.mean(shorter, axis=0)
            std = np.std(shorter, axis=0)
            if smoothing:
                smoothing_factor = get_smoothing_factor(max_length)
                mean = smooth_line(smoothing_factor, mean)
                std = smooth_line(smoothing_factor, std)

            # Which color to use?
            color = cmap(0.5 if count_lines == 1 else count_plotted / (count_lines - 1))

            # Debugging info
            print(feature1, feature2, type(feature1_list[i]), type(feature2_list[j]))

            # Special case for network.moving and network.position
            if feature1 == "network.moving" and feature2 == "network.position":
                cmap_pos = 2 if feature1_list[i] == True else 0
                cmap_pos += 1 if feature2_list[j] == None else 0
                color = cmap(cmap_pos / 3)

            # Plot shaded area
            plt.fill_between(x_axis[i][j][0][:max_length], mean - std, mean + std, color=color, alpha=0.5)

            # Pretty labels
            feature1_name = re.sub("[a-z]+\.", "", feature1).replace("_", " ").title()
            feature2_name = re.sub("[a-z]+\.", "", feature2).replace("_", " ").title()

            # Plot the line
            plt.plot(
                x_axis[i][j][0][:max_length],
                mean,
                label=feature1_name + ": " + str(feature1_list[i]) + " " + feature2_name + ": " + str(feature2_list[j]),
                color=color,
            )

            count_plotted += 1

    if yticks is not None:
        plt.yticks(yticks[0], yticks[1])
    plt.ylabel(ylabel)
    plt.xlabel("Generation")
    plt.legend()


def plot_convergence_plots_total(convergence, title, ylabel, yticks=None):
    # cmap and count_lines will help with color for the lines
    """
    Plot multiple convergence plots for different combinations of features.

    Args:
        convergence (list): A 4D list containing convergence data for different
            combinations of features. Each element is a list of lists representing
            different runs.
        title (str): The title of the plot.
        ylabel (str): The label for the y-axis.
        yticks (tuple, optional): A tuple containing the y-tick positions and labels.
            Defaults to None.

    The plot will display the convergence data across runs, with lines colored based
    on a colormap. The x-axis represents the generation number. The convergence data
    is split into subplots for each combination of features.
    """
    # Just using plasma cmap, this is just for me
    cmap = plt.cm.plasma

    # Counting the number of lines
    count_lines = sum(1 for row in convergence for sublist in row if sublist)

    plt.figure()
    plt.title(title)
    count_plotted = 0  # For keeping track of which color to use

    N, M = len(feature1_list), len(feature2_list)

    # Going through the matrix of features
    for i in range(N):
        for j in range(M):
            # Skip if there is no data
            if convergence[i][j] == []:
                continue

            # Which color to use?
            color = cmap(count_plotted / count_lines)

            plt.subplot(N, M, i * M + j + 1)

            # Plot the line for each run for this set of features
            for k, line in enumerate(convergence[i][j]):
                plt.plot(x_axis[i][j][k], line, color=color)

            # Pretty labels, not used currently
            feature1_name = re.sub("[a-z]+\.", "", feature1).replace("_", " ").title()
            feature2_name = re.sub("[a-z]+\.", "", feature2).replace("_", " ").title()

            ax = plt.gca()
            ax.set_title(str(feature1_list[i]) + " " + str(feature2_list[j]))

            # Put axes labels only at edges
            if i == N - 1:
                ax.set_xlabel("Generation")
            if j == 0:
                ax.set_ylabel(ylabel)
            if yticks is not None:
                ax.set_yticks(yticks[0], yticks[1])

            count_plotted += 1
    plt.legend()


def mean_across_inhomogeneous_dimensions(input_array: list):
    # Mean across inhomogeneous dimensions. Numpy doesn't like to do this, so I do it myself
    """
    Takes a 3D list of lists, and returns the mean across each of the inner lists.
    This is useful for taking the mean across inhomogeneous arrays, which numpy
    doesn't support.

    Args:
        input_array (list): A 3D list of lists.

    Returns:
        means (numpy.ndarray):
            A 2D array of the same shape as input_array, where each element is the
            mean of the corresponding element in input_array.
    """
    N, M = len(input_array), len(input_array[0])
    means = np.zeros((N, M))
    for x in range(N):
        for y in range(M):
            means[x, y] = np.mean(input_array[x][y])

    return means


def get_keys(feature1, feature2, feature1_list, feature2_list):
    """
    Return the indices of the given features in the feature lists.

    Args:
        feature1 (str): The name of the first feature.
        feature2 (str): The name of the second feature.
        feature1_list (list): The list of values of the first feature.
        feature2_list (list): The list of values of the second feature.

    Returns:
        tuple: A tuple of two integers, the first representing the index of feature1 in feature1_list, and the second representing the index of feature2 in feature2_list.
    """

    def get_key(feature, feature_list):
        feature_eval = eval(f"config.{feature}")
        print(feature_eval)
        if type(feature_eval) not in [str, int, float, list]:
            return np.where(feature_list == str(feature_eval))[0][0]
        else:
            if feature_eval not in feature_list:
                return np.where(feature_list == str(feature_eval))[0][0]
            return np.where(feature_list == feature_eval)[0][0]

    key0 = get_key(feature1, feature1_list)
    key1 = get_key(feature2, feature2_list)

    return key0, key1


######################################## HEATMAP ##########################################

# Get feature 1 and feature 2
feature1_list, feature2_list = get_features(path, feature1, feature2)

heatmap = lambda: [[[] for _ in range(len(feature2_list))] for _ in range(len(feature1_list))]
heatmap_loss_train = heatmap()
heatmap_acc_train = heatmap()
heatmap_loss_test = heatmap()
heatmap_acc_test = heatmap()

# For all subfolders, get the data
for sub_folder in os.listdir(path):
    sub_path = path + "/" + sub_folder
    if os.path.isdir(sub_path):
        # Data and config
        data = get_plotting_data(sub_path)
        config = get_config(sub_path)

        # Key in heatmap based on ordered feature 1 and feature 2
        key0, key1 = get_keys(feature1, feature2, feature1_list, feature2_list)

        # Add data to heatmap based on key
        # Use the last 10 samples for mean because optimization is highly stochastic
        heatmap_loss_train[key0][key1].append(np.mean(data["test_loss_train_size"][-10:]))
        heatmap_loss_test[key0][key1].append(np.mean(data["test_loss_test_size"][-10:]))

        heatmap_acc_train[key0][key1].append(np.mean(data["test_accuracy_train_size"][-10:]))
        heatmap_acc_test[key0][key1].append(np.mean(data["test_accuracy_test_size"][-10:]))

# Fill empty lists with 1 or 0 to not trip up np.mean
for x in range(len(feature1_list)):
    for y in range(len(feature2_list)):
        for hm in (heatmap_loss_train, heatmap_loss_test):
            if hm[x][y] == []:
                hm[x][y] = [np.nan, np.nan, np.nan, np.nan]  # Loss gets nan
        for hm in (heatmap_acc_train, heatmap_acc_test):
            if hm[x][y] == []:
                hm[x][y] = [np.nan, np.nan, np.nan, np.nan]  # Accuracy gets nan

# Take mean
heatmap_loss_train = mean_across_inhomogeneous_dimensions(heatmap_loss_train)
heatmap_acc_train = mean_across_inhomogeneous_dimensions(heatmap_acc_train)
heatmap_acc_test = mean_across_inhomogeneous_dimensions(heatmap_acc_test)
heatmap_loss_test = mean_across_inhomogeneous_dimensions(heatmap_loss_test)

# Plot heatmaps
plt.figure()
ax1 = plt.subplot(2, 1, 1)
plot_heatmap(heatmap_loss_train, "Loss Train", feature1, feature2, high_is_worse=True)
plt.subplot(2, 1, 2, sharex=ax1)
plot_heatmap(heatmap_loss_test, "Loss Test", feature1, feature2, high_is_worse=True)
plt.figure()
ax1 = plt.subplot(2, 1, 1)
plot_heatmap(heatmap_acc_train, "Accuracy Train", feature1, feature2, high_is_worse=False)
plt.subplot(2, 1, 2, sharex=ax1)
plot_heatmap(heatmap_acc_test, "Accuracy Test", feature1, feature2, high_is_worse=False)
plt.show()

######################################## CONVERGENCE ##########################################

convergence_loss = heatmap()
convergence_acc = heatmap()

x_axis = heatmap()

# For every subfolder, get convergence plot for run and add to convergence
for sub_folder in os.listdir(path):
    sub_path = path + "/" + sub_folder
    if os.path.isdir(sub_path):
        data = get_plotting_data(sub_path)
        config = get_config(sub_path)

        # Key in heatmap based on ordered feature 1 and feature 2
        key0, key1 = get_keys(feature1, feature2, feature1_list, feature2_list)

        x_axis[key0][key1].append(data["x_axis"])

        # Add the full plot to the convergence matrix
        convergence_loss[key0][key1].append(data["test_loss_train_size"])
        convergence_acc[key0][key1].append(data["test_accuracy_train_size"])

# Plot everything
plot_convergence_plots(convergence_loss, title="Loss Convergence", ylabel="Loss")
plot_convergence_plots(
    convergence_acc,
    title="Accuracy Convergence",
    ylabel="Accuracy",
    yticks=(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10)),
)

# Plot everything with smoothing
plot_convergence_plots(convergence_loss, title="Loss Convergence Smoothed", ylabel="Loss", smoothing=True)
plot_convergence_plots(
    convergence_acc,
    title="Accuracy Convergence Smoothed",
    ylabel="Accuracy",
    yticks=(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10)),
    smoothing=True,
)

plot_convergence_plots_total(convergence_loss, title="Loss Convergence All", ylabel="Loss")
plot_convergence_plots_total(
    convergence_acc,
    title="Accuracy Convergence All",
    ylabel="Accuracy",
    yticks=(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10)),
)

plt.show()
