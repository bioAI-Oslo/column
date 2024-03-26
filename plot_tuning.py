"""Script for plotting tuning results from the folder experiments/tuning. Should never be imported from. 
If you want to do that, place those functions in plotting_utils.py"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from src.plotting_utils import get_plotting_data, smooth_line
from src.utils import get_config

sns.set()

################################ GLOBALS ################################

path = "./experiments/color_cifar"

# Detect difference
feature1, feature2 = "network.hidden_channels", "network.hidden_neurons"

################################ FUNCTIONS ################################


def get_features(path, feature1, feature2):
    """Gets lists over values the features can have"""
    feature1_list = []
    feature2_list = []

    for sub_folder in os.listdir(path):
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):
            # The IDE might say this isn't used, but it is in the if test below
            # Sorry about the potentially bad code (<.<)
            config = get_config(sub_path)

            # Add non-existent feature to list
            if eval(f"config.{feature1}") not in feature1_list:
                feature1_list.append(eval(f"config.{feature1}"))
            if eval(f"config.{feature2}") not in feature2_list:
                feature2_list.append(eval(f"config.{feature2}"))

    # Sorting for nicer plotting
    try:
        feature1_list = np.sort(feature1_list)
    except TypeError:
        feature1_list = np.array(feature1_list)
        print(
            "Trying to sort None and str. If one of the features isn't str or None, don't trust the output of this script"
        )

    try:
        feature2_list = np.sort(feature2_list)
    except TypeError:
        feature2_list = np.array(feature2_list)
        print(
            "Trying to sort None and str. If one of the features isn't str or None, don't trust the output of this script"
        )

    return feature1_list, feature2_list


def plot_heatmap(heatmap, title, feature1, feature2, high_is_worse=False):
    plt.title(title)
    sns.heatmap(heatmap, annot=True, cmap="plasma" if not high_is_worse else "plasma_r")

    # Lists should be sorted already, so this becomes pretty
    plt.yticks(np.arange(len(feature1_list)) + 0.5, feature1_list)
    plt.xticks(np.arange(len(feature2_list)) + 0.5, feature2_list)

    # Prettier labels made with regex
    ylabel = re.sub("[a-z]+\.", "", feature1).replace("_", " ").title()
    xlabel = re.sub("[a-z]+\.", "", feature2).replace("_", " ").title()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)


def plot_convergence_plots(convergence, title, ylabel, yticks=None):
    # cmap and count_lines will help with color for the lines
    cmap = plt.cm.plasma
    count_lines = sum(1 for row in convergence for sublist in row if sublist)

    plt.figure()
    plt.title(title)
    count_plotted = 0  # For keeping track of which color to use

    for i in range(len(feature1_list)):
        for j in range(len(feature2_list)):
            if convergence[i][j] == []:
                continue  # Skip if there is no data

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

            color = cmap(count_plotted / count_lines)

            plt.fill_between(x_axis[i][j][0][:max_length], mean - std, mean + std, color=color, alpha=0.5)
            plt.plot(
                x_axis[i][j][0][:max_length],
                mean,
                label=str(feature1_list[i]) + " " + str(feature2_list[j]),
                color=color,
            )

            count_plotted += 1

    if yticks is not None:
        plt.yticks(yticks[0], yticks[1])
    plt.ylabel(ylabel)
    plt.xlabel("Generation")
    plt.legend()


def mean_across_inhomogeneous_dimensions(input_array: list):
    """Mean across inhomogeneous dimensions. Numpy doesn't like to do this, so I do it myself"""
    N, M = len(input_array), len(input_array[0])
    means = np.zeros((N, M))
    for x in range(N):
        for y in range(M):
            means[x, y] = np.mean(input_array[x][y])

    return means


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
        key0 = np.where(feature1_list == eval(f"config.{feature1}"))[0][0]
        key1 = np.where(feature2_list == eval(f"config.{feature2}"))[0][0]

        # Add data to heatmap based on key
        heatmap_loss_train[key0][key1].append(data["bestever_score_history"][-1])
        heatmap_loss_test[key0][key1].append(data["test_loss_test_size"][-1])

        heatmap_acc_train[key0][key1].append(data["test_accuracy_train_size"][-1])
        heatmap_acc_test[key0][key1].append(data["test_accuracy_test_size"][-1])

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

        # Same as before
        key0 = np.where(feature1_list == eval(f"config.{feature1}"))[0][0]
        key1 = np.where(feature2_list == eval(f"config.{feature2}"))[0][0]

        x_axis[key0][key1].append(data["x_axis"])

        # Add the full plot to the convergence matrix
        convergence_loss[key0][key1].append(data["training_best_loss_history"])
        convergence_acc[key0][key1].append(data["test_accuracy_train_size"])

# Plot everything
plot_convergence_plots(convergence_loss, title="Loss Convergence", ylabel="Loss")
plot_convergence_plots(
    convergence_acc,
    title="Accuracy Convergence",
    ylabel="Accuracy",
    yticks=(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10)),
)

plt.show()
