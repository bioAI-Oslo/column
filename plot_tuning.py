"""Script for plotting tuning results from the folder experiments/tuning. Should never be imported from. 
If you want to do that, place those functions in plotting_utils.py"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from src.plotting_utils import get_plotting_data, smooth_line
from src.utils import get_config

sns.set()

################################ FUNCTIONS ################################


def get_sigmas(path):
    sigmas = []
    iterations = []

    for sub_folder in os.listdir(path):
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):
            config = get_config(sub_path)

            if config.training.init_sigma not in sigmas:
                sigmas.append(config.training.init_sigma)
            if config.network.iterations not in iterations:
                iterations.append(config.network.iterations)

    sigmas = np.sort(sigmas)
    iterations = np.sort(iterations)

    return sigmas, iterations


def plot_heatmap(heatmap, title, high_is_worse=False):
    plt.title(title)
    sns.heatmap(heatmap, annot=True, cmap="plasma" if not high_is_worse else "plasma_r")
    plt.yticks(np.arange(len(sigmas)) + 0.5, sigmas)
    plt.xticks(np.arange(len(iterations)) + 0.5, iterations)
    plt.ylabel("Sigma")
    plt.xlabel("Iterations")


def plot_convergence_plots(convergence, title, ylabel, yticks=None):
    cmap = plt.cm.plasma
    count_lines = 0
    for i in range(len(sigmas)):
        for j in range(len(iterations)):
            if convergence_loss[i][j] == []:
                continue
            count_lines += 1

    plt.figure()
    plt.title(title)
    count_plotted = 0
    for i in range(len(sigmas)):
        for j in range(len(iterations)):
            if convergence[i][j] == []:
                continue
            mean = np.mean(convergence[i][j], axis=0)
            std = np.std(convergence[i][j], axis=0)

            color = cmap(count_plotted / count_lines)

            plt.fill_between(x_axis[i][j][0], mean - std, mean + std, color=color, alpha=0.5)

            plt.plot(
                x_axis[i][j][0],
                mean,
                label=str(sigmas[i]) + " " + str(iterations[j]),
                color=color,
            )

            count_plotted += 1

    if yticks is not None:
        plt.yticks(yticks[0], yticks[1])
    plt.ylabel(ylabel)
    plt.xlabel("Generation")
    plt.legend()


######################################## HEATMAP ##########################################

path = "./experiments/tuning"

sigmas, iterations = get_sigmas(path)

heatmap = lambda: [[[] for _ in range(len(iterations))] for _ in range(len(sigmas))]
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

        # Key in heatmap based on ordered sigma and iterations
        key0 = np.where(sigmas == config.training.init_sigma)[0][0]
        key1 = np.where(iterations == config.network.iterations)[0][0]

        # Add data to heatmap based on key
        heatmap_loss_train[key0][key1].append(data["bestever_score_history"][-1])
        heatmap_loss_test[key0][key1].append(data["test_loss_test_size"][-1])

        heatmap_acc_train[key0][key1].append(data["test_accuracy_train_size"][-1])
        heatmap_acc_test[key0][key1].append(data["test_accuracy_test_size"][-1])

# Fill empty lists with 1 or 0 to not trip up np.mean
for x in range(len(sigmas)):
    for y in range(len(iterations)):
        for hm in (heatmap_loss_train, heatmap_loss_test):
            if hm[x][y] == []:
                hm[x][y] = [np.nan, np.nan, np.nan, np.nan]  # Loss gets high value
        for hm in (heatmap_acc_train, heatmap_acc_test):
            if hm[x][y] == []:
                hm[x][y] = [np.nan, np.nan, np.nan, np.nan]  # Accuracy gets low value

# Take mean
heatmap_loss_train = np.mean(heatmap_loss_train, axis=2)
heatmap_acc_train = np.mean(heatmap_acc_train, axis=2)
heatmap_acc_test = np.mean(heatmap_acc_test, axis=2)
heatmap_loss_test = np.mean(heatmap_loss_test, axis=2)

# Plot heatmaps
plt.figure()
ax1 = plt.subplot(2, 1, 1)
plot_heatmap(heatmap_loss_train, "Loss Train", high_is_worse=True)
plt.subplot(2, 1, 2, sharex=ax1)
plot_heatmap(heatmap_loss_test, "Loss Test", high_is_worse=True)
plt.figure()
ax1 = plt.subplot(2, 1, 1)
plot_heatmap(heatmap_acc_train, "Accuracy Train", high_is_worse=False)
plt.subplot(2, 1, 2, sharex=ax1)
plot_heatmap(heatmap_acc_test, "Accuracy Test", high_is_worse=False)
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
        key0 = np.where(sigmas == config.training.init_sigma)[0][0]
        key1 = np.where(iterations == config.network.iterations)[0][0]

        x_axis[key0][key1].append(data["x_axis"])

        # Add the full plot to the convergence matrix
        convergence_loss[key0][key1].append(data["training_best_loss_history"])
        convergence_acc[key0][key1].append(data["test_accuracy_train_size"])

# Plt everything
plot_convergence_plots(convergence_loss, title="Loss Convergence", ylabel="Loss")
plot_convergence_plots(
    convergence_acc,
    title="Accuracy Convergence",
    ylabel="Accuracy",
    yticks=(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10)),
)

plt.show()
