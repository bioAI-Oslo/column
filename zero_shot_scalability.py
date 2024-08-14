import os

import matplotlib.pyplot as plt
import numpy as np
from localconfig import config
from main import evaluate_nca, evaluate_nca_batch, get_from_config
from src.data_processing import get_labels, get_MNIST_data, get_MNIST_fashion_data
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
)
from tqdm import tqdm


def get_config(path):
    config.read(path + "/config")

    return config


def get_score_subfolder(sub_path):
    print("Doing sub path:", sub_path)
    # Load the saved network for run "sub_path"
    winner_flat = Logger.load_checkpoint(sub_path)

    # Also load its config
    config = get_config(sub_path)

    moving_nca_kwargs, loss_function, predicting_method, data_func, kwargs = get_from_config(config)

    # Get the data to use for all the tests on this network
    print("Fetching data")
    kwargs["SAMPLES_PER_CLASS"] = NUM_DATA
    # Get test data for new evaluation
    test_data, target_data = data_func(**kwargs)

    # For each test size, record performance on test data
    result_loss = []
    result_acc = []
    for test_size in tqdm(to_test):
        loss, acc = evaluate_nca_batch(
            winner_flat,
            test_data,
            target_data,
            moving_nca_kwargs,
            loss_function,
            predicting_method,
            verbose=False,
            visualize=False,
            return_accuracy=True,
            N_neo=test_size,
            M_neo=test_size,
        )
        result_loss.append(loss)
        result_acc.append(acc)

    return result_acc, result_loss, config.scale.train_n_neo


def plot_all_same_train_neo(to_test, NUM_DATA, path):
    # Record all accuracy and loss lines, means and stds are taken afterwards
    mean_acc = []
    mean_loss = []

    # Loop start
    for numebr, sub_folder in enumerate(os.listdir(path)):  # For each subfolder in folder path
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):  # If it is a folder
            result_acc, result_loss, train_n_neo = get_score_subfolder(sub_path)

            mean_acc.append(result_acc)
            mean_loss.append(result_loss)

    ###### PLOT THE FINDINGS

    cmap = plt.cm.plasma

    fig, ax = plt.subplots(1)

    # Plot a baseline to show how bad you could possibly do with a random policy
    ax.plot(to_test, [0.2 for _ in to_test], label="Random accuracy", color="black")

    # Plot standard deviation for loss and accuracy
    ax.fill_between(
        to_test,
        np.mean(mean_loss, axis=0) - np.std(mean_loss, axis=0),
        np.mean(mean_loss, axis=0) + np.std(mean_loss, axis=0),
        color=cmap(0.2),
        alpha=0.5,
    )
    ax.fill_between(
        to_test,
        np.mean(mean_acc, axis=0) - np.std(mean_acc, axis=0),
        np.mean(mean_acc, axis=0) + np.std(mean_acc, axis=0),
        color=cmap(0.8),
        alpha=0.5,
    )

    # Plot mean loss and accuracy
    ax.plot(to_test, np.mean(mean_loss, axis=0), label="Loss", color=cmap(0.2))
    ax.plot(to_test, np.mean(mean_acc, axis=0), label="Accuracy", color=cmap(0.8))

    # Plot best network's accuracy
    best_network = np.argmax(np.max(mean_acc, axis=1))  # Get best network
    ax.plot(to_test, mean_acc[best_network], label="Best network's accuracy", color=cmap(0.5))

    # Make sure axis goes down to 0
    ax.set_ylim(ymin=0)

    # Make second axis for accuracy
    y2 = ax.twinx()
    y2.set_ylim(ax.get_ylim())
    y2.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    y2.set_ylabel("Accuracy (%)")

    # Finishing up
    ax.set_xticks(to_test, to_test)
    # ax.set_xticks([15], ["\n15"], weight="bold", minor=True) # Hmmmmm
    ax.set_xlabel("NCA size (^2)")
    ax.set_ylabel("Loss")

    ax.legend()

    ################################ Absolute figure ##################################

    # Compare all the networks to their trained for size
    size_trained = np.argmin(np.abs(to_test - config.scale.train_n_neo))

    mean_acc_compared = []
    for scale_score in mean_acc:
        mean_acc_compared.append(scale_score[size_trained] - np.array(scale_score))

    fig, ax = plt.subplots(1)

    # Plot standard deviation for loss and accuracy
    for i, scale_score in enumerate(mean_acc_compared):
        ax.plot(
            to_test,
            1 - scale_score,
            color=cmap(i / (len(mean_acc_compared) - 1) if len(mean_acc_compared) > 1 else 0.5),
        )

    # Make sure axis goes down to 0
    ax.set_ylim(ymin=0)
    ax.set_xticks(to_test, to_test)
    ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    ax.set_xlabel("NCA size (^2)")
    ax.set_ylabel("Retained accuracy (%)")

    plt.show()


def plot_different_train_neos(to_test, NUM_DATA, path):
    # Record all accuracy and loss lines, means and stds are taken afterwards
    findings = {}

    # Loop start
    for numebr, sub_folder in enumerate(os.listdir(path)):  # For each subfolder in folder path
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):  # If it is a folder
            result_acc, result_loss, train_n_neo = get_score_subfolder(sub_path)

            if str(train_n_neo) not in findings:
                findings[str(train_n_neo)] = {}
                findings[str(train_n_neo)]["acc"] = [result_acc]
                findings[str(train_n_neo)]["loss"] = [result_loss]
            else:
                findings[str(train_n_neo)]["acc"].append(result_acc)
                findings[str(train_n_neo)]["loss"].append(result_loss)

    # Plot findings
    cmap = plt.cm.plasma
    fig, ax = plt.subplots(1)

    keys = findings.keys()
    keys = [int(key) for key in keys]
    keys = sorted(keys)

    keys = [str(key) for key in keys]

    count_lines = len(keys)
    for i, train_n_neo in enumerate(keys):
        ax.fill_between(
            to_test,
            np.mean(findings[train_n_neo]["acc"], axis=0) - np.std(findings[train_n_neo]["acc"], axis=0),
            np.mean(findings[train_n_neo]["acc"], axis=0) + np.std(findings[train_n_neo]["acc"], axis=0),
            color=cmap(i / (count_lines - 1)) if count_lines > 1 else cmap(0.5),
            alpha=0.3,
        )
        ax.plot(
            to_test,
            np.mean(findings[train_n_neo]["acc"], axis=0),
            label=train_n_neo,
            color=cmap(i / (count_lines - 1)) if count_lines > 1 else cmap(0.5),
        )

    ax.set_xticks(to_test, to_test)
    ax.set_xlabel("NCA size (^2)")
    ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    ax.set_ylabel("Accuracy (%)")

    ax.legend()

    plt.show()


# Changeables
to_test = np.arange(1, 26 + 1, 1)
NUM_DATA = 10
path = "experiments/neo_size_experiment5"

plot_different_train_neos(to_test, NUM_DATA, path)
