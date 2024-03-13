import os

import matplotlib.pyplot as plt
import numpy as np
from localconfig import config
from main import evaluate_nca, evaluate_nca_batch
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
    scale_loss,
)
from src.mnist_processing import get_MNIST_data
from tqdm import tqdm


def get_config(path):
    config.read(path + "/config")

    return config


# Changeables
to_test = np.arange(1, 26 + 1, 1)
NUM_DATA = 100
path = "experiments/current_pos_winners"

# Record all accuracy and loss lines, means and stds are taken afterwards
mean_acc = []
mean_loss = []

# I want to use the same data for all networks, so I must save it now.
# This means that if not all networks were trained on the same digits, we will get nonsense, so beware
test_data, target_data = None, None

# Loop start
for numebr, sub_folder in enumerate(os.listdir(path)):  # For each subfolder in folder path
    sub_path = path + "/" + sub_folder
    if os.path.isdir(sub_path):  # If it is a folder
        # Load the saved network for run "sub_path"
        winner_flat = Logger.load_checkpoint(sub_path)

        # Also load its config
        config = get_config(sub_path)

        # Fetch info from config and enable environment for testing
        mnist_digits = eval(config.dataset.mnist_digits)

        moving_nca_kwargs = {
            "size_image": (28, 28),
            "num_classes": len(mnist_digits),
            "num_hidden": config.network.hidden_channels,
            "hidden_neurons": config.network.hidden_neurons,
            "iterations": config.network.iterations,
            "position": str(config.network.position),
            "moving": config.network.moving,
            "mnist_digits": mnist_digits,
        }

        loss_function = eval(config.training.loss)
        predicting_method = eval(config.training.predicting_method)

        # Get the data to use for all the tests on this network
        if test_data is None:
            print("Fetching data")
            data_func = get_MNIST_data
            kwargs = {
                "MNIST_DIGITS": mnist_digits,
                "SAMPLES_PER_DIGIT": NUM_DATA,
                "verbose": False,
                "test": True,
            }
            test_data, target_data = data_func(**kwargs)
        else:
            print("Data already loaded, continuing")

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
plt.show()
