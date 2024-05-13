"""This script is for plotting the results of a run. Never import from here. If you want to do that, place those functions in plotting_utils.py"""

import argparse
import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.plotting_utils import get_plotting_data, get_smoothing_factor, smooth_line
from src.utils import get_config

sns.set()

################################ PARSER & DATA ################################

parser = argparse.ArgumentParser(
    prog="PlotRunner", description="This program plots runs based on config.", epilog="Text at the bottom of help"
)
parser.add_argument("path", type=str, help="The stored run to plot")

args = parser.parse_args()

# Get data
data = get_plotting_data(args.path)
x = data["x_axis"]

# Calculate how much to smooth data based on resolution
smoothing_factor = get_smoothing_factor(len(x))
print("Smoothing factor:", smoothing_factor)

################################ LOSS ################################

plt.figure()
# Plotting smoothed mean
smooth_mean = smooth_line(smoothing_factor, data["mean_loss_history"])
smooth_std = smooth_line(smoothing_factor, data["std_loss_history"])
plt.fill_between(
    x,
    np.array(smooth_mean) - np.array(smooth_std),
    np.array(smooth_mean) + np.array(smooth_std),
    color="#005F6055",
)
plt.plot(x, smooth_mean, label="Smoothed mean with std", linewidth=3, color="#005F60")

# Plot raw data first in pale
plt.plot(x, data["training_best_loss_history"], label="Best training loss", color="#249EA055")
# Then smoothed data in opaque
plt.plot(
    x,
    smooth_line(smoothing_factor, data["training_best_loss_history"]),
    label="Smoothed",
    linewidth=3,
    color="#249EA0",
)
# Continue with pale and opaque pattern
plt.plot(x, data["test_loss_train_size"], label="Best testing loss, train size", color="#FAAB3655")
plt.plot(x, smooth_line(smoothing_factor, data["test_loss_train_size"]), label="Smoothed", linewidth=3, color="#FAAB36")
plt.plot(x, data["test_loss_test_size"], label="Best testing loss, test size", color="#FD590155")
plt.plot(x, smooth_line(smoothing_factor, data["test_loss_test_size"]), label="Smoothed", linewidth=3, color="#FD5901")

# Axes
plt.ylabel("Loss")
plt.xlabel("Generation")
plt.legend()

################################ ACCURACY ################################

# Get minimum accuracy
config = get_config(args.path)
minim_acc = 1 / len(eval(config.dataset.mnist_digits))

plt.figure()
# First, plot minim accuracy
plt.plot(x, [minim_acc] * len(x), label="Minimum accuracy", color="gray", linestyle="--")

# Plot train accuracy first thinly, then strongly
plt.plot(x, data["test_accuracy_train_size"], label="Train size", color="darkcyan")
plt.plot(
    x,
    smooth_line(smoothing_factor, data["test_accuracy_train_size"]),
    label="Smoothed",
    linewidth=3,
    color="darkcyan",
)
# Same for test accuracy
plt.plot(x, data["test_accuracy_test_size"], label="Test size", color="darkorange")
plt.plot(
    x,
    smooth_line(smoothing_factor, data["test_accuracy_test_size"]),
    label="Smoothed",
    linewidth=3,
    color="darkorange",
)

# Axes
plt.yticks(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10))
plt.ylabel("Accuracy (%) on test data")
plt.xlabel("Generation")
plt.legend()

plt.show()
