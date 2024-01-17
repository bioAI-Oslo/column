import argparse
import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


def smooth_line(filter_size, line):
    half = int((filter_size - 1) / 2)
    new_line = deepcopy(line)

    # The next few lines I were just given by the autocomplete (Codeium?)
    for i in range(len(line)):
        if i < half:
            new_line[i] = np.mean(line[: i + half + 1])
        elif i > len(line) - half - 1:
            new_line[i] = np.mean(line[i - half :])
        else:
            new_line[i] = np.mean(line[i - half : i + half + 1])
    return new_line


parser = argparse.ArgumentParser(
    prog="PlotRunner", description="This program plots runs based on config.", epilog="Text at the bottom of help"
)
parser.add_argument("path", type=str, help="The stored run to plot")

args = parser.parse_args()

with open(args.path + "/plotting_data", "r") as file:
    data = json.load(file)

x = data["x_axis"]
smoothing_factor = int(11 * (len(x) // 50))
print(smoothing_factor)
if smoothing_factor % 2 == 0:
    smoothing_factor += 1

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

# Training raw data
plt.plot(x, data["training_best_loss_history"], label="Best training loss", color="#249EA055")
plt.plot(
    x,
    smooth_line(smoothing_factor, data["training_best_loss_history"]),
    label="Smoothed",
    linewidth=3,
    color="#249EA0",
)
plt.plot(x, data["test_loss_train_size"], label="Best testing loss, train size", color="#FAAB3655")
plt.plot(x, smooth_line(smoothing_factor, data["test_loss_train_size"]), label="Smoothed", linewidth=3, color="#FAAB36")
plt.plot(x, data["test_loss_test_size"], label="Best testing loss, test size", color="#FD590155")
plt.plot(x, smooth_line(smoothing_factor, data["test_loss_test_size"]), label="Smoothed", linewidth=3, color="#FD5901")


plt.ylabel("Loss")
plt.xlabel("Generation")
plt.legend()


plt.figure()
plt.plot(x, data["test_accuracy_train_size"], label="Train size", color="darkcyan")
plt.plot(
    x,
    smooth_line(smoothing_factor, data["test_accuracy_train_size"]),
    label="Smoothed",
    linewidth=3,
    color="darkcyan",
)
plt.plot(x, data["test_accuracy_test_size"], label="Test size", color="darkorange")
# Smoothing accuracy
plt.plot(
    x,
    smooth_line(smoothing_factor, data["test_accuracy_test_size"]),
    label="Smoothed",
    linewidth=3,
    color="darkorange",
)

plt.yticks(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10))
plt.ylabel("Accuracy (%) on test data")
plt.xlabel("Generation")
plt.legend()

plt.show()
