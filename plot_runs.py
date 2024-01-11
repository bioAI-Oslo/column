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


plt.figure()
plt.fill_between(
    x,
    np.array(data["mean_loss_history"]) - np.array(data["std_loss_history"]),
    np.array(data["mean_loss_history"]) + np.array(data["std_loss_history"]),
    color="#2365A355",
)
plt.plot(x, data["mean_loss_history"], label="Mean", color="#2365A3")
plt.plot(x, data["training_best_loss_history"], label="Best training loss", color="red")
plt.plot(x, data["test_loss_train_size"], label="Best testing loss, train size", color="green")
plt.plot(x, data["test_loss_test_size"], label="Best testing loss, test size", color="magenta")
# Smoothing mean
plt.plot(x, smooth_line(11, data["mean_loss_history"]), label="Smoothed mean", linewidth=3, color="black")
plt.plot(
    x, smooth_line(11, data["training_best_loss_history"]), label="training_best_loss_history", linewidth=3, color="red"
)
plt.plot(x, smooth_line(11, data["test_loss_train_size"]), label="test_loss_train_size", linewidth=3, color="green")
plt.plot(x, smooth_line(11, data["test_loss_test_size"]), label="test_loss_test_size", linewidth=3, color="magenta")


plt.ylabel("Loss")
plt.xlabel("Generation")
plt.legend()


plt.figure()
plt.plot(x, data["test_accuracy_train_size"], label="Train size")
plt.plot(x, data["test_accuracy_test_size"], label="Test size")
# Smoothing accuracy
plt.plot(
    x,
    smooth_line(11, data["test_accuracy_train_size"]),
    label="Smoothed accuracy, train size",
    linewidth=3,
    color="black",
)

plt.yticks(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10))
plt.ylabel("Accuracy (%) on test data")
plt.xlabel("Generation")
plt.legend()

plt.show()
