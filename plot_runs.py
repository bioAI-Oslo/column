import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

with open("../column_findings/plotting_data_pool2", "r") as file:
    data = json.load(file)

x = data["x_axis"]

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
smoothing_num = 21
half = smoothing_num // 2
smooth_mean = np.empty(len(data["mean_loss_history"]))
for i in range(len(data["mean_loss_history"])):
    mini = max(0, i - smoothing_num + 1)
    smooth_mean[i] = np.sum(data["mean_loss_history"][mini : i + 1]) / (i + 1 - mini)

plt.plot(x, smooth_mean, label="Smoothed mean", linewidth=3, color="black")

plt.ylabel("Loss")
plt.xlabel("Generation")
plt.legend()

plt.figure()
plt.plot(x, data["test_accuracy_train_size"], label="Train size")
plt.plot(x, data["test_accuracy_test_size"], label="Test size")

# Smoothing accuracy
smoothing_num = 21
half = smoothing_num // 2
smooth_acc = np.empty(len(data["test_accuracy_train_size"]))
for i in range(len(data["test_accuracy_train_size"])):
    mini = max(0, i - smoothing_num + 1)
    smooth_acc[i] = np.sum(data["test_accuracy_train_size"][mini : i + 1]) / (i + 1 - mini)

plt.plot(x, smooth_acc, label="Smoothed accuracy, train size", linewidth=3, color="black")

plt.yticks(np.arange(0, 1.1, 0.1), np.arange(0, 110, 10))
plt.ylabel("Accuracy (%) on test data")
plt.xlabel("Generation")
plt.legend()

plt.show()
