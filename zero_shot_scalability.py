import os

import matplotlib.pyplot as plt
import numpy as np
from localconfig import config
from main import evaluate_nca
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_L2,
    scale_loss,
)
from src.mnist_processing import get_MNIST_data
from tqdm import tqdm


def get_config(path):
    config.read(path + "/config")

    return config


DATA_NUM = 5
to_test = np.arange(1, 26 + 1, 1)

mean_acc = []
mean_loss = []

path = "experiments/tuning"
for numebr, sub_folder in enumerate(os.listdir(path)):
    if numebr >= 4:
        break
    sub_path = path + "/" + sub_folder
    if os.path.isdir(sub_path):
        winner_flat = Logger.load_checkpoint(sub_path)

        config = get_config(sub_path)

        mnist_digits = eval(config.dataset.mnist_digits)

        moving_nca_kwargs = {
            "size_image": (28, 28),
            "num_classes": len(mnist_digits),
            "num_hidden": config.network.hidden_channels,
            "iterations": config.network.iterations,
            "current_pos": config.network.current_pos,
            "moving": config.network.moving,
            "mnist_digits": mnist_digits,
        }

        data_func = get_MNIST_data
        kwargs = {
            "MNIST_DIGITS": mnist_digits,
            "SAMPLES_PER_DIGIT": 10,
            "verbose": False,
            "test": True,
        }

        loss_function = eval(config.training.loss)
        predicting_method = eval(config.training.predicting_method)

        training_data, target_data = [], []
        for _ in range(DATA_NUM):
            x, y = data_func(**kwargs)
            training_data.append(x)
            target_data.append(y)

        result_loss = []
        result_acc = []
        for test_size in tqdm(to_test):
            loss_sum = 0
            acc_sum = 0
            for i in range(DATA_NUM):
                loss, acc = evaluate_nca(
                    winner_flat,
                    training_data[i],
                    target_data[i],
                    moving_nca_kwargs,
                    loss_function,
                    predicting_method,
                    verbose=False,
                    visualize=False,
                    return_accuracy=True,
                    N_neo=test_size,
                    M_neo=test_size,
                )
                loss_sum += loss
                acc_sum += acc
            result_loss.append(loss_sum / DATA_NUM)
            result_acc.append(float(acc_sum) / float(DATA_NUM))

        mean_acc.append(result_acc)
        mean_loss.append(result_loss)

cmap = plt.cm.plasma

plt.xticks(to_test, to_test)
plt.plot(to_test, np.mean(mean_loss, axis=0), label="Loss", color=cmap(0.2))
plt.plot(to_test, np.mean(mean_acc, axis=0), label="Accuracy", color=cmap(0.8))
plt.fill_between(
    to_test,
    np.mean(mean_loss, axis=0) - np.std(mean_loss, axis=0),
    np.mean(mean_loss, axis=0) + np.std(mean_loss, axis=0),
    color=cmap(0.2),
    alpha=0.5,
)
plt.fill_between(
    to_test,
    np.mean(mean_acc, axis=0) - np.std(mean_acc, axis=0),
    np.mean(mean_acc, axis=0) + np.std(mean_acc, axis=0),
    color=cmap(0.8),
    alpha=0.5,
)
plt.xlabel("NCA size (^2)")
plt.legend()
plt.show()
