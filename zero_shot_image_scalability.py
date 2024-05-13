from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from main import evaluate_nca, evaluate_nca_batch
from src.data_processing import get_labels, get_MNIST_data
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
    pixel_wise_L2_and_CE,
    scale_loss,
)
from src.moving_nca import MovingNCA
from src.utils import get_config
from tqdm import tqdm


def plot_zero_shot_image_scalability():
    metrics_dict = {
        "losses_train_size": [],
        "losses_size": [],
        "accuracies_train_size": [],
        "accuracies_size": [],
    }

    sizes = np.concatenate((np.arange(28 // 4, 26, 1), np.arange(26, 28 * 2, 4)))  # Starts on 7

    for size in tqdm(sizes):
        resized_x_data = []
        for img, lab in zip(x_data, y_data):
            resized_x_data.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))

        resized_x_data = np.array(resized_x_data)
        if len(resized_x_data.shape) == 3:
            resized_x_data = np.expand_dims(resized_x_data, axis=3)

        moving_nca_kwargs = {
            "size_image": (size, size),
            "num_hidden": config.network.hidden_channels,
            "hidden_neurons": config.network.hidden_neurons,
            "img_channels": config.network.img_channels,
            "iterations": config.network.iterations,
            "position": str(config.network.position),
            "moving": config.network.moving,
            "mnist_digits": mnist_digits,
            "labels": get_labels(data_func, mnist_digits),
        }

        evaluate_kwargs = {
            "flat_weights": winner_flat,
            "training_data": resized_x_data,
            "target_data": y_data,
            "moving_nca_kwargs": moving_nca_kwargs,
            "loss_function": loss_function,
            "predicting_method": predicting_method,
            "verbose": False,
            "visualize": False,
            "return_accuracy": True,
            "pool_training": False,
            "stable": False,
            "return_confusion": False,
        }

        loss, acc = evaluate_nca_batch(**evaluate_kwargs, N_neo=size - 2, M_neo=size - 2)

        metrics_dict["losses_size"].append(loss)
        metrics_dict["accuracies_size"].append(acc)

        loss, acc = evaluate_nca_batch(
            **evaluate_kwargs, N_neo=config.scale.train_n_neo, M_neo=config.scale.train_m_neo
        )

        metrics_dict["losses_train_size"].append(loss)
        metrics_dict["accuracies_train_size"].append(acc)

    cmap = plt.cm.plasma

    plt.figure()
    plt.plot(
        sizes,
        metrics_dict["accuracies_size"],
        label="Test size",
        color=cmap(0.25),
    )
    plt.plot(
        sizes,
        metrics_dict["accuracies_train_size"],
        label="Train size",
        color=cmap(0.75),
    )

    plt.legend()

    plt.yticks(np.arange(0, 1.2, 0.2), np.arange(0, 120, 20))
    plt.ylabel("Accuracy (%) on test-set data")
    plt.xlabel("Size of image")

    plt.figure()
    plt.plot(sizes, metrics_dict["losses_size"], label="Test size", color=cmap(0.0))
    plt.plot(sizes, metrics_dict["losses_train_size"], label="Train size", color=cmap(0.5))

    plt.legend()

    plt.ylabel("Loss on test-set data")
    plt.xlabel("Size of image")

    plt.show()


if __name__ == "__main__":
    sns.set()

    # path = "experiments/tuning_iter_and_sigma/29-1-24_11:41"
    # path = "experiments/tuning_neu_vs_ch_20k/23-2-24_1:35"
    # path = "experiments/8-2-24_11:4"  # Trained on padded data
    path = "experiments/mnist_final/21-4-24_0:34_2"
    config = get_config(path)
    winner_flat = Logger.load_checkpoint(path)

    mnist_digits = eval(config.dataset.mnist_digits)

    data_func = eval(config.dataset.data_func)
    kwargs = {
        "CLASSES": mnist_digits,
        "SAMPLES_PER_CLASS": 100,
        "verbose": False,
        "test": True,
    }

    x_data, y_data = data_func(**kwargs)

    loss_function = eval(config.training.loss)
    predicting_method = eval(config.training.predicting_method)

    plot_zero_shot_image_scalability()
