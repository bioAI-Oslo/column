from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from main import evaluate_nca
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
from src.mnist_processing import get_MNIST_data
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

    even = deepcopy(metrics_dict)
    odd = deepcopy(metrics_dict)

    even_or_odd = [odd, even]

    sizes = np.arange(28 // 4, 28 * 2, 6)  # Starts on 7

    for size in tqdm(sizes):
        for i in range(2):
            resized_x_data = []
            for img, lab in zip(x_data, y_data):
                resized_x_data.append(cv2.resize(img, (size + i, size + i), interpolation=cv2.INTER_AREA))

            resized_x_data = np.array(resized_x_data)

            moving_nca_kwargs = {
                "size_image": (size + i, size + i),
                "num_classes": len(mnist_digits),
                "num_hidden": config.network.hidden_channels,
                "hidden_neurons": config.network.hidden_neurons,
                "iterations": config.network.iterations,
                "position": config.network.position,
                "moving": config.network.moving,
                "mnist_digits": mnist_digits,
            }
            loss, acc = evaluate_nca(
                winner_flat,
                resized_x_data,
                y_data,
                moving_nca_kwargs,
                loss_function,
                predicting_method,
                verbose=False,
                visualize=False,
                return_accuracy=True,
                N_neo=size + i - 2,
                M_neo=size + i - 2,
            )

            even_or_odd[i]["losses_size"].append(loss)
            even_or_odd[i]["accuracies_size"].append(acc)

            loss, acc = evaluate_nca(
                winner_flat,
                resized_x_data,
                y_data,
                moving_nca_kwargs,
                loss_function,
                predicting_method,
                verbose=False,
                visualize=False,
                return_accuracy=True,
                N_neo=size + i - 2 if size + i < 17 else 15,
                M_neo=size + i - 2 if size + i < 17 else 15,
            )

            """if i == 0 and size == 7:
                img_raw = resized_x_data[0].reshape(size, size, 1)
                network = MovingNCA.get_instance_with(winner_flat, size_neo=(size - 2, size - 2), **moving_nca_kwargs)
                network.classify(img_raw, visualize=True)"""

            even_or_odd[i]["losses_train_size"].append(loss)
            even_or_odd[i]["accuracies_train_size"].append(acc)

    cmap = plt.cm.plasma

    for i in range(2):
        add_on = " Even" if i == 1 else " Odd"
        plt.plot(sizes, even_or_odd[i]["losses_size"], label="Loss size=test size" + add_on, color=cmap(0.5 * i))
        plt.plot(
            sizes,
            even_or_odd[i]["accuracies_size"],
            label="Accuracy size=test size" + add_on,
            color=cmap(0.5 * i),
        )

        plt.plot(
            sizes,
            even_or_odd[i]["losses_train_size"],
            label="Loss size=train size" + add_on,
            color=cmap(0.5 * i + 3.5 / 5),
        )
        plt.plot(
            sizes,
            even_or_odd[i]["accuracies_train_size"],
            label="Accuracy size=train size" + add_on,
            color=cmap(0.5 * i + 3.5 / 5),
        )

    plt.yticks(np.arange(0, 1.2, 0.2), np.round(np.arange(0, 1.2, 0.2), 1))
    plt.xlabel("Size of image")
    plt.legend()

    plt.show()


def plot_zero_shot_image_scalability_on_original_size():
    metrics_dict = {
        "losses_train_size": [],
        "losses_test_size": [],
        "accuracies_train_size": [],
        "accuracies_test_size": [],
    }

    sizes = np.arange(28 // 4, 28)  # Starts on 7

    for size in tqdm(sizes):
        resized_x_data = []
        for img, lab in zip(x_data, y_data):
            resized_x_data.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))

        resized_x_data_28x28 = []
        for img in resized_x_data:
            resized_x_data_28x28.append(cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA))

        resized_x_data = np.array(resized_x_data_28x28)

        moving_nca_kwargs = {
            "size_image": (28, 28),
            "num_classes": len(mnist_digits),
            "num_hidden": config.network.hidden_channels,
            "hidden_neurons": config.network.hidden_neurons,
            "iterations": config.network.iterations,
            "position": config.network.position,
            "moving": config.network.moving,
            "mnist_digits": mnist_digits,
        }
        loss, acc = evaluate_nca(
            winner_flat,
            resized_x_data,
            y_data,
            moving_nca_kwargs,
            loss_function,
            predicting_method,
            verbose=False,
            visualize=False,
            return_accuracy=True,
            N_neo=config.scale.train_n_neo,
            M_neo=config.scale.train_m_neo,
        )

        metrics_dict["losses_train_size"].append(loss)
        metrics_dict["accuracies_train_size"].append(acc)

        loss, acc = evaluate_nca(
            winner_flat,
            resized_x_data,
            y_data,
            moving_nca_kwargs,
            loss_function,
            predicting_method,
            verbose=False,
            visualize=False,
            return_accuracy=True,
            N_neo=config.scale.test_n_neo,
            M_neo=config.scale.test_m_neo,
        )

        metrics_dict["losses_test_size"].append(loss)
        metrics_dict["accuracies_test_size"].append(acc)

        """if size == 7:
            img_raw = resized_x_data[0].reshape(28, 28, 1)
            network = MovingNCA.get_instance_with(winner_flat, size_neo=(26, 26), **moving_nca_kwargs)
            network.reset()
            network.classify(img_raw, visualize=True)"""

    cmap = plt.cm.plasma

    plt.plot(sizes, metrics_dict["losses_test_size"], label="Loss size=test size", color=cmap(0.0))
    plt.plot(
        sizes,
        metrics_dict["accuracies_test_size"],
        label="Accuracy size=test size",
        color=cmap(0.25),
    )

    plt.plot(
        sizes,
        metrics_dict["losses_train_size"],
        label="Loss size=train size",
        color=cmap(0.5),
    )
    plt.plot(
        sizes,
        metrics_dict["accuracies_train_size"],
        label="Accuracy size=train size",
        color=cmap(0.75),
    )

    plt.yticks(np.arange(0, 1.2, 0.2), np.round(np.arange(0, 1.2, 0.2), 1))
    plt.xlabel("Size of image before resizing")
    plt.legend()

    plt.show()


def plot_zero_shot_image_scalability_on_padded_image():
    metrics_dict = {
        "losses_train_size": [],
        "losses_test_size": [],
        "accuracies_train_size": [],
        "accuracies_test_size": [],
    }

    sizes = np.arange(28 // 4, 56)  # Starts on 7

    for size in tqdm(sizes):
        resized_x_data = []
        for img, lab in zip(x_data, y_data):
            resized_x_data.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))

        # Pad with zeros so that the network will accept it
        diff = 56 - size
        diff_x, diff_y = int(np.floor(diff / 2)), int(np.ceil(diff / 2))

        resized_x_data = np.pad(resized_x_data, ((0, 0), (diff_x, diff_y), (diff_x, diff_y)), "constant")

        moving_nca_kwargs = {
            "size_image": (56, 56),
            "num_classes": len(mnist_digits),
            "num_hidden": config.network.hidden_channels,
            "hidden_neurons": config.network.hidden_neurons,
            "iterations": config.network.iterations,
            "position": config.network.position,
            "moving": config.network.moving,
            "mnist_digits": mnist_digits,
        }
        loss, acc = evaluate_nca(
            winner_flat,
            resized_x_data,
            y_data,
            moving_nca_kwargs,
            loss_function,
            predicting_method,
            verbose=False,
            visualize=False,
            return_accuracy=True,
            N_neo=config.scale.train_n_neo,
            M_neo=config.scale.train_m_neo,
        )

        metrics_dict["losses_train_size"].append(loss)
        metrics_dict["accuracies_train_size"].append(acc)

        loss, acc = evaluate_nca(
            winner_flat,
            resized_x_data,
            y_data,
            moving_nca_kwargs,
            loss_function,
            predicting_method,
            verbose=False,
            visualize=False,
            return_accuracy=True,
            N_neo=config.scale.test_n_neo,
            M_neo=config.scale.test_m_neo,
        )

        metrics_dict["losses_test_size"].append(loss)
        metrics_dict["accuracies_test_size"].append(acc)

        """if size in [7, 15, 28, 55]:
            img_raw = resized_x_data[0].reshape(56, 56, 1)
            network = MovingNCA.get_instance_with(winner_flat, size_neo=(26, 26), **moving_nca_kwargs)
            network.reset()
            network.classify(img_raw, visualize=True)"""

    cmap = plt.cm.plasma

    plt.plot(sizes, metrics_dict["losses_test_size"], label="Loss size=test size", color=cmap(0.0))
    plt.plot(
        sizes,
        metrics_dict["accuracies_test_size"],
        label="Accuracy size=test size",
        color=cmap(0.25),
    )

    plt.plot(
        sizes,
        metrics_dict["losses_train_size"],
        label="Loss size=train size",
        color=cmap(0.5),
    )
    plt.plot(
        sizes,
        metrics_dict["accuracies_train_size"],
        label="Accuracy size=train size",
        color=cmap(0.75),
    )

    plt.yticks(np.arange(0, 1.2, 0.2), np.round(np.arange(0, 1.2, 0.2), 1))
    plt.xlabel("Size of image before padding")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    sns.set()

    # path = "experiments/tuning_iter_and_sigma/29-1-24_11:41"
    # path = "experiments/tuning_neu_vs_ch_20k/23-2-24_1:35"
    # path = "experiments/8-2-24_11:4"  # Trained on padded data
    path = "experiments/current_pos/3-3-24_15:58_2"
    config = get_config(path)
    winner_flat = Logger.load_checkpoint(path)

    mnist_digits = eval(config.dataset.mnist_digits)

    data_func = get_MNIST_data
    kwargs = {
        "MNIST_DIGITS": mnist_digits,
        "SAMPLES_PER_DIGIT": 100,
        "verbose": False,
        "test": True,
    }

    x_data, y_data = data_func(**kwargs)

    loss_function = eval(config.training.loss)
    predicting_method = eval(config.training.predicting_method)

    plot_zero_shot_image_scalability()
