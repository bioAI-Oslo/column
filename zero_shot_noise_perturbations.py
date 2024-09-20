import json
import os
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from localconfig import config
from main import evaluate_nca, get_from_config
from src.data_processing import get_MNIST_data, get_simple_object
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
    pixel_wise_L2_and_CE,
)
from src.moving_nca_no_tf import MovingNCA
from src.utils import get_config
from tqdm import tqdm


def add_noise(data, noise_level):
    noise = np.random.normal(0, noise_level, data.shape)

    new_data = data + noise

    # Normalize between 0 and 1
    # new_data = (new_data - new_data.min()) / (new_data.max() - new_data.min())

    # Clamp between 0 and 1
    new_data = np.clip(new_data, 0, 1)

    return new_data


def get_scores(evaluate_nca_kwargs, to_test, test_data):
    scores = []

    for i, level in tqdm(enumerate(to_test)):
        noisy_data = add_noise(test_data, level)

        evaluate_nca_kwargs["training_data"] = noisy_data

        loss, acc = evaluate_nca(**evaluate_nca_kwargs)

        scores.append(acc)

    return scores


def build_evaluate_kwargs(config, path):
    moving_nca_kwargs, loss_function, predicting_method, data_func, kwargs = get_from_config(config)

    kwargs["SAMPLES_PER_CLASS"] = NUM_DATA
    kwargs["test"] = True
    test_data, target_data = data_func(**kwargs)

    winner_flat = Logger.load_checkpoint(path)

    evaluate_kwargs = {
        "flat_weights": winner_flat,
        "training_data": None,
        "target_data": target_data,
        "moving_nca_kwargs": moving_nca_kwargs,
        "loss_function": loss_function,
        "predicting_method": predicting_method,
        "verbose": False,
        "visualize": False,
        "N_neo": config.scale.train_n_neo,
        "M_neo": config.scale.train_m_neo,
        "return_accuracy": True,
        "pool_training": config.training.pool_training,
        "stable": config.training.stable,
        "return_confusion": False,
    }

    return evaluate_kwargs, test_data


def get_scores_from_path(path, to_test):
    config = get_config(path)

    evaluate_kwargs, test_data = build_evaluate_kwargs(config, path)

    scores = get_scores(evaluate_kwargs, to_test, test_data)

    return scores


def get_scores_from_paths(paths, to_test):
    scores = {}
    for path in paths:
        scores[path] = get_scores_from_path(path, to_test)

    return scores


def visualize_noise(to_test):
    data_x, data_y = get_MNIST_data(CLASSES=[0], SAMPLES_PER_CLASS=1, test=True)

    # Add noise
    plt.figure()
    for i, level in enumerate(to_test):
        noisy = add_noise(data_x[0], level)

        plt.subplot(3, 4, i + 1)
        plt.title(r"$N(0,\sigma), \sigma $ = " + str(np.round(level, 1)))
        plt.imshow(noisy, cmap="gray", vmin=0, vmax=1)
    plt.show()


def get_paths(path):
    paths = []
    for sub_folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, sub_folder)):
            config = get_config(os.path.join(path, sub_folder))
            if config.scale.train_n_neo < 27:
                paths.append(os.path.join(path, sub_folder))

    return paths


def record_score(path, to_test):
    scores = get_scores_from_paths(get_paths(path), to_test)

    scores["test_sizes"] = to_test.tolist()
    json.dump(scores, open(path + "/image_noise_robustness.json", "w"))


def plot_scores_comparison():
    sns.set()

    folder_moving = "experiments/mnist3_robust_26"
    folder_nonmoving = "experiments/mnist3_robust_nonmoving_26"
    folder_cnn = "experiments/cnn/mnist3"

    folder_robustness_moving = json.load(open(folder_moving + "/image_noise_robustness.json"))
    folder_robustness_nonmoving = json.load(open(folder_nonmoving + "/image_noise_robustness.json"))
    folder_robustness_cnn = json.load(open(folder_cnn + "/image_noise_robustness.json"))

    colors = {"moving": "tab:blue", "nonmoving": "tab:orange", "cnn": "tab:green"}

    test_sizes = None

    for name, folder in zip(
        ["moving", "nonmoving", "cnn"], [folder_robustness_moving, folder_robustness_nonmoving, folder_robustness_cnn]
    ):
        scores = []

        xs = []
        ys = []

        for key in folder.keys():
            if key == "test_sizes":
                test_sizes = folder[key]
                continue

            scores.append(folder[key])

            for x, y in zip(folder["test_sizes"], folder[key]):
                xs.append(x)
                ys.append(y)

        mean = np.mean(scores, axis=0)
        print(name, mean)

        # plt.scatter(xs, ys, alpha=0.5, color=colors[name])

        plt.plot(folder["test_sizes"], mean, label=name, linewidth=3, alpha=0.8, color=colors[name])
        plt.fill_between(
            folder["test_sizes"],
            mean - np.std(scores, axis=0),
            mean + np.std(scores, axis=0),
            alpha=0.2,
            color=colors[name],
        )

        for score in scores:
            plt.plot(folder["test_sizes"], score, alpha=0.3, color=colors[name])

    plt.xticks(test_sizes)
    plt.yticks(np.linspace(0.0, 1.0, 11), np.round(np.linspace(0, 100, 11)).astype(int))
    plt.xlabel(r"Noise added, $\sigma$ in $N(0, \sigma)$")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # sns.set()

    path = "experiments/mnist3_robust_nonmoving_26"
    to_test = np.linspace(0, 1.0, 11)
    NUM_DATA = 1

    # record_score(path, to_test)

    plot_scores_comparison()
