import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from localconfig import config
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_score_robustness(folder_dict):
    scores = {}
    for key in folder_dict.keys():
        if key == "test_sizes":
            continue

        score_init = folder_dict[key][0]

        score_rest = folder_dict[key][1:-1]
        score_rest = [min(score_rest_i, score_init) for score_rest_i in score_rest]

        absolute_score = (np.array(score_rest) - 1 / 3) / (score_init - 1 / 3)

        score = np.sum(absolute_score)

        scores[key] = score / len(score_rest)

    return scores


def get_score_scalability(folder_dict):
    scores = {}
    for key in folder_dict.keys():
        if key == "test_sizes":
            continue

        config.read(key + "/config")

        train_pos = np.argmin(abs(np.array(folder_dict["test_sizes"]) - int(config.scale.train_n_neo)))

        init_score = folder_dict[key][train_pos]

        rest = folder_dict[key][:train_pos] + folder_dict[key][train_pos + 1 :]
        rest = [min(score_rest_i, init_score) for score_rest_i in rest]

        absolute_score = (np.array(rest) - 1 / 3) / (init_score - 1 / 3)

        score = np.sum(absolute_score)

        score = score / len(rest)

        scores[key] = score

    return scores


def get_sizes(folder_dict):
    sizes = {}
    for key in folder_dict.keys():
        if key == "test_sizes":
            continue

        config.read(key + "/config")

        sizes[key] = int(config.scale.train_n_neo * config.scale.train_m_neo)

    return sizes


def plot_robustness():
    sns.set()

    folder_moving = "experiments/mnist3_robust_26"
    folder_nonmoving = "experiments/mnist3_robust_nonmoving_26"
    folder_cnn = "experiments/cnn/mnist3"

    folder_robustness_moving = json.load(open(folder_moving + "/square_silencing_robustness.json"))
    folder_robustness_nonmoving = json.load(open(folder_nonmoving + "/square_silencing_robustness.json"))
    folder_robustness_cnn = json.load(open(folder_cnn + "/square_silencing_robustness.json"))

    colors = {"moving": "tab:blue", "nonmoving": "tab:orange", "cnn": "tab:green"}

    for name, folder in zip(
        ["moving", "nonmoving", "cnn"], [folder_robustness_moving, folder_robustness_nonmoving, folder_robustness_cnn]
    ):
        scores = []

        xs = []
        ys = []

        for key in folder.keys():
            if key == "test_sizes":
                continue

            if "moving" in name:
                config.read(key + "/config")
                if config.scale.train_n_neo != 26:
                    continue

            scores.append(folder[key])

            for x, y in zip(folder["test_sizes"], folder[key]):
                xs.append(x)
                ys.append(y)

        mean = np.mean(scores, axis=0)
        print(name, mean)

        plt.scatter(xs, ys, alpha=0.5, color=colors[name])

        plt.plot(folder["test_sizes"], mean, label=name, linewidth=3, alpha=0.8, color=colors[name])
        plt.fill_between(
            folder["test_sizes"],
            mean - np.std(scores, axis=0),
            mean + np.std(scores, axis=0),
            alpha=0.2,
            color=colors[name],
        )

    plt.xticks(np.linspace(0.0, 1.0, 11), np.round(np.linspace(0, 100, 11)).astype(int))
    plt.yticks(np.linspace(0.0, 1.0, 11), np.round(np.linspace(0, 100, 11)).astype(int))
    plt.xlabel("Number of cells silenced (%)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


def plot_scalability():
    folder_moving = "experiments/mnist3_robust"
    folder_nonmoving = "experiments/mnist3_robust_nonmoving"

    folder_robustness_moving = json.load(open(folder_moving + "/square_silencing_robustness.json"))
    folder_robustness_nonmoving = json.load(open(folder_nonmoving + "/square_silencing_robustness.json"))

    folder_scalability_moving = json.load(open(folder_moving + "/scalabilities.json"))
    folder_scalability_nonmoving = json.load(open(folder_nonmoving + "/scalabilities.json"))

    plt.figure()

    sizes = {}

    for key in folder_scalability_moving.keys():
        if key == "test_sizes":
            continue

        config.read(key + "/config")

        if config.scale.train_n_neo not in sizes:
            sizes[config.scale.train_n_neo] = []

        sizes[config.scale.train_n_neo].append(folder_scalability_moving[key])

    for key in sizes:
        plt.plot(np.mean(sizes[key], axis=0), label=key)

    plt.show()


def plot_robustness_vs_scalability():
    folder_moving = "experiments/mnist3_robust"
    folder_nonmoving = "experiments/mnist3_robust_nonmoving"

    folder_robustness_moving = json.load(open(folder_moving + "/square_silencing_robustness.json"))
    folder_robustness_nonmoving = json.load(open(folder_nonmoving + "/square_silencing_robustness.json"))

    folder_scalability_moving = json.load(open(folder_moving + "/scalabilities.json"))
    folder_scalability_nonmoving = json.load(open(folder_nonmoving + "/scalabilities.json"))

    ### score begin

    scores_robustness_moving = get_score_robustness(folder_robustness_moving)
    scores_robustness_nonmoving = get_score_robustness(folder_robustness_nonmoving)

    scores_scalability_moving = get_score_scalability(folder_scalability_moving)
    scores_scalability_nonmoving = get_score_scalability(folder_scalability_nonmoving)

    sizes_moving = get_sizes(folder_scalability_moving)
    sizes_nonmoving = get_sizes(folder_scalability_nonmoving)

    xs_moving, ys_moving, zs_moving = [], [], []
    for key in scores_robustness_moving.keys():
        print(key, scores_robustness_moving[key], scores_scalability_moving[key])
        ys_moving.append(scores_robustness_moving[key])
        zs_moving.append(sizes_moving[key])
        xs_moving.append(scores_scalability_moving[key])

    sns.set()

    plt.scatter(
        xs_moving,
        ys_moving,
        s=50 + 100 * (25 < np.array(zs_moving)) + 100 * (100 < np.array(zs_moving)) + 100 * (225 < np.array(zs_moving)),
        alpha=1 - (np.sqrt(np.array(zs_moving))) / (35),
        label="Moving",
    )

    xs_nonmoving, ys_nonmoving, zs_nonmoving = [], [], []
    for key in scores_robustness_nonmoving.keys():
        print(key, scores_robustness_nonmoving[key], scores_scalability_nonmoving[key])
        ys_nonmoving.append(scores_robustness_nonmoving[key])
        zs_nonmoving.append(sizes_nonmoving[key])
        xs_nonmoving.append(scores_scalability_nonmoving[key])

    plt.scatter(
        xs_nonmoving,
        ys_nonmoving,
        s=(np.array(zs_nonmoving)),
        alpha=1 - (np.sqrt(np.array(zs_nonmoving))) / (35),
        label="Non-Moving",
    )

    """X = np.zeros((len(xs_moving) + len(xs_nonmoving), 3))
    X[:, 0] = 1
    X[0 : len(xs_moving), 1] = np.array(xs_moving)
    X[len(xs_moving) :, 1] = np.array(xs_nonmoving)
    X[0 : len(xs_moving), 2] = (np.array(xs_moving)) ** 2
    X[len(xs_moving) :, 2] = (np.array(xs_nonmoving)) ** 2
    

    Y = np.array(ys_moving + ys_nonmoving)

    beta = np.linalg.inv(X.T.dot(X) + 0.0001 * np.eye(X.shape[1])).dot(X.T).dot(Y)
    print(f"robustness = {beta[0]} + {beta[1]}*scalability + {beta[2]}*exp(scalability)")

    X_sorted = X[np.argsort(X[:, 1])]
    Y_pred = X_sorted.dot(beta)
    plt.plot(
        X_sorted[:, 1],
        Y_pred,
        "r--",
        label=f"r = {np.round(beta[0], 2)} {np.round(beta[1], 2)} $s$ + {np.round(beta[2], 2)}" + r"$e^s$",
    )"""

    plt.xlabel("Scalability")
    plt.ylabel("Robustness")

    plt.legend()

    plt.show()


plot_robustness()

# plot_robustness_vs_scalability()
