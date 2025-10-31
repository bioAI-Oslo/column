import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from localconfig import config
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# "#7597FF" "#6A52DC" "#DC267F" "#FFB000"

# #7597ff #6c63e4 #a33bac #e5495e #ffb000

colors = {
    "Moving ANCA": "#DC267F",
    "Moving ANCA 2C 40N": "#DC267F",
    "Moving ANCA 5C 20N": "#7597FF",
    "Non-Moving ANCA": "#FFB000",
    "Random ANCA": "#7597FF",
    "Selective ANCA": "#a33bac",
    "CNN": "#00000000",  # "#f48725", # Black for appendix, orange for main results. It was hard to see the difference between the CNN and the non-moving otherwise
    "ViT": "#6c63e4",
}  # Modified IBM design library palette


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


def plot_robustness(folders, line=True, silencing_method="square"):
    sns.set()

    file_name = f"/{silencing_method}_silencing_robustness.json"

    for name in folders.keys():

        folder_robustness = json.load(open(folders[name] + file_name))

        scores = []

        for key in folder_robustness.keys():
            if key == "test_sizes":
                continue

            scores.append(folder_robustness[key])

        mean = np.mean(scores, axis=0)
        print(name, mean)

        plt.plot(folder_robustness["test_sizes"], mean, label=name, linewidth=3, alpha=0.8, color=colors[name])
        plt.fill_between(
            folder_robustness["test_sizes"],
            mean - np.std(scores, axis=0),
            mean + np.std(scores, axis=0),
            alpha=0.1,
            color=colors[name],
        )

        if line:
            for key in folder_robustness.keys():
                if key == "test_sizes":
                    continue

                plt.plot(
                    folder_robustness["test_sizes"],
                    folder_robustness[key],
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                    color=colors[name],
                )
                scores.append(folder_robustness[key])
        else:
            xs = []
            ys = []
            for key in folder_robustness.keys():
                if key == "test_sizes":
                    continue

                scores.append(folder_robustness[key])

                for x, y in zip(folder_robustness["test_sizes"], folder_robustness[key]):
                    xs.append(x)
                    ys.append(y)

            plt.scatter(xs, ys, alpha=0.5, color=colors[name])

    plt.xticks(np.linspace(0.0, 1.0, 11), np.round(np.linspace(0, 100, 11)).astype(int))
    plt.yticks(np.linspace(0.0, 1.0, 11), np.round(np.linspace(0, 100, 11)).astype(int))
    plt.xlabel("Number of cells silenced (%)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


def plot_scalability(folders):
    sns.set()

    for name, folder in folders.items():
        folder_scalability = json.load(open(folder + "/scalabilities.json"))

        sizes = {}

        for key in folder_scalability.keys():
            if key == "test_sizes":
                continue

            config.read(key + "/config")

            if config.scale.train_n_neo not in sizes:
                sizes[config.scale.train_n_neo] = []

            sizes[config.scale.train_n_neo].append(folder_scalability[key])

        # IBM design library palette
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["#648fff", "#785ef0", "#DC267F", "#f4701e", "#FFB000"]
        )

        keys = list(sizes.keys())
        keys.sort()

        plt.figure()
        plt.title(name)

        for i, key in enumerate(keys):
            x_axis = folder_scalability["test_sizes"]
            mean = np.mean(sizes[key], axis=0)
            std = np.std(sizes[key], axis=0)
            plt.plot(x_axis, mean, label=key, color=cmap(i / 5))
            plt.fill_between(x_axis, mean - std, mean + std, color=cmap(i / 5), alpha=0.3)
        plt.xlabel("NCA size (^2)")
        plt.ylabel("Accuracy (%)")
        plt.yticks(np.linspace(0.0, 1.0, 11), ["0", "", "20", "", "40", "", "60", "", "80", "", "100"])
        plt.ylim([0, 1])
        plt.xticks(range(1, 27), [str(size) if size in [1, 3, 5, 7, 10, 15, 20, 25, 26] else "" for size in x_axis])

        plt.legend(title="Substrate sizes:", ncol=2)
    plt.show()


def plot_robustness_vs_scalability(folders):

    sns.set()

    x_data = []
    y_data = []

    for name in folders.keys():
        robustness_info = json.load(open(folders[name] + "/square_silencing_robustness.json"))
        scalabilities_info = json.load(open(folders[name] + "/scalabilities.json"))

        scores_robustnesses = get_score_robustness(robustness_info)
        scores_scalabilities = get_score_scalability(scalabilities_info)
        sizes = get_sizes(scalabilities_info)

        xs, ys, zs = [], [], []
        for folder in scores_robustnesses.keys():
            print(folder, scores_robustnesses[folder], scores_scalabilities[folder])
            ys.append(scores_robustnesses[folder])
            zs.append(sizes[folder])
            xs.append(scores_scalabilities[folder])

        for x, y in zip(xs, ys):
            x_data.append(x)
            y_data.append(y)

        plt.scatter(
            xs,
            ys,
            s=50 + 100 * (25 < np.array(zs)) + 100 * (100 < np.array(zs)) + 100 * (225 < np.array(zs)),
            alpha=1 - (np.sqrt(np.array(zs))) / (35),
            color=colors[name],
            label=name,
        )

    # Compute the polynomial trend using linear regression
    x_data = np.array(x_data)
    y_data = np.array(y_data).reshape(-1, 1)
    design_matrix = np.array([np.ones(len(x_data)), x_data, x_data**2]).T
    beta = np.linalg.inv(design_matrix.T @ design_matrix) @ design_matrix.T @ y_data

    plot_x = np.unique(x_data)
    plot_y = beta[0] + beta[1] * plot_x + beta[2] * plot_x**2

    predicted_y = beta[0] + beta[1] * x_data + beta[2] * x_data**2

    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print(ss_res, ss_tot)
        return r2

    r2 = r2_score(y_data.flatten(), np.array(predicted_y).flatten())
    print(f"R2 score: {r2}")

    plt.plot(
        plot_x,
        plot_y,
        color="black",
        label="Trendline",
    )

    plt.xlabel("Scalability")
    plt.ylabel("Robustness")

    plt.legend()

    plt.show()


def plot_size_vs_scalability(folders):

    sns.set()

    for name in folders.keys():
        scalabilities_info = json.load(open(folders[name] + "/scalabilities.json"))

        scores_scalabilities = get_score_scalability(scalabilities_info)
        sizes = get_sizes(scalabilities_info)

        xs, ys = [], []
        for folder in scores_scalabilities.keys():
            ys.append(scores_scalabilities[folder])
            xs.append(np.sqrt(sizes[folder]).astype(int))

        plt.scatter(
            xs,
            ys,
            color=colors[name],
            label=name,
        )

    plt.xlabel("NCA size (^2)")
    plt.ylabel("Scalability")

    plt.legend()

    plt.show()


"""plot_robustness(
    folders={
        "Moving ANCA 2C 40N": "experiments/mnist_final15_2:40",
        "Moving ANCA 5C 20N": "experiments/mnist_final15",
        "CNN": "experiments/cnn/mnist5",
        "ViT": "experiments/vit/mnist5",
        # "Moving ANCA": "experiments/mnist3_robust_26",
        # "Non-Moving ANCA": "experiments/mnist3_robust_nonmoving_26",
        # "CNN": "experiments/cnn/mnist3",
        # "ViT": "experiments/vit/mnist3",
        # "Random ANCA": "experiments/mnist3_robust_random_26",
        # "Selective ANCA": "experiments/mnist3_robust_selective_aggregated_26",
    },
    silencing_method="square",
)"""

"""plot_robustness_vs_scalability(
    folders={
        # "Moving ANCA 2C 40N": "experiments/mnist_final15_2:40",
        # "Moving ANCA 5C 20N": "experiments/mnist_final15",
        "Moving ANCA": "experiments/mnist3_robust",
        "Non-Moving ANCA": "experiments/mnist3_robust_nonmoving",
        "Random ANCA": "experiments/mnist3_robust_random",
        "Selective ANCA": "experiments/mnist3_robust_selective_aggregated",
    }
)"""

"""plot_size_vs_scalability(
    folders={
        # "Moving ANCA 2C 40N": "experiments/mnist_final15_2:40",
        # "Moving ANCA 5C 20N": "experiments/mnist_final15",
        "Moving ANCA": "experiments/mnist3_robust",
        "Non-Moving ANCA": "experiments/mnist3_robust_nonmoving",
        "Random ANCA": "experiments/mnist3_robust_random",
        "Selective ANCA": "experiments/mnist3_robust_selective_aggregated",
    }
)"""

plot_scalability(
    folders={
        "Moving ANCA": "experiments/neo_size_experiment",
        "Non-Moving ANCA": "experiments/neo_size_experiment_nonmoving",
        "Random ANCA": "experiments/neo_size_experiment_random",
        "Selective ANCA": "experiments/neo_size_experiment_selective_aggregated",
    }
)
