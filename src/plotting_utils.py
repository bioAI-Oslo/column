import json
from copy import deepcopy

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


def get_smoothing_factor(length_list):
    smoothing_factor = int(11 * (length_list // 50))
    if smoothing_factor % 2 == 0:
        smoothing_factor += 1

    return smoothing_factor


def get_plotting_data(path):
    with open(path + "/plotting_data", "r") as file:
        data = json.load(file)
        file.close()

    return data


def get_plotting_ticks(image):
    """I'm altering xticks and yticks for when showing the datsets because I want to show the most important info
    which to me is the middle of the image and the image size"""
    N, M = len(image), len(image[0])
    xticks = [-0.5, (M // 2) - 0.5, M - 0.5], [0, M // 2, M]
    yticks = [-0.5, (N // 2) - 0.5, N - 0.5], [0, N // 2, N]

    return xticks, yticks
