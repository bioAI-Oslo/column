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


def get_plotting_data(path):
    with open(path + "/plotting_data", "r") as file:
        data = json.load(file)
        file.close()

    return data
