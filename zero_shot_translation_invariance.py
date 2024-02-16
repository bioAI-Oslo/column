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
from src.utils import get_config, translate
from tqdm import tqdm

path = "experiments/8-2-24_11:4"
config = get_config(path)
winner_flat = Logger.load_checkpoint(path)

loss_function = eval(config.training.loss)
predicting_method = eval(config.training.predicting_method)

mnist_digits = eval(config.dataset.mnist_digits)

data_func = get_MNIST_data
kwargs = {
    "MNIST_DIGITS": mnist_digits,
    "SAMPLES_PER_DIGIT": 10,
    "verbose": False,
    "test": True,
}

x_data, y_data = data_func(**kwargs)

new_length = 70

translated_x_data = translate(x_data, (new_length, new_length))

moving_nca_kwargs = {
    "size_image": (new_length, new_length),
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
    translated_x_data,
    y_data,
    moving_nca_kwargs,
    loss_function,
    predicting_method,
    verbose=False,
    visualize=False,
    return_accuracy=True,
    N_neo=15,
    M_neo=15,
)

print(loss, acc)
