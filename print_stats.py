"""A file to print the statistics of an experiment folder. It will gather the stats, then print them."""

import argparse
import os

import numpy as np
from common_funcs import get_network
from main import evaluate_nca_batch
from src.data_processing import (
    get_CIFAR_data,
    get_labels,
    get_max_samples_balanced,
    get_MNIST_data,
    get_MNIST_fashion_data,
    get_simple_object,
    get_simple_object_translated,
    get_simple_pattern,
)
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


def get_performance(sub_path, config, test_data_used=False, num_data=None):
    winner_flat = Logger.load_checkpoint(sub_path)

    # Fetch info from config and enable environment for testing
    mnist_digits = eval(config.dataset.mnist_digits)
    predicting_method = eval(config.training.predicting_method)
    loss_function = eval(config.training.loss)

    moving_nca_kwargs = {
        "size_image": (config.dataset.size, config.dataset.size),
        "num_hidden": config.network.hidden_channels,
        "hidden_neurons": config.network.hidden_neurons,
        "iterations": config.network.iterations,
        "position": str(config.network.position),
        "moving": config.network.moving,
        "mnist_digits": mnist_digits,
        "img_channels": config.network.img_channels,
    }

    data_func = eval(config.dataset.data_func)
    kwargs = {
        "CLASSES": mnist_digits,
        "SAMPLES_PER_CLASS": 1,
        "verbose": False,
        "test": test_data_used,
        "colors": True if config.network.img_channels == 3 else False,
    }

    if num_data is None:
        num_data = get_max_samples_balanced(data_func, **kwargs)
    print(num_data)
    kwargs["SAMPLES_PER_CLASS"] = num_data

    training_data, target_data = data_func(**kwargs)

    loss, acc = evaluate_nca_batch(
        winner_flat,
        training_data,
        target_data,
        moving_nca_kwargs,
        loss_function,
        predicting_method,
        verbose=False,
        visualize=False,
        return_accuracy=True,
        N_neo=config.scale.train_n_neo,
        M_neo=config.scale.train_m_neo,
        return_confusion=False,
        pool_training=config.training.pool_training,
        stable=config.training.stable,
    )

    return loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="The path to the experiment folder")
    parser.add_argument("hc", type=int, help="The number of hidden channels")
    parser.add_argument("hn", type=int, help="The number of hidden neurons")
    args = parser.parse_args()
    path = args.path

    num_data_train = 100
    num_data_test = 100

    performances_train = []
    performances_test = []

    for sub_folder in os.listdir(path):
        sub_path = path + "/" + sub_folder
        if os.path.isdir(sub_path):
            config = get_config(sub_path)
            print(config.network.hidden_channels, config.network.hidden_neurons, args.hc, args.hn)
            if config.network.hidden_channels != args.hc or config.network.hidden_neurons != args.hn:
                continue

            loss, acc = get_performance(sub_path, config, test_data_used=False, num_data=num_data_train)
            performances_train.append((loss, acc, sub_path))

            loss, acc = get_performance(sub_path, config, test_data_used=True, num_data=num_data_test)
            performances_test.append((loss, acc, sub_path))

    best_performance = min(performances_train, key=lambda x: x[0])

    print()
    print(performances_train)
    print()
    print("Train data performance:")
    print("Best performance:", best_performance)
    print("Mean accuracy:", np.mean([x[1] for x in performances_train]))
    print("Mean loss:", np.mean([x[0] for x in performances_train]))

    best_performance_test = None
    for x in performances_test:
        if x[2] == best_performance[2]:
            best_performance_test = x

    print()
    print(performances_test)
    print()
    print("Test data performance:")
    print("Best performance:", best_performance_test)
    print("Mean accuracy:", np.mean([x[1] for x in performances_test]))
    print("Mean loss:", np.mean([x[0] for x in performances_test]))
