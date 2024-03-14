""" This file copies selected functions from
https://github.com/sidneyp/neural-cellular-robot-substrate/blob/main/src/utils.py
For myself: If this ends up in a published repo, you need better reference and
liscence """

import json
import pickle
import random
import time

import numpy as np
import tensorflow as tf
from localconfig import config


# Sidney's function
def get_weights_info(weights):
    weight_shape_list = []
    for layer in weights:
        weight_shape_list.append(tf.shape(layer))

    weight_amount_list = [tf.reduce_prod(w_shape) for w_shape in weight_shape_list]
    weight_amount = tf.reduce_sum(weight_amount_list)

    return weight_shape_list, weight_amount_list, weight_amount


# Sidney's function
def get_model_weights(flat_weights, weight_amount_list, weight_shape_list):
    split_weight = tf.split(flat_weights, weight_amount_list)
    return [tf.reshape(split_weight[i], weight_shape_list[i]) for i in tf.range(len(weight_shape_list))]


# Sidney's function
def get_flat_weights(weights):
    flat_weights = []
    for layer in weights:
        flat_weights.extend(list(layer.numpy().flatten()))

    return flat_weights


# Mia's function
def add_channels_single(img: np.ndarray, N_channels: int):
    img = img[:, :, np.newaxis]
    N, M, _ = img.shape

    channels = np.zeros((N, M, N_channels))
    return np.concatenate((img, channels), axis=2)


# Mia's function
def add_channels_batch(img_list, N_channels):
    res_list = []
    for img in img_list:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        N, M, O = img.shape

        channels = np.zeros((N, M, N_channels))
        res = np.concatenate((img, channels), axis=2)

        res_list.append(res)

    return np.array(res_list)


# Mia's function
def add_channels_single_preexisting(img: np.ndarray, channels: np.ndarray):
    # img = img[:,:,np.newaxis]

    return np.concatenate((img, channels), axis=2)


# Mia's function
def get_config(path):
    config.read(path + "/config")

    return config


# Mia's function
def translate(image_batch, new_length: tuple):
    assert len(image_batch.shape) == 4, "Translate only works with batches of NxM images"
    _, N, M, _ = image_batch.shape
    new_length = (new_length[0] - N, new_length[1] - M)
    new_image_batch = []
    for image in image_batch:
        random_split_i = np.random.randint(0, new_length[0])
        rest_i = new_length[0] - random_split_i
        random_split_j = np.random.randint(0, new_length[1])
        rest_j = new_length[1] - random_split_j
        new_image_batch.append(np.pad(image, ((random_split_i, rest_i), (random_split_j, rest_j), (0, 0)), "constant"))

    new_image_batch = np.array(new_image_batch)
    return new_image_batch


# Mia's function
def get_unique_lists(super_list, number_of_sublist_elements):
    """
    For a super_list, f.ex. [0, 1, 2, 3, 4, 5, 6], return every combination with
    number_of_sublist_elements elements. Combinations are ordered, sampled without
    replacement. F.ex. [0, 1, 2] are included, but not [1, 0, 2] or [0, 0, 0].

    Args:
        super_list: list
        number_of_sublist_elements: int

    Returns: list
    """
    combinations = []

    def _unique_lists(current_list, options):
        if len(current_list) == number_of_sublist_elements:
            combinations.append([i for i in current_list])
        else:
            for o in options:
                new_options = [i for i in options if i > o]
                _unique_lists(current_list + [o], new_options)

    _unique_lists([], super_list)

    return combinations


def shuffle(X_data: list, y_data: list):
    """
    Shuffles X_data while shuffling y_data in the same manner (they always correspond).
    F.ex. X_data = [[1, 2], [3, 4]], y_data = [0, 1] -> X_data = [[3, 4], [1, 2]], y_data = [1, 0]

    Returns:
        list, list
    """
    temp = list(zip(X_data, y_data))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    training_data, target_data = np.array(res1), np.array(res2)

    return training_data, target_data
