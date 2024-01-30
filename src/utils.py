""" This file copies selected functions from
https://github.com/sidneyp/neural-cellular-robot-substrate/blob/main/src/utils.py
For myself: If this ends up in a published repo, you need better reference and
liscence """

import json
import pickle
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
    assert len(image_batch.shape) == 3, "Translate only works with batches of NxM images"
    _, N, M = image_batch.shape
    new_length = (new_length[0] - N, new_length[1] - M)
    new_image_batch = []
    for image in image_batch:
        random_split_i = np.random.randint(0, new_length[0])
        rest_i = new_length[0] - random_split_i
        random_split_j = np.random.randint(0, new_length[1])
        rest_j = new_length[1] - random_split_j
        new_image_batch.append(np.pad(image, ((random_split_i, rest_i), (random_split_j, rest_j)), "constant"))

    new_image_batch = np.array(new_image_batch)
    return new_image_batch


if __name__ == "__main__":
    img = np.ones((28, 28))
    N_channels = 3
    N, M = img.shape

    channels = np.random.rand(N, M, N_channels)
    new_img = add_channels_single_preexisting(img, channels)

    print(new_img[0, 0])
