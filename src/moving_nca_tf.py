import copy
import multiprocessing as mp
import time

import numpy as np
import tensorflow as tf
from numba import jit
from src.animate import animate
from src.perception_matrix import get_perception_matrix
from src.utils import (
    add_channels_single_preexisting,
    get_model_weights,
    get_weights_info,
)
from tensorflow.keras.layers import Conv2D, Dense, Input


class MovingNCA(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        num_hidden,
        hidden_neurons=10,
        iterations=50,
        position="None",
        size_neo=None,
        size_image=None,
        moving=True,
        mnist_digits=(0, 3, 4),
    ):
        super().__init__()

        if size_image is None:
            size_image = (28, 28)
        self.size_neo = get_dimensions(size_image, size_neo[0], size_neo[1])
        self.size_image = size_image
        self._size_active = (size_image[0] - 2, size_image[1] - 2)

        self.img_channels = 1
        self.act_channels = 2
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.input_dim = self.img_channels + self.num_hidden + self.num_classes
        self.output_dim = self.num_hidden + self.num_classes + self.act_channels
        self.iterations = iterations
        self.moving = moving
        self.position = position
        self.position_addon = 0 if self.position == None else 2

        self.mnist_digits = mnist_digits

        # Adjustable size
        self.define_model()

        # Function to stop gradients for a layer
        def stop_gradients_for_layer(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel = tf.stop_gradient(layer.kernel)
                layer.bias = tf.stop_gradient(layer.bias)
            return layer

        # Apply stop_gradients_for_layer function to each layer in the model
        for i, layer in enumerate(self.dmodel.layers):
            self.dmodel.layers[i] = stop_gradients_for_layer(layer)

        self.reset()

        # Let's make some dummy data to build the model
        inp = np.zeros((self.size_image[0], self.size_image[1], self.img_channels))
        state_inp = np.zeros((self.size_neo[0] + 2, self.size_neo[1] + 2, self.input_dim - self.img_channels))

        # Standard perception matrix also as dummy data
        self_perceptions_expanded = expand(self.perceptions)

        # Padding with None for tf reasons and expanding perception matrix
        inp_batch, state_batch = self.prepare_for_dmodel(inp)

        # Dummy calls to build the model
        self.dmodel([inp_batch, state_batch])

    def define_model(self):
        N_neo, M_neo = self.size_neo

        # Processing state
        state = Input(shape=(N_neo + 2, M_neo + 2, self.input_dim - self.img_channels))  # 1 x N_neo+2 x M_neo+2 x 2
        comms = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), activation="linear", padding="valid")(
            state
        )  # 1 x N_neo x M_neo x 1

        # Processing perception
        # Image will already be gathered version of perception
        image = Input(shape=(N_neo * 3, M_neo * 3, self.img_channels))  # 1 x N_neo*3 x M_neo*3 x 1
        perception = Conv2D(1, kernel_size=(3, 3), strides=(3, 3), activation="linear", padding="valid")(
            image
        )  # 1 x N_neo x M_neo x 1
        full_in = tf.concat((comms, perception), axis=-1)  # 1 x N_neo x M_neo x 2

        probs = Conv2D(self.output_dim, kernel_size=(1, 1), activation="sigmoid")(full_in)  # 1 x N_neo x M_neo x 1
        self.dmodel = tf.keras.Model(inputs=[image, state], outputs=probs)

        # self.dmodel.summary()

    def reset(self):
        """
        Resets the state by resetting the dmodel layers, the state and the perception matrix
        """
        self.dmodel.reset_states()  # Reset the state if any dmodel.layers is stateful. If not, does nothing.
        self.perceptions = get_perception_matrix(
            self.size_image[0] - 2, self.size_image[1] - 2, self.size_neo[0], self.size_neo[1]
        )
        # The internal state of the artificial neocortex needs to be reset as well
        self.state = np.zeros((self.size_neo[0] + 2, self.size_neo[1] + 2, self.input_dim - self.img_channels))

    def prepare_for_dmodel(self, inp):
        self_perceptions_expanded = expand(self.perceptions)
        # gathered = tf.gather_nd(params=inp_batch, indices=self_perceptions_expanded, batch_dims=1)
        gathered = gather_mine(inp, self_perceptions_expanded)

        return gathered[None], self.state[None]

    # @tf.function
    def call(self, img, visualize=False):
        return self.classify(img, visualize)

    def predict_step(self, data):
        # return super().predict_step(data)

        return NotImplementedError()

    def classify(self, img_raw, visualize=False):
        """
        Classify the input image using the trained model.

        Parameters:
            img_raw (np.ndarray): The raw input image.
            visualize (bool, optional): Whether to visualize the classification process. Defaults to False.

        Returns:
            np.ndarray: The state of the model after classification.
            np.ndarray: The guesses made by the model.
        """

        if visualize:
            images = []
            perceptions_through_time = []
            outputs_through_time = []
            actions = []

        N_neo, M_neo = self.size_neo
        N_active, M_active = self._size_active

        guesses = None
        for _ in range(self.iterations):
            # Padding with None to resemble a batch and expanding perception matrix
            inp_batch, state_batch = self.prepare_for_dmodel(img_raw)

            # Dummy calls to build the model
            outputs = self.dmodel([inp_batch, state_batch])[0]
            outputs = outputs.numpy()

            self.state[1 : 1 + N_neo, 1 : 1 + M_neo, :] = (
                self.state[1 : 1 + N_neo, 1 : 1 + M_neo, :] + outputs[:, :, : self.input_dim - self.img_channels]
            )

            if self.moving:
                alter_perception_slicing(
                    self.perceptions, outputs[:, :, -self.act_channels :], N_neo, M_neo, N_active, M_active
                )

            if visualize:
                img = add_channels_single_preexisting(img_raw, self.state)
                images.append(copy.deepcopy(img))
                perceptions_through_time.append(copy.deepcopy(self.perceptions))
                outputs_through_time.append(copy.deepcopy(outputs))
                actions.append(copy.deepcopy(outputs[:, :, -self.act_channels :]))

        if visualize:
            self.visualize(
                images, perceptions_through_time, actions, self.num_hidden, self.num_classes, self.mnist_digits
            )

        return self.state[1 : 1 + N_neo, 1 : 1 + M_neo, -self.num_classes :], guesses

    def visualize(self, images, perceptions_through_time, actions, HIDDEN_CHANNELS, CLASS_CHANNELS, MNIST_DIGITS):
        # It's slower, however the animate function spawns many objects and leads to memory leaks. By using the
        # function in a new process, all objects should be cleaned up at close and the animate function
        # can be used as many times as wanted
        p = mp.Process(
            target=animate,
            args=(images, perceptions_through_time, actions, HIDDEN_CHANNELS, CLASS_CHANNELS, MNIST_DIGITS),
        )
        p.start()
        p.join()
        p.close()
        # animate(images, perceptions_through_time) # Leads to memory leaks

    @staticmethod
    def get_instance_with(
        flat_weights,
        num_classes,
        num_hidden,
        hidden_neurons,
        iterations,
        position,
        size_neo=None,
        size_image=None,
        moving=True,
        mnist_digits=(0, 3, 4),
    ):
        network = MovingNCA(
            num_classes=num_classes,
            num_hidden=num_hidden,
            hidden_neurons=hidden_neurons,
            iterations=iterations,
            position=position,
            size_neo=size_neo,
            size_image=size_image,
            moving=moving,
            mnist_digits=mnist_digits,
        )
        network.set_weights(flat_weights)
        return network

    def set_weights(self, flat_weights):
        weight_shape_list, weight_amount_list, _ = get_weights_info(self.weights)
        shaped_weight = get_model_weights(flat_weights, weight_amount_list, weight_shape_list)
        self.dmodel.set_weights(shaped_weight)

        return None  # Why does it explicitly return None?


def custom_round_slicing(x: list):
    """
    Rounds the values in the input list by applying slicing.
    Negative values are rounded down to -1, positive values are rounded
    up to 1, and zero values are rounded to 0.

    Parameters:
        x (list): The input list of values.

    Returns:
        list: The list of rounded values.
    """
    x_new = np.zeros(x.shape, dtype=np.int64)
    negative = x < -0.0007
    positive = x > 0.0007
    zero = np.logical_not(np.logical_or(positive, negative))
    # zero = ~ (positive + negative) Markus suggests this

    x_new[negative] = -1
    x_new[positive] = 1
    x_new[zero] = 0

    return x_new


@jit
def clipping(array, N, M):
    # This function clips the values in the array to the range [0, N]
    # It alters the array in place
    for x in range(len(array)):
        for y in range(len(array[0])):
            array[x, y, 0] = min(max(array[x, y, 0], 0), N)
            array[x, y, 1] = min(max(array[x, y, 1], 0), M)


def add_action_slicing(perception: list, action: list, N: int, M: int) -> np.ndarray:
    perception += custom_round_slicing(action)
    assert N == M, "The code currently does not support N != M"
    clipping(perception, N - 1, M - 1)  # Changes array in place


def alter_perception_slicing(perceptions, actions, N_neo, M_neo, N_active, M_active):
    # TODO: Remove this fucntion, you only need the one below
    add_action_slicing(perceptions, actions, N_active, M_active)


def get_dimensions(data_shape, N_neo, M_neo):
    N, M = data_shape
    N_neo = N - 2 if N_neo is None else N_neo
    M_neo = M - 2 if M_neo is None else M_neo
    return N_neo, M_neo


def test_expand():
    arr = np.array([[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]], [[2, 0], [2, 1], [2, 2]]])

    arr_expanded = np.array(
        [
            [[0, 0], [0, 1], [0, 2], [0, 1], [0, 2], [0, 3], [0, 2], [0, 3], [0, 4]],
            [[1, 0], [1, 1], [1, 2], [1, 1], [1, 2], [1, 3], [1, 2], [1, 3], [1, 4]],
            [[2, 0], [2, 1], [2, 2], [2, 1], [2, 2], [2, 3], [2, 2], [2, 3], [2, 4]],
            [[1, 0], [1, 1], [1, 2], [1, 1], [1, 2], [1, 3], [1, 2], [1, 3], [1, 4]],
            [[2, 0], [2, 1], [2, 2], [2, 1], [2, 2], [2, 3], [2, 2], [2, 3], [2, 4]],
            [[3, 0], [3, 1], [3, 2], [3, 1], [3, 2], [3, 3], [3, 2], [3, 3], [3, 4]],
            [[2, 0], [2, 1], [2, 2], [2, 1], [2, 2], [2, 3], [2, 2], [2, 3], [2, 4]],
            [[3, 0], [3, 1], [3, 2], [3, 1], [3, 2], [3, 3], [3, 2], [3, 3], [3, 4]],
            [[4, 0], [4, 1], [4, 2], [4, 1], [4, 2], [4, 3], [4, 2], [4, 3], [4, 4]],
        ]
    )

    for row_fasit, row_test in zip(arr_expanded, expand(arr)):
        assert np.array_equal(row_fasit, row_test)


def expand(arr):
    N, M, _ = arr.shape
    expanded_arr = np.zeros((N * 3, M * 3, 2), dtype=np.int32)
    x_p, y_p = arr[:, :, 0], arr[:, :, 1]

    for i in range(3):
        for j in range(3):
            expanded_arr[i::3, j::3, 0] = x_p + i
            expanded_arr[i::3, j::3, 1] = y_p + j

    return expanded_arr  # Fucking "in16" is not supported...


@jit
def gather_mine(arr, movement_expanded):
    N_neo, M_neo, _ = movement_expanded.shape
    new_arr = np.empty((N_neo, M_neo, arr.shape[2]), dtype=arr.dtype)

    for x in range(N_neo // 3):
        for y in range(M_neo // 3):
            new_arr[x, y, :] = arr[movement_expanded[x, y, 0], movement_expanded[x, y, 1], :]

    return new_arr
