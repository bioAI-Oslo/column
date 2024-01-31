import copy
import multiprocessing as mp

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

CURRENT_POS = True


class MovingNCA(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        num_hidden,
        iterations,
        current_pos,
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
        self.current_pos = current_pos

        self.mnist_digits = mnist_digits

        # Original architecture
        self.dmodel = tf.keras.Sequential(
            [
                Dense(self.input_dim * 3 * 3 + 2, input_dim=self.input_dim * 3 * 3 + 2, activation="linear"),
                Dense(self.output_dim, activation="linear"),  # or linear
            ]
        )

        # Testing architecture 31-1-2024 18:53
        """self.dmodel = tf.keras.Sequential(
            [
                Dense(20, input_dim=self.input_dim * 3 * 3 + 2, activation="linear"),
                Dense(self.output_dim, activation="linear"),  # or linear
            ]
        )"""

        """position = Input(2)
        image = Input(shape=self.input_dim * 3 * 3)
        reshapen_image = tf.layers.Reshape((3, 3, self.input_dim))(image)
        conv = Conv2D(1, kernel_size=(3, 3), activation="linear", padding="valid")(reshapen_image)
        input_img_plus_pos = tf.concat((conv, position), axis=-1)
        probs = Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(input_img_plus_pos)
        self.dmodel = tf.models.Model(inputs=image, outputs=probs)"""

        self.reset()

        # dummy calls to build the model
        self.dmodel(tf.zeros([1, 3 * 3 * self.input_dim + 2]))

    def reset(self):
        """
        Resets the state by resetting the dmodel layers, the state and the perception matrix
        """
        self.dmodel.reset_states()  # Reset the state if any dmodel.layers is stateful. If not, does nothing.
        self.perceptions = get_perception_matrix(
            self.size_image[0] - 2, self.size_image[1] - 2, self.size_neo[0], self.size_neo[1]
        )
        # The internal state of the artificial neocortex needs to be reset as well
        self.state = np.zeros((self.size_image[0], self.size_image[1], self.input_dim - self.img_channels))

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
            input = np.empty((N_neo * M_neo, 3 * 3 * self.input_dim + 2))
            collect_input(input, img_raw, self.state, self.perceptions, N_neo, M_neo)

            guesses = self.dmodel(input)
            guesses = guesses.numpy()

            outputs = np.reshape(guesses[:, :], (N_neo, M_neo, self.output_dim))

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
        iterations,
        current_pos,
        size_neo=None,
        size_image=None,
        moving=True,
        mnist_digits=(0, 3, 4),
    ):
        network = MovingNCA(
            num_classes=num_classes,
            num_hidden=num_hidden,
            iterations=iterations,
            current_pos=current_pos,
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


@jit
def collect_input(input, img, state, perceptions, N_neo, M_neo):
    N, M, _ = state.shape
    for x in range(N_neo):
        for y in range(M_neo):
            x_p, y_p = perceptions[x, y]
            perc = img[x_p : x_p + 3, y_p : y_p + 3, :1]
            comms = state[x : x + 3, y : y + 3, :]
            dummy = np.concatenate((perc, comms), axis=2)
            input[x * M_neo + y, :-2] = dummy.flatten()
            if CURRENT_POS:
                input[x * M_neo + y, -2] = (x_p - N // 2) / (N // 2)
                input[x * M_neo + y, -1] = (y_p - M // 2) / (M // 2)
            else:  # Only init pos
                input[x * M_neo + y, -2] = (float(x) * N / float(N_neo) - N // 2) / (N // 2)
                input[x * M_neo + y, -1] = (float(y) * M / float(M_neo) - M // 2) / (M // 2)
            # input.append(np.concatenate((perc, comms), axis=2))


def get_dimensions(data_shape, N_neo, M_neo):
    N, M = data_shape
    N_neo = N - 2 if N_neo is None else N_neo
    M_neo = M - 2 if M_neo is None else M_neo
    return N_neo, M_neo
