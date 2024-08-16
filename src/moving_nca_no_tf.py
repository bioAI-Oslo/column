import copy

import numpy as np
from numba import jit, njit
from src.perception_matrix import get_perception_matrix
from src.utils import get_model_weights


@njit
def relu(x):
    return np.maximum(0, x)


@njit
def linear(x):
    return x


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@njit
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


@jit
def layer_math(x, weight, bias):
    return x @ weight + bias


class MovingNCA:
    def __init__(
        self,
        num_hidden,
        hidden_neurons=10,
        img_channels=1,
        iterations=50,
        position="None",
        size_neo=None,
        size_image=None,
        moving=True,
        mnist_digits=(0, 3, 4),
        labels=None,
        activation=None,
    ):

        if size_image is None:
            size_image = (28, 28)
        self.size_image = size_image

        self.size_neo = get_dimensions(size_image, size_neo)
        self._size_active = (size_image[0] - 2, size_image[1] - 2)

        self.img_channels = img_channels
        self.act_channels = 2 if moving else 0
        self.num_classes = len(mnist_digits)
        self.num_hidden = num_hidden
        self.input_dim = self.img_channels + self.num_hidden + self.num_classes
        self.output_dim = self.num_hidden + self.num_classes + self.act_channels

        self.activation = eval(activation) if activation is not None else linear

        self.iterations = iterations
        self.moving = moving
        self.position = position
        self.position_addon = 0 if self.position == "None" else 2

        self.mnist_digits = mnist_digits
        self.labels = labels

        # Make the weight shape list. Init with the input layer
        self.weight_shape_list = [
            (self.input_dim * 3 * 3 + self.position_addon, hidden_neurons[0]),
            (hidden_neurons[0]),
        ]
        # Add the hidden layers
        for i in range(1, len(hidden_neurons)):
            self.weight_shape_list.append((hidden_neurons[i - 1], hidden_neurons[i]))  # weight
            self.weight_shape_list.append((hidden_neurons[i]))  # bias

        # Add the output layer
        self.weight_shape_list.append((hidden_neurons[-1], self.output_dim))
        self.weight_shape_list.append((self.output_dim))

        # Init the weights and the weight amounts list
        self.weights = []
        self.weight_amount_list = []
        for shape in self.weight_shape_list:
            self.weights.append(np.zeros(shape))
            self.weight_amount_list.append(self.weights[-1].size)

        # This creates a model with the weights above
        self.create_model()

    def reset(self):
        """
        Resets the state by resetting the dmodel layers, the state and the perception matrix
        """
        # self.dmodel.reset_states()  # Reset the state if any dmodel.layers is stateful. If not, does nothing.
        self.perceptions = get_perception_matrix(
            self.size_image[0] - 2, self.size_image[1] - 2, self.size_neo[0], self.size_neo[1]
        )
        # The internal state of the artificial neocortex needs to be reset as well
        self.state = np.zeros((self.size_neo[0] + 2, self.size_neo[1] + 2, self.input_dim - self.img_channels))

    def reset_batched(self, batch_size):
        """
        Resets the state by resetting the dmodel layers, the state and the perception matrix
        But batched.
        """
        # self.dmodel.reset_states()  # Reset the state if any dmodel.layers is stateful. If not, does nothing.

        # Resetting, or setting, perception matrix (batched version)
        self.perceptions_batched = []

        for _ in range(batch_size):
            self.perceptions_batched.append(
                get_perception_matrix(
                    self.size_image[0] - 2, self.size_image[1] - 2, self.size_neo[0], self.size_neo[1]
                )
            )
        self.perceptions_batched = np.array(self.perceptions_batched)

        # The internal state of the artificial neocortex needs to be reset as well
        self.state_batched = np.zeros(
            (batch_size, self.size_neo[0] + 2, self.size_neo[1] + 2, self.input_dim - self.img_channels)
        )

    # @tf.function
    def call(self, img, visualize=False):
        return self.classify(img, visualize)

    def predict_step(self, data):
        # return super().predict_step(data)

        return NotImplementedError()

    def classify_batch(self, images_raw, visualize=False, step=None):
        """
        Classify a batch of images using the networks perception and state, and update the state and perception accordingly.
        This function alters the object's state and perception (batched).
        If you wish to reset the state and perception, remember to call reset_batched().

        Args:
            images_raw: The raw input images
            visualize: Not supported, kept for ease of coding

        Returns:
            A tuple containing the updated state_batched and the guesses
        """
        B = len(images_raw)
        N_neo, M_neo = self.size_neo
        N_active, M_active = self._size_active

        guesses = None
        iterations = self.iterations if step is None else 1
        for _ in range(iterations):
            # The input vector is the perception and state
            input = np.empty((B * N_neo * M_neo, 3 * 3 * self.input_dim + self.position_addon))
            # This function alters "input" in place
            collect_input_batched(
                input,
                images_raw,
                self.state_batched,
                self.perceptions_batched,
                self.position,
                N_neo,
                M_neo,
                N_active,
                M_active,
            )

            # The output "guesses" are the state and movement
            guesses = self.dmodel(input)
            # guesses = guesses.numpy()

            # Reshape back into network shape
            outputs = np.reshape(guesses[:, :], (B, N_neo, M_neo, self.output_dim))

            # Update the state
            self.state_batched[:, 1 : 1 + N_neo, 1 : 1 + M_neo, :] = (
                self.state_batched[:, 1 : 1 + N_neo, 1 : 1 + M_neo, :]
                + outputs[:, :, :, : self.input_dim - self.img_channels]
            )

            # Update the perception
            if self.moving:
                alter_perception_slicing_batched(
                    self.perceptions_batched, outputs[:, :, :, -self.act_channels :], N_neo, M_neo, N_active, M_active
                )

        # I should really phase out having guesses here, I don't use it for anything... TODO
        return self.state_batched[:, 1 : 1 + N_neo, 1 : 1 + M_neo, -self.num_classes :], guesses

    def classify(self, img_raw, visualize=False, step=None):
        """
        Classify the input image using the trained model.

        Parameters:
            img_raw (np.ndarray): The raw input image.
            visualize (bool, optional): Whether to visualize the classification process. Defaults to False.

        Returns:
            np.ndarray: The state of the model after classification.
            np.ndarray: The guesses made by the model.
        """

        if visualize and (step is None or step == 0):
            self.images = []
            self.states = []
            self.actions = []
            self.perceptions_through_time = []

        N_neo, M_neo = self.size_neo
        N_active, M_active = self._size_active

        guesses = None
        iterations = self.iterations if step is None else 1
        for _ in range(iterations):
            input = np.empty((N_neo * M_neo, 3 * 3 * self.input_dim + self.position_addon))
            collect_input(input, img_raw, self.state, self.perceptions, self.position, N_neo, M_neo, N_active, M_active)

            # guesses = tf.stop_gradient(self.dmodel(input)) # This doesn't make a difference
            guesses = self.dmodel(input)
            # guesses = guesses.numpy()

            outputs = np.reshape(guesses[:, :], (N_neo, M_neo, self.output_dim))

            self.state[1 : 1 + N_neo, 1 : 1 + M_neo, :] = (
                self.state[1 : 1 + N_neo, 1 : 1 + M_neo, :] + outputs[:, :, : self.input_dim - self.img_channels]
            )

            if self.moving:
                alter_perception_slicing(
                    self.perceptions, outputs[:, :, -self.act_channels :], N_neo, M_neo, N_active, M_active
                )

            if visualize:
                self.images.append(copy.deepcopy(img_raw))
                self.states.append(copy.deepcopy(self.state))
                if self.moving:
                    self.actions.append(copy.deepcopy(outputs[:, :, -self.act_channels :]))
                self.perceptions_through_time.append(copy.deepcopy(self.perceptions))

        if visualize and (step is None or step == self.iterations - 1):
            self.visualize(
                self.images,
                self.states,
                self.actions if len(self.actions) != 0 else None,
                self.perceptions_through_time,
                self.num_hidden,
                self.num_classes,
                self.labels,
            )
            self.images = None
            self.states = None
            self.actions = None
            self.perceptions_through_time = None

        return self.state[1 : 1 + N_neo, 1 : 1 + M_neo, -self.num_classes :], guesses

    def visualize(
        self, images, states, actions, perceptions_through_time, HIDDEN_CHANNELS, CLASS_CHANNELS, MNIST_DIGITS
    ):
        # It's slower, however the animate function spawns many objects and leads to memory leaks. By using the
        # function in a new process, all objects should be cleaned up at close and the animate function
        # can be used as many times as wanted
        import multiprocessing as mp

        from src.animate import animate

        p = mp.Process(
            target=animate,
            args=(images, states, actions, perceptions_through_time, HIDDEN_CHANNELS, CLASS_CHANNELS, MNIST_DIGITS),
        )
        p.start()
        p.join()
        p.close()
        # animate(images, perceptions_through_time) # Leads to memory leaks

    @staticmethod
    def get_instance_with(
        flat_weights,
        num_hidden,
        hidden_neurons,
        img_channels,
        iterations,
        position,
        size_neo=None,
        size_image=None,
        moving=True,
        mnist_digits=(0, 3, 4),
        labels=None,
        activation=None,
    ):
        network = MovingNCA(
            num_hidden=num_hidden,
            hidden_neurons=hidden_neurons,
            img_channels=img_channels,
            iterations=iterations,
            position=position,
            size_neo=size_neo,
            size_image=size_image,
            moving=moving,
            mnist_digits=mnist_digits,
            labels=labels,
            activation=activation,
        )

        network.set_weights(flat_weights)
        return network

    def set_weights(self, flat_weights):
        self.weights = get_model_weights(flat_weights, self.weight_amount_list, self.weight_shape_list)

        self.create_model()

    def create_model(self):

        # Func is a feed forward network that takes input x and returns output y
        def func(x):
            res = x
            for i in range(0, len(self.weights) - 2, 2):
                res = self.activation(layer_math(res, self.weights[i], self.weights[i + 1]))

            return layer_math(res, self.weights[-2], self.weights[-1])

        self.dmodel = func


def custom_round_slicing(x: list):
    """
    Works for batches.
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
    x_new[zero] = 0  # Technically not needed, x_new is already 0 everywhere

    return x_new


@njit
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


@njit
def clipping_batched(array, N, M):
    # This function clips the values in the array to the range [0, N]
    # It alters the array in place
    B, N_neo, M_neo, _ = array.shape
    for b in range(B):
        for x in range(N_neo):
            for y in range(M_neo):
                array[b, x, y, 0] = min(max(array[b, x, y, 0], 0), N)
                array[b, x, y, 1] = min(max(array[b, x, y, 1], 0), M)


def add_action_slicing_batched(perceptions_batched: list, actions_batched: list, N: int, M: int) -> np.ndarray:
    perceptions_batched += custom_round_slicing(actions_batched)
    assert N == M, "The code currently does not support N != M"
    clipping_batched(perceptions_batched, N - 1, M - 1)  # Changes array in place


def alter_perception_slicing_batched(perceptions_batched, actions_batched, N_neo, M_neo, N_active, M_active):
    add_action_slicing_batched(perceptions_batched, actions_batched, N_active, M_active)


@njit
def collect_input(input, img, state, perceptions, position, N_neo, M_neo, N_active, M_active):
    for x in range(N_neo):
        for y in range(M_neo):
            x_p, y_p = perceptions[x, y]
            perc = img[x_p : x_p + 3, y_p : y_p + 3, :]
            comms = state[x : x + 3, y : y + 3, :]
            dummy = np.concatenate((perc, comms), axis=2)
            dummy_flat = dummy.flatten()
            input[x * M_neo + y, : len(dummy_flat)] = dummy_flat

            # When position is None, the input is just the perception and comms
            if position == "current":
                # input[x * M_neo + y, -2] = (x_p - N // 2) / (N // 2)
                # input[x * M_neo + y, -1] = (y_p - M // 2) / (M // 2)
                input[x * M_neo + y, -2] = (x_p / (N_active - 1)) * 2 - 1
                input[x * M_neo + y, -1] = (y_p / (M_active - 1)) * 2 - 1
            """elif position == "initial":
                input[x * M_neo + y, -2] = (float(x) * N / float(N_neo) - N // 2) / (N // 2)
                input[x * M_neo + y, -1] = (float(y) * M / float(M_neo) - M // 2) / (M // 2)"""


@njit
def collect_input_batched(
    input, images, state_batched, perceptions_batched, position, N_neo, M_neo, N_active, M_active
):
    B, _, _, _ = images.shape

    for x in range(N_neo):
        for y in range(M_neo):
            for b in range(B):
                x_p, y_p = perceptions_batched[b, x, y].T
                perc = images[b, x_p : x_p + 3, y_p : y_p + 3, :]
                comms = state_batched[b, x : x + 3, y : y + 3, :]

                dummy = np.concatenate((perc, comms), axis=-1)
                dummy_flat = dummy.flatten()
                input[b * N_neo * M_neo + x * M_neo + y, : len(dummy_flat)] = dummy_flat

                # When position is None, the input is just the perception and comms
                if position == "current":
                    # input[b * N_neo * M_neo + x * M_neo + y, -2] = (x_p - N // 2) / (N // 2)
                    # input[b * N_neo * M_neo + x * M_neo + y, -1] = (y_p - M // 2) / (M // 2)
                    input[b * N_neo * M_neo + x * M_neo + y, -2] = (x_p / (N_active - 1)) * 2 - 1
                    input[b * N_neo * M_neo + x * M_neo + y, -1] = (y_p / (M_active - 1)) * 2 - 1
                """elif position == "initial":
                    input[b * N_neo * M_neo + x * M_neo + y, -2] = (float(x) * N / float(N_neo) - N // 2) / (N // 2)
                    input[b * N_neo * M_neo + x * M_neo + y, -1] = (float(y) * M / float(M_neo) - M // 2) / (M // 2)"""

    # Below is the old version, almost as fast, but not quite.
    # If something is wrong with the new version, I trust this version to be correct
    """for i in range(B):
        input_inner = np.empty((N_neo * M_neo, input.shape[1]))
        collect_input(
            input_inner,
            images[i],
            state_batched[i],
            perceptions_batched[i],
            position,
            N_neo,
            M_neo,
        )
        input[i * N_neo * M_neo : (i + 1) * N_neo * M_neo] = input_inner"""


def get_dimensions(data_shape, neo_shape):
    N, M = data_shape
    N_neo = N - 2 if neo_shape is None else neo_shape[0]
    M_neo = M - 2 if neo_shape is None else neo_shape[1]
    return N_neo, M_neo


### The functions below are not really used.


def expand(arr):
    B, N, M, _ = arr.shape
    expanded_arr = np.zeros((B, N * 3, M * 3, 2), dtype=np.int32)
    x_p, y_p = arr[:, :, :, 0], arr[:, :, :, 1]

    for i in range(3):
        for j in range(3):
            expanded_arr[:, i::3, j::3, 0] = x_p + i
            expanded_arr[:, i::3, j::3, 1] = y_p + j

    return expanded_arr  # Fucking "in16" is not supported...


def gather_mine(arr, movement_expanded):
    B, N_neo, M_neo, _ = movement_expanded.shape
    new_arr = np.empty((B, N_neo, M_neo, arr.shape[-1]), dtype=arr.dtype)

    for b in range(B):
        for x in range(N_neo):
            for y in range(M_neo):
                new_arr[b, x, y, :] = arr[b, movement_expanded[b, x, y, 0], movement_expanded[b, x, y, 1], :]

    return new_arr
