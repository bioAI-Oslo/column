"""
The ActiveNCA class represents an active neural cellular automata (ANCA) architecture.

It is implemented in numpy, which for the optimization used proved to be the fastest version
(mostly because of the speedup of numba, using CPUs, and not needing the functionality of tensorflow).
However, it also possible to make the architecture in tensorflow or similar.
"""

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


class ActiveNCA:
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
        selection=False,
    ):
        """
        Initialize an ActiveNCA instance.

        Args:
            num_hidden (int): Number of hidden channels.
            hidden_neurons (int or list, optional): if int: One layer with hidden_neurons nodes.
                                          if list: Number of nodes in each layer.
                                          Defaults to 10.
            img_channels (int, optional): Number of image channels (1 or 3, typically). Defaults to 1.
            iterations (int, optional): Number of iterations for the episodes. Defaults to 50.
            position (str, optional): Position type for the NCA. Defaults to "None".
            size_neo (tuple or None, optional): Size of the NEO (Neural Evolution Optimization). If None, it is calculated
                                    based on size_image. Defaults to (size_image[0]-2, size_image[1]-2) if None.
            size_image (tuple or None, optional): Size of the input image. Defaults to (28, 28) if None.
            moving (bool, optional): Whether the NCA is moving or not. Defaults to True.
            mnist_digits (tuple, optional): Tuple of MNIST digits to be used. Defaults to (0, 3, 4).
            labels (list or None, optional): Optional labels for the data. Defaults to None.
            activation (str or None, optional): Activation function to be used. If None, defaults to a linear activation. Defaults to linear if None.
            selection (bool, optional): Whether to use selection or not. Defaults to False.

        """
        if size_image is None:
            size_image = (28, 28)
        self.size_image = size_image

        self.size_neo = get_dimensions(size_image, size_neo)
        self._size_active = (size_image[0] - 2, size_image[1] - 2)

        self.img_channels = img_channels
        self.act_channels = 2 if moving == True else 0
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

        # Add the selection layer. Now the selection layer will be last
        self.selection = selection
        if selection:
            self.weight_shape_list.append((self.num_classes * 3 * 3, 1))  #  + self.position_addon
            self.weight_shape_list.append((1))

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
            images_raw (np.ndarray): The raw input images
            visualize (bool, optional): Not supported, kept for ease of coding
            step (int, optional): The current step number. If None, will classify once, for the full episode.

        Returns:
            np.ndarray: The states of the model after classification (only class channels).
            np.ndarray: The guesses made by the model.
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
            if self.moving == True:
                alter_perception_slicing_batched(
                    self.perceptions_batched, outputs[:, :, :, -self.act_channels :], N_neo, M_neo, N_active, M_active
                )
            if self.moving == "random":
                # Generate random actions between -0.5 and 0.5
                rand_actions = np.random.rand(B, N_neo, M_neo, 2) - 0.5

                alter_perception_slicing_batched(
                    self.perceptions_batched, rand_actions, N_neo, M_neo, N_active, M_active
                )

        output = self.state_batched[:, :, :, -self.num_classes :]  # Only the class channels, but keep the padded size
        if self.selection and (step is None or step == self.iterations - 1):
            input_to_select = np.empty((B * N_neo * M_neo, 3 * 3 * self.num_classes))  # + self.position_addon))
            collect_input_to_select_batched(
                input_to_select, output, self.perceptions_batched, self.position, N_neo, M_neo
            )

            # Get the weight for each output pixel
            select_weights_flat = self.selection_layer(input_to_select)
            select_weights = np.reshape(select_weights_flat, (B, N_neo, M_neo, 1))

            # Apply the weight to the output
            output = output[:, 1 : 1 + N_neo, 1 : 1 + M_neo] * select_weights

            # Take the mean across N_neo and M_neo, but keep those dimensions
            # output = np.mean(output, axis=(1, 2), keepdims=True)  # Remember to change lambda if this is changed

            return output, guesses

        # I should really phase out having guesses here, I don't use it for anything...
        return output[:, 1 : 1 + N_neo, 1 : 1 + M_neo], guesses

    def classify(self, img_raw, visualize=False, step=None, correct_label_index=None):  # , silencing_indexes=None):
        """
        Classify the input image using the trained model.

        Args:
            img_raw (np.ndarray): The raw input image.
            visualize (bool, optional): Whether to visualize the classification process. Defaults to False.
            step (int, optional): The current step number. If None, will classify once, for the full episode.
            correct_label_index (int, optional): The index of the correct label.

        Returns:
            np.ndarray: The state of the model after classification (only class channels).
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
            # The input vector is constructed
            input = np.empty((N_neo * M_neo, 3 * 3 * self.input_dim + self.position_addon))
            collect_input(input, img_raw, self.state, self.perceptions, self.position, N_neo, M_neo, N_active, M_active)

            # The output "guesses" are made by the model for each cell
            guesses = self.dmodel(input)
            # guesses = guesses.numpy()

            # Reshape back into substrate shape
            outputs = np.reshape(guesses[:, :], (N_neo, M_neo, self.output_dim))

            # Update the state/substrate
            self.state[1 : 1 + N_neo, 1 : 1 + M_neo, :] = (
                self.state[1 : 1 + N_neo, 1 : 1 + M_neo, :] + outputs[:, :, : self.input_dim - self.img_channels]
            )

            # Update the perception if moving
            if self.moving == True:
                alter_perception_slicing(
                    self.perceptions, outputs[:, :, -self.act_channels :], N_neo, M_neo, N_active, M_active
                )
            if self.moving == "random":
                # Generate random actions between -0.5 and 0.5
                rand_actions = np.random.rand(N_neo, M_neo, 2) - 0.5

                alter_perception_slicing(self.perceptions, rand_actions, N_neo, M_neo, N_active, M_active)

            # Collect visualization info if needed
            if visualize:
                self.images.append(copy.deepcopy(img_raw))
                self.states.append(copy.deepcopy(self.state))
                if self.moving == True:
                    self.actions.append(copy.deepcopy(outputs[:, :, -self.act_channels :]))
                self.perceptions_through_time.append(copy.deepcopy(self.perceptions))

        # Last step? If visualize, then visualize
        if visualize and (step is None or step == self.iterations - 1):
            self.visualize(
                self.images,
                self.states,
                self.actions if len(self.actions) != 0 else None,
                self.perceptions_through_time,
                self.num_hidden,
                self.num_classes,
                self.labels,
                correct_label_index,
            )
            # Clear visualization info
            self.images = None
            self.states = None
            self.actions = None
            self.perceptions_through_time = None

        output = self.state[:, :, -self.num_classes :]
        if self.selection and (step is None or step == self.iterations - 1):
            input_to_select = np.empty((1 * N_neo * M_neo, 3 * 3 * self.num_classes))  # + self.position_addon))
            collect_input_to_select_batched(
                input_to_select, output[None], self.perceptions[None], self.position, N_neo, M_neo
            )

            # Get the weight for each output pixel
            select_weights_flat = self.selection_layer(input_to_select)
            select_weights = np.reshape(select_weights_flat, (N_neo, M_neo, 1))

            # Apply the weight to the output
            output = output[1 : 1 + N_neo, 1 : 1 + M_neo] * select_weights

            # Take the mean across N_neo and M_neo, but keep those dimensions
            # output = np.mean(output, axis=(0, 1), keepdims=True) # Remember to change lambda if this is changed

            return output, guesses

        return output[1 : 1 + N_neo, 1 : 1 + M_neo, :], guesses

    def visualize(
        self,
        images,
        states,
        actions,
        perceptions_through_time,
        HIDDEN_CHANNELS,
        CLASS_CHANNELS,
        MNIST_DIGITS,
        correct_label_index,
    ):
        """
        Visualize the ActiveNCA's behavior. This function creates a new process and calls the animate function.

        Args:
            images (List of 2D numpy arrays): The images seen by the ActiveNCA
            states (List of 3D numpy arrays): The states of the ActiveNCA
            actions (List of 3D numpy arrays): The actions taken by the ActiveNCA
            perceptions_through_time (List of 3D numpy arrays): The perceptions of the ActiveNCA over time
            HIDDEN_CHANNELS (int): The number of hidden channels in the ActiveNCA
            CLASS_CHANNELS (int): The number of class channels in the ActiveNCA
            MNIST_DIGITS (tuple of int): The digits that the ActiveNCA was trained on
            correct_label_index (int): The index of the correct label in the MNIST_DIGITS tuple
        """
        # It's slower, however the animate function spawns many objects and leads to memory leaks. By using the
        # function in a new process, all objects should be cleaned up at close and the animate function
        # can be used as many times as wanted
        import multiprocessing as mp

        from src.animate import animate

        p = mp.Process(
            target=animate,
            args=(
                images,
                states,
                actions,
                perceptions_through_time,
                HIDDEN_CHANNELS,
                CLASS_CHANNELS,
                MNIST_DIGITS,
                correct_label_index,
            ),
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
        selection=False,
    ):
        """
        Get an instance of the ActiveNCA class with the given parameters and set its weights to the given flat weights.

        Args:
            num_hidden (int): Number of hidden channels.
            hidden_neurons (int or list, optional): if int: One layer with hidden_neurons nodes.
                                          if list: Number of nodes in each layer.
                                          Defaults to 10.
            img_channels (int, optional): Number of image channels (1 or 3, typically). Defaults to 1.
            iterations (int, optional): Number of iterations for the episodes. Defaults to 50.
            position (str, optional): Position type for the NCA. Defaults to "None".
            size_neo (tuple or None, optional): Size of the NEO (Neural Evolution Optimization). If None, it is calculated
                                    based on size_image. Defaults to (size_image[0]-2, size_image[1]-2) if None.
            size_image (tuple or None, optional): Size of the input image. Defaults to (28, 28) if None.
            moving (bool, optional): Whether the NCA is moving or not. Defaults to True.
            mnist_digits (tuple, optional): Tuple of MNIST digits to be used. Defaults to (0, 3, 4).
            labels (list or None, optional): Optional labels for the data. Defaults to None.
            activation (str or None, optional): Activation function to be used. If None, defaults to a linear activation. Defaults to linear if None.

        Returns:
            ActiveNCA: The ActiveNCA instance with the given parameters and weights.
        """
        network = ActiveNCA(
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
            selection=selection,
        )

        network.set_weights(flat_weights)
        return network

    def set_weights(self, flat_weights):
        """
        Set the weights of the model using the provided flattened weights.

        Args:
            flat_weights (np.ndarray): The flattened weights to be reshaped and set for the model.
        """
        self.weights = get_model_weights(flat_weights, self.weight_amount_list, self.weight_shape_list)

        self.create_model()

    def create_model(self):
        """
        Create a feed forward neural network model from the weights and activation functions

        Creates a feed forward neural network model from the weights and activation functions stored in the object.
        The model is then stored in the object as self.dmodel.
        """

        def get_network(weights):
            # Func is a feed forward network that takes input x and returns output y
            def func(x):
                res = x
                for i in range(0, len(weights) - 2, 2):
                    res = self.activation(layer_math(res, weights[i], weights[i + 1]))

                return layer_math(res, weights[-2], weights[-1])

            return func

        if self.selection:
            self.dmodel = get_network(self.weights[:-2])
            self.selection_layer = get_network(
                self.weights[-2:]
            )  # For now, the selection layer is simply linear. Consider sigmoid
        else:
            self.dmodel = get_network(self.weights)


def custom_round_slicing(x: list):
    """
    Works for batches.
    Rounds the values in the input list by applying slicing.
    Negative values are rounded down to -1, positive values are rounded
    up to 1, and zero values are rounded to 0.

    Args:
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
    """
    Clip the values in the array to the range [0, N].

    Alters the array in place.

    Args:
        array (np.ndarray): The array to be clipped.
        N (int): The upper bound to clip to.
        M (int): The upper bound to clip to (should be the same as N).
    """
    for x in range(len(array)):
        for y in range(len(array[0])):
            array[x, y, 0] = min(max(array[x, y, 0], 0), N)
            array[x, y, 1] = min(max(array[x, y, 1], 0), M)


def add_action_slicing(perception: list, action: list, N: int, M: int) -> np.ndarray:
    """
    Modify the perception array by adding a rounded version of the action array
    and clip the resulting values to the range [0, N-1].

    Args:
        perception (np.ndarray): The perception array to be modified.
        action (np.ndarray): The action array to be added to the perception.
        N (int): The valid image coordinate number (so for image size 28x28, N = 28-2 = 26).
        M (int): Must be equal to N, otherwise an assertion error is raised.

    Returns None: The perception array is modified in place.
    """
    perception += custom_round_slicing(action)
    assert N == M, "The code currently does not support N != M"
    clipping(perception, N - 1, M - 1)  # Changes array in place


def alter_perception_slicing(perceptions, actions, N_neo, M_neo, N_active, M_active):
    # TODO: Remove this fucntion, you only need the one below
    """
    Modify the perception array by adding a rounded version of the action array
    and clip the resulting values to the range [0, N_active-1].

    Args:
        perceptions (np.ndarray): The perception array to be modified.
        actions (np.ndarray): The action array to be added to the perception.
        N_neo (int): Not needed
        M_neo (int): Not needed
        N_active (int): The valid image coordinate number (so for image size 28x28, N = 28-2 = 26).
        M_active (int): Must be equal to N_active, otherwise an assertion error is raised.

    Returns None: The perception array is modified in place.
    """
    add_action_slicing(perceptions, actions, N_active, M_active)


@njit
def clipping_batched(array, N, M):
    """
    Clip the values in the array to the range [0, N].

    Alters the array in place.

    Args:
        array (np.ndarray): The array to be clipped.
        N (int): The upper bound to clip to.
        M (int): The upper bound to clip to (should be the same as N).
    """
    B, N_neo, M_neo, _ = array.shape
    for b in range(B):
        for x in range(N_neo):
            for y in range(M_neo):
                array[b, x, y, 0] = min(max(array[b, x, y, 0], 0), N)
                array[b, x, y, 1] = min(max(array[b, x, y, 1], 0), M)


def add_action_slicing_batched(perceptions_batched: list, actions_batched: list, N: int, M: int) -> np.ndarray:
    """
    Modify the perception array by adding a rounded version of the action array
    and clip the resulting values to the range [0, N-1].

    Args:
        perceptions_batched (np.ndarray): The perception array to be modified.
        actions_batched (np.ndarray): The action array to be added to the perception.
        N (int): The valid image coordinate number (so for image size 28x28, N = 28-2 = 26).
        M (int): Must be equal to N, otherwise an assertion error is raised.

    Returns None: The perception array is modified in place.
    """
    perceptions_batched += custom_round_slicing(actions_batched)
    assert N == M, "The code currently does not support N != M"
    clipping_batched(perceptions_batched, N - 1, M - 1)  # Changes array in place


def alter_perception_slicing_batched(perceptions_batched, actions_batched, N_neo, M_neo, N_active, M_active):
    """
    Modify the perception array by adding a rounded version of the action array
    and clip the resulting values to the range [0, N-1].

    Args:
        perceptions_batched (np.ndarray): The perception array to be modified.
        actions_batched (np.ndarray): The action array to be added to the perception.
        N_neo (int): Not needed
        M_neo (int): Not needed
        N_active (int): The valid image coordinate number (so for image size 28x28, N = 28-2 = 26).
        M_active (int): Must be equal to N_active, otherwise an assertion error is raised.

    Returns None: The perception array is modified in place.
    """

    add_action_slicing_batched(perceptions_batched, actions_batched, N_active, M_active)


@njit
def collect_input(input, img, state, perceptions, position, N_neo, M_neo, N_active, M_active):
    """
    Populate the input array for the ActiveNCA.

    Args:
        input (np.ndarray): The array to be populated. Should be of shape (N_neo * M_neo, -1).
        img (np.ndarray): The image to be input to the ActiveNCA. Should be of shape (N, M, -1).
        state (np.ndarray): The state of the ActiveNCA. Should be of shape (N_neo, M_neo, -1).
        perceptions (np.ndarray): The positions of the neo-pixels. Should be of shape (N_neo, M_neo, 2).
        position (str): Which position to use for the input. Can be "current" or "None".
        N_neo (int): The number of neo-pixels in the x-direction.
        M_neo (int): The number of neo-pixels in the y-direction.
        N_active (int): The valid image coordinate number (so for image size 28x28, N = 28-2 = 26).
        M_active (int): Must be equal to N_active, otherwise an assertion error is raised.

    Returns None: The input array is modified in place.
    """
    for x in range(N_neo):
        for y in range(M_neo):
            x_p, y_p = perceptions[x, y]
            perc = img[x_p : x_p + 3, y_p : y_p + 3, :]  # Get the perception, the receptive field neighborhood
            comms = state[x : x + 3, y : y + 3, :]  # Get the communication, the substrate neighborhood
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
    """
    Populate the input array for a batch of images in the ActiveNCA model. Batched version.

    Args:
        input (np.ndarray): The array to be populated. Should be of shape (B * N_neo * M_neo, -1).
        images (np.ndarray): The batch of images to be input to the ActiveNCA. Should be of shape (B, N, M, -1).
        state_batched (np.ndarray): The batched state of the ActiveNCA. Should be of shape (B, N_neo, M_neo, -1).
        perceptions_batched (np.ndarray): The batched positions of the neo-pixels. Should be of shape (B, N_neo, M_neo, 2).
        position (str): Which position to use for the input. Can be "current" or "None".
        N_neo (int): The number of neo-pixels in the x-direction.
        M_neo (int): The number of neo-pixels in the y-direction.
        N_active (int): The valid image coordinate number (e.g., for image size 28x28, N = 28-2 = 26).
        M_active (int): Must be equal to N_active, otherwise an assertion error is raised.

    Returns:
        None: The input array is populated in place.
    """
    B, _, _, _ = images.shape

    for x in range(N_neo):
        for y in range(M_neo):
            for b in range(B):
                x_p, y_p = perceptions_batched[b, x, y].T
                perc = images[
                    b, x_p : x_p + 3, y_p : y_p + 3, :
                ]  # Get the perception, the receptive field neighborhood
                comms = state_batched[b, x : x + 3, y : y + 3, :]  # Get the communication, the substrate neighborhood

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
    # Also keeping it for reference, as the above should be doing what is below
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


def collect_input_to_select_batched(input_to_select, output, perceptions_batched, position, N_neo, M_neo):
    B = len(output)
    for x in range(N_neo):
        for y in range(M_neo):
            for b in range(B):
                dummy_flat = output[b, x : x + 3, y : y + 3, :].flatten()
                input_to_select[b * N_neo * M_neo + x * M_neo + y, : len(dummy_flat)] = dummy_flat

                """if position == "current":
                    x_p, y_p = perceptions_batched[b, x, y].T
                    input_to_select[b * N_neo * M_neo + x * M_neo + y, -2] = (x_p / (N_neo - 1)) * 2 - 1
                    input_to_select[b * N_neo * M_neo + x * M_neo + y, -1] = (y_p / (M_neo - 1)) * 2 - 1"""


def get_dimensions(data_shape, neo_shape):
    """
    Calculate the number of active neo-pixels in the x and y direction.

    Args:
        data_shape (tuple): The shape of the image data. Should be of form (N, M, -1).
        neo_shape (tuple or None): The shape of the neo-pixels. Should be of form (N_neo, M_neo).
            If None, the number of active neo-pixels is set to N_neo = N - 2 and M_neo = M - 2.

    Returns:
        tuple: The number of active neo-pixels in the x and y direction.
    """
    N, M = data_shape
    N_neo = N - 2 if neo_shape is None else neo_shape[0]
    M_neo = M - 2 if neo_shape is None else neo_shape[1]
    return N_neo, M_neo
