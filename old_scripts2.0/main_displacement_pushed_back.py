import copy
import multiprocessing as mp
import time
import warnings

import cma
import numpy as np
import tensorflow as tf
from numba import jit

# Suppressing deprecation warnings from numba because it floods
# the error logs files.
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from src.animate import animate
from src.logger import Logger
from src.loss import pixel_wise_L2, scale_loss
from src.mnist_processing import get_MNIST_data
from src.perception_matrix import get_perception_matrix
from src.utils import (
    add_channels_single_preexisting,
    get_model_weights,
    get_weights_info,
)
from tensorflow.keras.layers import Dense

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
# Supression part over

import random

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# Adjustables
MAXGEN = 10000
ITERATIONS = 50  # 50
POPSIZE = 80
THREADS = 80
MOVING = True  # If we are using the new (moving) or old method (not moving)
POOL_TRAINING = True
CURRENT_POS = True
plotting_interval = 100
saving_interval = 200
visualize_interval = 70000

TRAIN_N_NEO = 15
TRAIN_M_NEO = 15
TEST_N_NEO = 26
TEST_M_NEO = 26

# Saving info
SAVING = False
CONTINUE = False
FILE_ADD_ON = "_displacement"  # "_displacement"

# Visualization
VIS_NUM = 1
VISUALIZE = False
TEST = True

# Data
MNIST_DIGITS = (0, 3, 4)  # (0, 2, 3, 4)
SAMPLES_PER_DIGIT = 5
N_DATAPOINTS = len(MNIST_DIGITS) * SAMPLES_PER_DIGIT

# Channel parameters
HIDDEN_CHANNELS = 3
# Should not be altered:
CLASS_CHANNELS = len(MNIST_DIGITS)
ACT_CHANNELS = 2 * int(MOVING)
IMG_CHANNELS = 1  # Cannot be changed without altering a lot of code

# Script wide functions
loss_function = pixel_wise_L2
data_func = get_MNIST_data
kwargs = {"MNIST_DIGITS": MNIST_DIGITS, "SAMPLES_PER_DIGIT": SAMPLES_PER_DIGIT, "verbose": False}

# Constants that will be used
NUM_CHANNELS = IMG_CHANNELS + HIDDEN_CHANNELS + CLASS_CHANNELS
end_of_class = NUM_CHANNELS
input_dim = NUM_CHANNELS
output_dim = HIDDEN_CHANNELS + CLASS_CHANNELS + ACT_CHANNELS


class NeuralNetwork_TF(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Original architecture
        self.dmodel = tf.keras.Sequential(
            [
                Dense(input_dim * 3 * 3 + 2, input_dim=input_dim * 3 * 3 + 2, activation="linear"),
                Dense(output_dim, activation="linear"),  # or linear
            ]
        )

        # An architecture that should be tested
        """self.dmodel = tf.keras.Sequential(
            [
                Dense(input_dim + 2, input_dim=input_dim * 3 * 3 + 2, activation="relu"),
                Dense(input_dim + 2, input_dim=input_dim + 2, activation="tanh"),
                Dense(output_dim, activation="linear"),  # or linear
            ]
        )"""

        # dummy calls to build the model
        self(tf.zeros([1, 3 * 3 * input_dim + 2]))

    # @tf.function
    def call(self, x):
        return self.dmodel(x)

    @staticmethod
    def get_instance_with(flat_weights):
        network = NeuralNetwork_TF()
        weight_shape_list, weight_amount_list, _ = get_weights_info(network.weights)
        shaped_weight = get_model_weights(flat_weights, weight_amount_list, weight_shape_list)
        network.dmodel.set_weights(shaped_weight)

        return network


# Function to round actions to +1 or -1 pixel, with a very short interval
# for non-moving to encourage moving
# @jit
def custom_round(x: float) -> int:
    """
    Rounds a float value to the nearest integer.
    If the float value is less than -0.0007, returns -1.
    If the float value is greater than 0.0007, returns 1.
    Otherwise, returns 0.

    Parameters:
        x (float): The float value to be rounded.

    Returns:
        int: The rounded integer value.
    """
    if x < -0.0007:
        return -1
    elif x > 0.0007:
        return 1
    return 0


# Adding action to perception while caring for the bounary
# Boundary is handled by stopping movement
# @jit
def add_action(perception: list, action: list, N: int, M: int) -> np.ndarray:
    x_p, y_p = perception
    x_a = custom_round(action[0])
    y_a = custom_round(action[1])
    return np.array([min(max(x_p + x_a, 0), N - 1), min(max(y_p + y_a, 0), M - 1)])


# @jit
def alter_perception(perceptions, actions, N_neo, M_neo, N_active, M_active):
    for x in range(N_neo):
        for y in range(M_neo):
            action = actions[x, y]
            perception = perceptions[x, y]
            perceptions[x, y] = add_action(perception, action, N_active, M_active)


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
    # TODO: Be explicit that this function will change the array
    for x in range(len(array)):
        for y in range(len(array[0])):
            array[x, y, 0] = min(max(array[x, y, 0], 0), N)
            array[x, y, 1] = min(max(array[x, y, 1], 0), M)


def add_action_slicing(perception: list, action: list, N: int, M: int) -> np.ndarray:
    perception += custom_round_slicing(action)
    assert N == M, "The code currently does not support N != M"
    clipping(perception, N - 1, M - 1)  # TODO: Admit that you change the array


def alter_perception_slicing(perceptions, actions, N_neo, M_neo, N_active, M_active):
    # TODO: Remove this fucntion, you only need the one below
    add_action_slicing(perceptions, actions, N_active, M_active)


@jit
def collect_input(input, img, perceptions, N_neo, M_neo):
    N, M, _ = img.shape
    for x in range(N_neo):
        for y in range(M_neo):
            x_p, y_p = perceptions[x, y]
            perc = img[x_p : x_p + 3, y_p : y_p + 3, :1]
            comms = img[x : x + 3, y : y + 3, 1:]
            dummy = np.concatenate((perc, comms), axis=2)
            input[x * M_neo + y, :-2] = dummy.flatten()
            if CURRENT_POS:
                input[x * M_neo + y, -2] = (x_p - N // 2) / (N // 2)
                input[x * M_neo + y, -1] = (y_p - M // 2) / (M // 2)
            else:  # Only init pos
                input[x * M_neo + y, -2] = (float(x) * N / float(N_neo) - N // 2) / (N // 2)
                input[x * M_neo + y, -1] = (float(y) * M / float(M_neo) - M // 2) / (M // 2)
            # input.append(np.concatenate((perc, comms), axis=2))


def classify(network, img, perceptions, N_active, M_active, N_neo, M_neo, visualize=False):
    if visualize:
        images = []
        perceptions_through_time = []
        outputs_through_time = []
        actions = []

    guesses = None
    for _ in range(ITERATIONS):
        """
        Getting input loop takes 0.00013494491577148438 seconds
        Classifying takes 0.0005781650543212891 seconds
        Casting guesses takes 5.0067901611328125e-06 seconds
        Reshapimng outputs takes 3.0994415283203125e-06 seconds
        Taking actions takes 2.8848648071289062e-05 seconds

        Suggestion: Have everything be tensor from the start
        """

        input = np.empty((N_neo * M_neo, 3 * 3 * NUM_CHANNELS + 2))
        collect_input(input, img, perceptions, N_neo, M_neo)

        guesses = network(input)
        guesses = guesses.numpy()

        outputs = np.reshape(guesses[:, :], (N_neo, M_neo, output_dim))

        img[1 : 1 + N_neo, 1 : 1 + M_neo, 1:] = (
            img[1 : 1 + N_neo, 1 : 1 + M_neo, 1:] + outputs[:, :, : end_of_class - IMG_CHANNELS]
        )

        if MOVING:
            alter_perception_slicing(perceptions, outputs[:, :, -ACT_CHANNELS:], N_neo, M_neo, N_active, M_active)

        if visualize:
            images.append(copy.deepcopy(img))
            perceptions_through_time.append(copy.deepcopy(perceptions))
            outputs_through_time.append(copy.deepcopy(outputs))
            actions.append(copy.deepcopy(outputs[:, :, -ACT_CHANNELS:]))

    if visualize:
        # It's slower, however the animate function spawns many objects and leads to memory leaks. By using the function in a new process, all objects should be cleaned up at close and the animate function can be used as many times as wanted
        p = mp.Process(
            target=animate,
            args=(images, perceptions_through_time, actions, HIDDEN_CHANNELS, CLASS_CHANNELS, MNIST_DIGITS),
        )
        p.start()
        p.join()
        p.close()
        # animate(images, perceptions_through_time) # Leads to memory leaks

    return img[1 : 1 + N_neo, 1 : 1 + M_neo, -CLASS_CHANNELS:], guesses, img[:, :, 1:]


# Evaluate one solution
def evaluate_nca(
    flat_weights,
    training_data,
    target_data,
    verbose=False,
    visualize=False,
    N_neo=None,
    M_neo=None,
    return_accuracy=False,
):
    network = NeuralNetwork_TF.get_instance_with(flat_weights)

    # TODO: Separate function?
    N, M = training_data.shape[1:3]
    N_active, M_active = N - 2, M - 2
    if N_neo is None:
        N_neo = N_active
    if M_neo is None:
        M_neo = M_active

    # TODO: Why is this here? What is it?
    seed_state = np.zeros((N, M, NUM_CHANNELS - IMG_CHANNELS))
    state = seed_state  # TODO: What does this do?

    # perceptions
    perceptions = get_perception_matrix(N_active, M_active, N_neo, M_neo)

    loss = 0
    accuracy = 0
    visualized = 0
    for sample, (img_raw, expected) in enumerate(zip(training_data, target_data)):
        if not POOL_TRAINING or (POOL_TRAINING and sample != 0 and sample % 2 == 0):
            # Restart
            state = seed_state
            perceptions = get_perception_matrix(N_active, M_active, N_neo, M_neo)

        img = add_channels_single_preexisting(img_raw, state)

        class_predictions, guesses, state = classify(
            network, img, perceptions, N_active, M_active, N_neo, M_neo, visualize=visualize and (visualized < VIS_NUM)
        )

        if visualize and (visualized < VIS_NUM):
            visualized += 1

        loss += loss_function(class_predictions, guesses, expected)

        if verbose or return_accuracy:
            belief = np.mean(class_predictions, axis=(0, 1))
            believed = np.argmax(belief)
            actual = np.argmax(expected)
            accuracy += int(believed == actual)
            if verbose:
                print("Expected", expected, "got", belief)

    if verbose:
        print("Accuracy:", np.round(accuracy * 100 / training_data.shape[0], 2), "%")

    # TODO: Remove :)
    network = None
    perceptions = None
    seed_state = None
    state = None
    img_raw = None
    expected = None

    if return_accuracy:
        return scale_loss(loss, N_DATAPOINTS), float(accuracy) / float(training_data.shape[0])
    return scale_loss(loss, N_DATAPOINTS)


def run_optimize():
    _, _, weight_amount = get_weights_info(NeuralNetwork_TF().weights)
    print("\nWeights:", int(weight_amount), "\n")

    # Init solution for the ES to initialize
    init_sol = None
    if CONTINUE:
        init_sol = Logger.load_checkpoint(FILE_ADD_ON)
    else:
        init_sol = int(weight_amount.numpy()) * [0.0]
    es = cma.CMAEvolutionStrategy(init_sol, 0.001)  # 0.001

    weight_amount = None
    init_sol = None

    logger_object = None
    generation_numbers = None
    if CONTINUE:
        logger_object = Logger.continue_run(plotting_interval, MAXGEN, FILE_ADD_ON)
        start_point = logger_object.data["x_axis"][-1]
        generation_numbers = range(start_point + 1, MAXGEN + start_point + 1)
    else:
        logger_object = Logger(plotting_interval, MAXGEN, FILE_ADD_ON)
        generation_numbers = range(0, MAXGEN + 1)

    pool = None
    if THREADS > 1:
        print("We're starting pools")
        mp.set_start_method("spawn")
        pool = mp.Pool(THREADS)  # , maxtasksperchild=10)

    bestever_score = np.inf
    bestever_weights = None

    try:
        start_run_time = time.time()
        for g in generation_numbers:
            start_time = time.time()
            print()
            print("Generation", g, flush=True)
            solutions = es.ask(number=POPSIZE)  # , sigma_fac=(((MAXGEN-g)/MAXGEN)*0.9)+0.1)
            training_data, target_data = data_func(**kwargs)

            if pool is None:
                solutions_fitness = [evaluate_nca(s, training_data, target_data) for s in solutions]
            else:
                jobs = [
                    pool.apply_async(
                        evaluate_nca, args=(s, training_data, target_data, False, False, TRAIN_N_NEO, TRAIN_M_NEO)
                    )
                    for s in solutions
                ]
                solutions_fitness = [job.get() for job in jobs]
                # solutions_fitness = pool.starmap(evaluate_nca,
                #    ((s, training_data, target_data, False, False, TRAIN_N_NEO, TRAIN_M_NEO) for s in solutions),
                #    chunksize=POPSIZE // THREADS
                # )
            es.tell(solutions, solutions_fitness)

            if np.min(solutions_fitness) < bestever_score:
                bestever_score = np.min(solutions_fitness)
                bestever_weights = solutions[np.argmin(solutions_fitness)]

            print("Best score:", np.min(solutions_fitness), flush=True)
            print("Bestever score:", bestever_score, flush=True)

            visualize_this_gen = VISUALIZE and g % visualize_interval == 0
            if g % plotting_interval == 0 or visualize_this_gen:
                winner_flat = solutions[np.argmin(solutions_fitness)]

                testing_data, target_data_test = data_func(**kwargs)

                # TODO: args dictionary to avoid clutter
                loss_train_size, acc_train_size = evaluate_nca(
                    winner_flat,
                    testing_data,
                    target_data_test,
                    verbose=visualize_this_gen,
                    visualize=visualize_this_gen,
                    return_accuracy=True,
                    N_neo=TRAIN_N_NEO,
                    M_neo=TRAIN_M_NEO,
                )
                loss_test_size, acc_test_size = evaluate_nca(
                    winner_flat,
                    testing_data,
                    target_data_test,
                    verbose=visualize_this_gen,
                    visualize=visualize_this_gen,
                    return_accuracy=True,
                    N_neo=TEST_N_NEO,
                    M_neo=TEST_M_NEO,
                )

                testing_data, target_data_test = None, None

                if g % plotting_interval == 0:
                    logger_object.store_plotting_data(
                        solutions_fitness,
                        acc_train_size,
                        loss_train_size,
                        acc_test_size,
                        loss_test_size,
                        bestever_score,
                    )

            if SAVING and g % saving_interval == 0:
                logger_object._save_checkpoint(bestever_weights)
                logger_object.save_plotting_data()

            end_time = time.time()
            print("Generation time:", end_time - start_time, "seconds", flush=True)
        end_run_time = time.time()
        print("The entire run took", end_run_time - start_run_time, "seconds")

    except KeyboardInterrupt:
        input("You've cancelled the run. Press enter.")

    if pool is not None:
        pool.close()

    if SAVING:
        logger_object._save_checkpoint(bestever_weights)
        logger_object.save_plotting_data()

    return bestever_weights


if __name__ == "__main__":
    if not TEST:
        winner_flat = run_optimize()
    else:
        winner_flat = Logger.load_checkpoint(FILE_ADD_ON)

    training_data, target_data = data_func(**kwargs, test=True)

    print("\nEvaluating winner:")
    loss, acc = evaluate_nca(
        winner_flat,
        training_data,
        target_data,
        verbose=True,
        visualize=VISUALIZE,
        return_accuracy=True,
        N_neo=TEST_N_NEO,
        M_neo=TEST_M_NEO,
    )

    print("Winner had a loss of", loss, "and an accuracy of", acc, "on test data")
