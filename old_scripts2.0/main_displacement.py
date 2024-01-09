import copy
import multiprocessing as mp
import sys
import time

import cma
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numba import jit
from tensorflow.keras.layers import Dense, Flatten

from column.src.animate import animate
from column.src.loss import pixel_wise_L2, scale_loss
from column.src.mnist_processing import get_MNIST_data
from column.src.perception_matrix import get_perception_matrix
from column.src.utils import (
    add_channels_single,
    add_channels_single_preexisting,
    get_flat_weights,
    get_model_weights,
    get_weights_info,
    load_checkpoint,
    save_checkpoint,
)

# Adjustables
MAXGEN = 1000
ITERATIONS = 50
POPSIZE = 80
THREADS = 10
MOVING = True  # If we are using the new (moving) or old method (not moving)
POOL_TRAINING = True
RANGE_XY = None
plotting_interval = 25
visualize_interval = 60000
saving_interval = 100

TRAIN_N_NEO = 15
TRAIN_M_NEO = 15
TEST_N_NEO = 26
TEST_M_NEO = 26

# Continuing
SAVING = True
CONTINUE = False

# Visualization
VIS_NUM = 3
VISUALIZE = False
TEST = False

# Data
MNIST_DIGITS = (0, 3, 4)
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

        self.dmodel = tf.keras.Sequential(
            [
                Dense(input_dim * 3 * 3 + 2, input_dim=input_dim * 3 * 3 + 2, activation="linear"),
                Dense(output_dim, activation="linear"),  # or linear
            ]
        )

        self(tf.zeros([1, 3 * 3 * input_dim + 2]))  # dummy calls to build the model

    @tf.function
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
@jit
def custom_round(x: float) -> int:
    if x < -0.0007:
        return -1
    elif x > 0.0007:
        return 1
    return 0


# Adding action to perception while caring for the bounary
# Boundary is handled by stopping movement
@jit
def add_action(perception: list, action: list, N: int, M: int, limits: tuple) -> np.ndarray:
    x_p, y_p = perception
    x_a = custom_round(action[0])
    y_a = custom_round(action[1])
    return np.array([min(max(x_p + x_a, limits[0, 0]), limits[0, 1]), min(max(y_p + y_a, limits[1, 0]), limits[1, 1])])


def custom_round_slicing(x: list):
    x_new = np.zeros(x.shape, dtype=np.int64)
    negative = x < -0.0007
    positive = x > 0.0007
    zero = np.logical_not(np.logical_or(positive, negative))

    x_new[negative] = -1
    x_new[positive] = 1
    x_new[zero] = 0

    return x_new


@jit
def clipping(array, N, M):
    for x in range(len(array)):
        for y in range(len(array[0])):
            array[x, y, 0] = min(max(array[x, y, 0], 0), N)
            array[x, y, 1] = min(max(array[x, y, 1], 0), M)


def add_action_slicing(perception: list, action: list, N: int, M: int) -> np.ndarray:
    perception += custom_round_slicing(action)
    assert N == M, "The code currently does not support N != M"
    clipping(perception, N - 1, M - 1)


@jit
def alter_perception(perceptions, actions, N_neo, M_neo, N_active, M_active, limits=None):
    for x in range(N_neo):
        for y in range(M_neo):
            action = actions[x, y]
            perception = perceptions[x, y]
            perceptions[x, y] = add_action(perception, action, N_active, M_active, limits[x, y])


def alter_perception_slicing(perceptions, actions, N_neo, M_neo, N_active, M_active):
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
            input[x * M_neo + y, -2] = (x_p - N // 2) / (N // 2)
            input[x * M_neo + y, -1] = (y_p - M // 2) / (M // 2)
            # input.append(np.concatenate((perc, comms), axis=2))


def create_limits(perceptions, N_neo, M_neo, N_active, M_active):
    limits_list = []
    for x in range(N_neo):
        limits_list.append([])
        for y in range(M_neo):
            x_p, y_p = perceptions[x, y]

            if RANGE_XY is None:
                limits_list[-1].append(((0, N_active - 1), (0, M_active - 1)))
            else:
                limits_list[-1].append(
                    (
                        (max(0, x_p - RANGE_XY), min(N_active - 1, x_p + RANGE_XY)),
                        (max(0, y_p - RANGE_XY), min(M_active - 1, y_p + RANGE_XY)),
                    )
                )
    return np.array(limits_list)


def classify(network, img, perceptions, limits_list, N_active, M_active, N_neo, M_neo, visualize=False):
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
        # input[:,:,:,4:7] = 0.0

        guesses = network(input)
        guesses = guesses.numpy()

        outputs = np.reshape(guesses[:, :], (N_neo, M_neo, output_dim))

        img[1 : 1 + N_neo, 1 : 1 + M_neo, 1:] = (
            img[1 : 1 + N_neo, 1 : 1 + M_neo, 1:] + outputs[:, :, : end_of_class - IMG_CHANNELS]
        )

        if MOVING:
            alter_perception_slicing(
                perceptions, outputs[:, :, -ACT_CHANNELS:], N_neo, M_neo, N_active, M_active, limits=limits_list
            )

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

    N, M = training_data.shape[1:3]
    N_active, M_active = N - 2, M - 2
    if N_neo is None:
        N_neo = N_active
    if M_neo is None:
        M_neo = M_active

    seed_state = np.zeros((N, M, NUM_CHANNELS - IMG_CHANNELS))
    state = seed_state

    # perceptions
    perceptions = get_perception_matrix(N_active, M_active, N_neo, M_neo)
    limits_list = create_limits(perceptions, N_neo, M_neo, N_active, M_active)

    loss = 0
    accuracy = 0
    visualized = 0
    for sample, (img_raw, expected) in enumerate(zip(training_data, target_data)):
        if sample != 0 and sample % 2 == 0:
            # Restart
            state = seed_state
            perceptions = get_perception_matrix(N_active, M_active, N_neo, M_neo)

        img = add_channels_single_preexisting(img_raw, state)

        if POOL_TRAINING:
            class_predictions, guesses, state = classify(
                network,
                img,
                perceptions,
                limits_list,
                N_active,
                M_active,
                N_neo,
                M_neo,
                visualize=visualize and (visualized < VIS_NUM),
            )
        else:
            class_predictions, guesses, _ = classify(
                network,
                img,
                perceptions,
                limits_list,
                N_active,
                M_active,
                N_neo,
                M_neo,
                visualize=visualize and (visualized < VIS_NUM),
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

    network = None
    perceptions = None
    limits_list = None
    seed_state = None
    state = None
    img_raw = None
    expected = None

    if return_accuracy:
        return scale_loss(loss, N_DATAPOINTS), float(accuracy) / float(training_data.shape[0])
    return scale_loss(loss, N_DATAPOINTS)


def internal_saving(
    best_weights,
    mean_loss_history,
    std_loss_history,
    best_loss_history,
    bestever_loss_history,
    best_accuracy_history,
    best_accuracy_history_test,
    best_loss_history_test,
):
    save_checkpoint(best_weights, "_displacement")

    plt.figure()
    x = np.arange(0, len(mean_loss_history) * plotting_interval, plotting_interval)
    plt.fill_between(
        x,
        np.array(mean_loss_history) - np.array(std_loss_history),
        np.array(mean_loss_history) + np.array(std_loss_history),
        color="#2365A355",
    )
    plt.plot(x, mean_loss_history, label="Mean", color="#2365A3")
    plt.plot(x, best_loss_history, label="Lowest loss", color="#A36588")
    plt.plot(x, bestever_loss_history, label="Lowest loss ever", color="#7702C6")
    plt.plot(x, [i if i < 2.0 else 2.0 for i in best_loss_history_test], label="Test loss (lowest)", color="#0277C6")

    plt.legend()

    plt.savefig("../column_findings/last_fitness_curve_displacement.png")

    plt.figure()
    plt.plot(x, np.ones(len(best_accuracy_history)) - best_accuracy_history, label="Labelling error")
    plt.plot(x, np.ones(len(best_accuracy_history_test)) - best_accuracy_history_test, label="Labelling error test")
    plt.legend()

    plt.savefig("../column_findings/error_curve_displacement.png")

    plt.close("all")
    x = None


def saving(
    best_weights,
    mean_loss_history,
    std_loss_history,
    best_loss_history,
    bestever_loss_history,
    best_accuracy_history,
    best_accuracy_history_test,
    best_loss_history_test,
):
    # More memory leak prevention
    p = mp.Process(
        target=internal_saving,
        args=(
            best_weights,
            mean_loss_history,
            std_loss_history,
            best_loss_history,
            bestever_loss_history,
            best_accuracy_history,
            best_accuracy_history_test,
            best_loss_history_test,
        ),
    )
    p.start()
    p.join()
    p.close()


def run():
    network = NeuralNetwork_TF()

    _, _, weight_amount = get_weights_info(network.weights)
    print("\nWeights:", int(weight_amount), "\n")

    # Init solution for the ES to initialize
    init_sol = None
    if CONTINUE:
        init_sol = load_checkpoint()
    else:
        init_sol = int(weight_amount.numpy()) * [0.0]
    es = cma.CMAEvolutionStrategy(init_sol, 0.001)  # 0.001

    mean_loss_history = []
    std_loss_history = []
    best_loss_history = []
    bestever_loss_history = []
    best_accuracy_history = []
    best_accuracy_history_test = []
    best_loss_history_test = []

    best_weights = None
    best_weights_score = np.inf

    if THREADS > 1:
        test_cores = min(THREADS // 2, 6)
        test_data_list = []
        for _ in range(test_cores):
            test_data_list.append(data_func(**kwargs, test=True))

    pool = None
    if THREADS > 1:
        print("We're starting pools")
        mp.set_start_method("spawn")
        pool = mp.Pool(THREADS)

    try:
        for g in range(MAXGEN):
            start_time = time.time()
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
            es.tell(solutions, solutions_fitness)
            print()
            print("Generation", g, flush=True)
            # es.result_pretty()

            if np.min(solutions_fitness) < best_weights_score:
                best_weights_score = np.min(solutions_fitness)
                best_weights = solutions[np.argmin(solutions_fitness)]

            print("Bestever loss:", best_weights_score)
            print("Current loss: ", np.min(solutions_fitness))

            visualize_this_gen = VISUALIZE and g % visualize_interval == 0
            if g % plotting_interval == 0 or visualize_this_gen:
                winner_flat = solutions[np.argmin(solutions_fitness)]

                if THREADS > 1:
                    jobs = [
                        pool.apply_async(
                            evaluate_nca,
                            args=(
                                winner_flat,
                                test_data_list[i][0],
                                test_data_list[i][1],
                                False,
                                False,
                                TRAIN_N_NEO,
                                TRAIN_M_NEO,
                                True,
                            ),
                        )
                        for i in range(len(test_data_list))
                    ] + [
                        pool.apply_async(
                            evaluate_nca,
                            args=(
                                winner_flat,
                                test_data_list[i][0],
                                test_data_list[i][1],
                                False,
                                False,
                                TEST_N_NEO,
                                TEST_M_NEO,
                                True,
                            ),
                        )
                        for i in range(len(test_data_list))
                    ]
                    test_fitness = [job.get() for job in jobs]

                    loss_reduced_window, acc_reduced_window = np.mean(test_fitness[: len(test_data_list)], axis=0)
                    loss_full_window, acc_full_window = np.mean(test_fitness[len(test_data_list) :], axis=0)

                else:
                    loss_reduced_window, acc_reduced_window = evaluate_nca(
                        winner_flat,
                        test_data_list[i][0],
                        test_data_list[i][1],
                        verbose=visualize_this_gen,
                        visualize=visualize_this_gen,
                        return_accuracy=True,
                        N_neo=TRAIN_N_NEO,
                        M_neo=TRAIN_M_NEO,
                    )
                    loss_full_window, acc_full_window = evaluate_nca(
                        winner_flat,
                        test_data_list[i][0],
                        test_data_list[i][1],
                        verbose=visualize_this_gen,
                        visualize=visualize_this_gen,
                        return_accuracy=True,
                        N_neo=TEST_N_NEO,
                        M_neo=TEST_M_NEO,
                    )

                if g % plotting_interval == 0:
                    best_accuracy_history.append(acc_reduced_window)
                    best_accuracy_history_test.append(acc_full_window)

                    mean_fit = np.mean(solutions_fitness)
                    std_fit = np.std(solutions_fitness)

                    mean_loss_history.append(mean_fit)
                    std_loss_history.append(std_fit)
                    best_loss_history.append(loss_reduced_window)
                    best_loss_history_test.append(loss_full_window)
                    bestever_loss_history.append(best_weights_score)

            if SAVING and g % saving_interval == 0 and g != 0:
                saving(
                    best_weights,
                    mean_loss_history,
                    std_loss_history,
                    best_loss_history,
                    bestever_loss_history,
                    best_accuracy_history,
                    best_accuracy_history_test,
                    best_loss_history_test,
                )

            end_time = time.time()
            print("ES size:", sys.getsizeof(es), "bytes")
            print("MLH size:", sys.getsizeof(mean_loss_history), "bytes")
            print("Net size:", sys.getsizeof(best_weights), "bytes")
            print("Pool size:", sys.getsizeof(pool), "bytes")
            print("Generation time:", end_time - start_time, "seconds", flush=True)

    except KeyboardInterrupt:
        input("You've cancelled the run. Press enter.")

    if pool is not None:
        pool.close()

    if SAVING:
        saving(
            best_weights,
            mean_loss_history,
            std_loss_history,
            best_loss_history,
            bestever_loss_history,
            best_accuracy_history,
            best_accuracy_history_test,
            best_loss_history_test,
        )

    return best_weights


if __name__ == "__main__":
    if not TEST:
        winner_flat = run()
    else:
        winner_flat = load_checkpoint()

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
