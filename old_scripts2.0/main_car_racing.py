import copy
import multiprocessing as mp
import time
import warnings

import cma
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numba import jit

# Suppressing deprecation warnings from numba because it floods the error logs files.
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from tensorflow.keras.layers import Dense

from column.src.animate import animate
from column.src.logger import Logger
from column.src.perception_matrix import get_perception_matrix
from column.src.utils import (
    add_channels_single_preexisting,
    get_model_weights,
    get_weights_info,
)

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
# Supression part over

# Adjustables
MAXGEN = 10000
EPISODE_LENGTH = 100
NUM_EPISODES = 5
POPSIZE = 80
THREADS = 80
MOVING = True  # If we are using the new (moving) or old method (not moving)
CURRENT_POS = True
plotting_interval = 100
saving_interval = 200
visualize_interval = 60000

TRAIN_N_NEO = 15
TRAIN_M_NEO = 15
TEST_N_NEO = 26
TEST_M_NEO = 26

# Saving info
SAVING = True
CONTINUE = False
FILE_ADD_ON = "_car"

# Visualization
VIS_NUM = 3
VISUALIZE = True
TEST = True

# Channel parameters
HIDDEN_CHANNELS = 3
# Should not be altered:
CLASS_CHANNELS = 3
ACT_CHANNELS = 2 * int(MOVING)
IMG_CHANNELS = 1  # Cannot be changed without altering a lot of code

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


def alter_perception_slicing(perceptions, actions, N_neo, M_neo, N_active, M_active):
    add_action_slicing(perceptions, actions, N_active, M_active)


@jit
def collect_input(input, observation, state, perceptions, N_neo, M_neo):
    N, M, _ = state.shape
    for x in range(N_neo):
        for y in range(M_neo):
            x_p, y_p = perceptions[x, y]
            perc = observation[x_p : x_p + 3, y_p : y_p + 3, :]
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


def get_reward(network, env, seed, N_active, M_active, N_neo, M_neo, visualize=False):
    if visualize:
        images = []
        perceptions_through_time = []
        outputs_through_time = []
        actions_recorded = []

    # perceptions
    perceptions = get_perception_matrix(N_active, M_active, N_neo, M_neo)

    observation, info = env.reset(seed=seed)
    state = np.zeros((96, 96, HIDDEN_CHANNELS + CLASS_CHANNELS))

    accumulated_reward = 0
    for _ in range(EPISODE_LENGTH):
        input = np.empty((N_neo * M_neo, 3 * 3 * NUM_CHANNELS + 2))
        gray_image = rgb_to_gray(observation) / 256
        collect_input(input, gray_image[:, :, np.newaxis], state, perceptions, N_neo, M_neo)

        network_output = network(input)
        network_output = network_output.numpy()

        outputs = np.reshape(network_output, (N_neo, M_neo, output_dim))

        # state[1:1+N_neo,1:1+M_neo,HIDDEN_CHANNELS:] = outputs[:,:,HIDDEN_CHANNELS:-ACT_CHANNELS]
        # state[1:1+N_neo,1:1+M_neo,:HIDDEN_CHANNELS] = state[1:1+N_neo,1:1+M_neo,:HIDDEN_CHANNELS] + outputs[:,:,:HIDDEN_CHANNELS]
        state[1 : 1 + N_neo, 1 : 1 + M_neo, :] = state[1 : 1 + N_neo, 1 : 1 + M_neo, :] + outputs[:, :, :-ACT_CHANNELS]

        actions = outputs[:, :, HIDDEN_CHANNELS:-ACT_CHANNELS]
        actions_mean = np.mean(actions, axis=(0, 1))

        for _ in range(5):
            observation, reward, terminated, truncated, info = env.step(actions_mean)
            accumulated_reward += reward

            if terminated or truncated:
                break

        if MOVING:
            alter_perception_slicing(perceptions, outputs[:, :, -ACT_CHANNELS:], N_neo, M_neo, N_active, M_active)

        if visualize:
            images.append(add_channels_single_preexisting(gray_image, state))
            perceptions_through_time.append(copy.deepcopy(perceptions))
            outputs_through_time.append(copy.deepcopy(outputs))
            actions_recorded.append(copy.deepcopy(outputs[:, :, -ACT_CHANNELS:]))

        if terminated or truncated:
            break

    if visualize:
        # It's slower, however the animate function spawns many objects and leads to memory leaks. By using the function in a new process, all objects should be cleaned up at close and the animate function can be used as many times as wanted
        p = mp.Process(
            target=animate,
            args=(images, perceptions_through_time, actions_recorded, HIDDEN_CHANNELS, CLASS_CHANNELS, (0, 1, 2)),
        )
        p.start()
        p.join()
        p.close()
        # animate(images, perceptions_through_time) # Leads to memory leaks

    return accumulated_reward


# Evaluate one solution
def evaluate_nca(flat_weights, seeds, verbose=False, visualize=False, N_neo=None, M_neo=None):
    network = NeuralNetwork_TF.get_instance_with(flat_weights)

    N, M = 96, 96
    N_active, M_active = N - 2, M - 2
    if N_neo is None:
        N_neo = N_active
    if M_neo is None:
        M_neo = M_active

    # Get environment
    env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="rgb_array")

    accumulated_reward = 0
    visualized = 0
    for seed in seeds:
        reward = get_reward(
            network, env, int(seed), N_active, M_active, N_neo, M_neo, visualize=visualize and (visualized < VIS_NUM)
        )
        accumulated_reward += reward

        if visualize and (visualized < VIS_NUM):
            visualized += 1

    avg_reward = accumulated_reward / len(seeds)

    if verbose:
        print("Reward:", avg_reward)

    network = None
    env.close()

    return 1000 - avg_reward


def run_optimize():
    _, _, weight_amount = get_weights_info(NeuralNetwork_TF().weights)
    print("\nWeights:", int(weight_amount), "\n")

    # Init solution for the ES to initialize
    init_sol = None
    if CONTINUE:
        init_sol = Logger.load_checkpoint(FILE_ADD_ON)
    else:
        init_sol = int(weight_amount.numpy()) * [0.0]
    es = cma.CMAEvolutionStrategy(init_sol, 0.005)  # 0.001

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
            seeds = np.random.randint(0, 10000, NUM_EPISODES)

            if pool is None:
                solutions_fitness = [evaluate_nca(s, seeds) for s in solutions]
            else:
                jobs = [
                    pool.apply_async(evaluate_nca, args=(s, seeds, False, False, TRAIN_N_NEO, TRAIN_M_NEO))
                    for s in solutions
                ]
                solutions_fitness = [job.get() for job in jobs]
                # solutions_fitness = pool.starmap(evaluate_nca,
                #    ((s, training_data, target_data, False, False, TRAIN_N_NEO, TRAIN_M_NEO) for s in solutions),
                #    chunksize=POPSIZE // THREADS
                # )
            es.tell(solutions, solutions_fitness)
            es.result_pretty()

            if np.min(solutions_fitness) < bestever_score:
                bestever_score = np.min(solutions_fitness)
                bestever_weights = solutions[np.argmin(solutions_fitness)]

            visualize_this_gen = VISUALIZE and g % visualize_interval == 0
            if g % plotting_interval == 0 or visualize_this_gen:
                winner_flat = solutions[np.argmin(solutions_fitness)]

                seeds = np.random.randint(0, 10000, NUM_EPISODES)

                reward_train_size = evaluate_nca(
                    winner_flat,
                    seeds,
                    verbose=visualize_this_gen,
                    visualize=visualize_this_gen,
                    N_neo=TRAIN_N_NEO,
                    M_neo=TRAIN_M_NEO,
                )
                reward_test_size = evaluate_nca(
                    winner_flat,
                    seeds,
                    verbose=visualize_this_gen,
                    visualize=visualize_this_gen,
                    N_neo=TEST_N_NEO,
                    M_neo=TEST_M_NEO,
                )

                seeds = None

                if g % plotting_interval == 0:
                    logger_object.store_plotting_data(
                        solutions_fitness, 0, reward_train_size, 0, reward_test_size, bestever_score
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


def rgb_to_gray(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    value_image = img[:, :, 2]

    return value_image


def car_racing():
    env = gym.make("CarRacing-v2", domain_randomize=False, render_mode="rgb_array")

    observation, info = env.reset(seed=123)

    acc_reward = 0

    for _ in range(1):
        action = [
            -0.1976405,
            0.8974769,
            0.7431769,
        ]  # env.action_space.sample()  # agent policy that uses the observation and info

        observation, reward, terminated, truncated, info = env.step(action)
        gray_image = rgb_to_gray(observation)

        acc_reward += reward

        if terminated or truncated:
            break
            # observation, info = env.reset()

    print(acc_reward)

    env.close()


if __name__ == "__main__":
    if not TEST:
        winner_flat = run_optimize()
    else:
        winner_flat = Logger.load_checkpoint(FILE_ADD_ON)

    seeds = np.random.randint(0, 10000, NUM_EPISODES)

    print("\nEvaluating winner:")
    reward = evaluate_nca(winner_flat, seeds, verbose=True, visualize=VISUALIZE, N_neo=TEST_N_NEO, M_neo=TEST_M_NEO)

    print("Winner had a reward of", reward, "on test data")
