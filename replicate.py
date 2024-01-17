import multiprocessing as mp

import cma
import numpy as np
import tensorflow as tf
from src.mnist_processing import get_MNIST_data
from src.utils import (
    add_channels_single,
    add_channels_single_preexisting,
    get_flat_weights,
    get_model_weights,
    get_weights_info,
)
from tensorflow.keras.layers import Dense, Flatten

MAXGEN = 10000
THREADS = 4
POPSIZE = 30

# Data
MNIST_DIGITS = (0, 3, 4)
SAMPLES_PER_DIGIT = 5
N_DATAPOINTS = len(MNIST_DIGITS) * SAMPLES_PER_DIGIT

# Script wide functions
data_func = get_MNIST_data
kwargs = {"MNIST_DIGITS": MNIST_DIGITS, "SAMPLES_PER_DIGIT": SAMPLES_PER_DIGIT, "verbose": False}


class NeuralNetwork_TF(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.dmodel = tf.keras.Sequential(
            [
                Flatten(input_shape=(28, 28)),
                Dense(5, input_dim=28 * 28, activation="linear"),
                Dense(3, activation="linear"),  # or linear
            ]
        )

        self(tf.zeros([1, 28, 28]))  # dummy calls to build the model

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


def loss_function(expected, predicted):
    return np.sum((expected - predicted) ** 2)


def evaluate(flat_weights, data_x, data_y):
    network = NeuralNetwork_TF.get_instance_with(flat_weights)

    predicted = network(data_x)

    loss = 0
    for label, pred in zip(data_y, predicted):
        loss += loss_function(label, pred)

    return loss / len(data_y)


def run_optimize():
    _, _, weight_amount = get_weights_info(NeuralNetwork_TF().weights)
    print("\nWeights:", int(weight_amount), "\n")

    # Init solution for the ES to initialize
    init_sol = int(weight_amount.numpy()) * [0.0]
    es = cma.CMAEvolutionStrategy(init_sol, 0.001)

    pool = None
    if THREADS > 1:
        mp.set_start_method("spawn")
        pool = mp.Pool(THREADS)  # , maxtasksperchild=10)

    for g in range(MAXGEN):
        solutions = es.ask(number=POPSIZE)  # , sigma_fac=(((MAXGEN-g)/MAXGEN)*0.9)+0.1)

        training_data, target_data = data_func(**kwargs)

        if pool is None:
            solutions_fitness = [evaluate(s, training_data, target_data) for s in solutions]
        else:
            jobs = [pool.apply_async(evaluate, args=(s, training_data, target_data)) for s in solutions]
            solutions_fitness = [job.get() for job in jobs]

        es.tell(solutions, solutions_fitness)
        print("\nGen", g, ":")
        es.result_pretty()

    return solutions[np.argmin(solutions_fitness)]


if __name__ == "__main__":
    winner_flat = run_optimize()
