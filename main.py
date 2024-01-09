import argparse
import multiprocessing as mp
import time
import warnings

import cma
import numpy as np
import tensorflow as tf
from localconfig import config

# Suppressing deprecation warnings from numba because it floods
# the error logs files.
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from src.logger import Logger
from src.loss import highest_value, pixel_wise_L2, scale_loss
from src.mnist_processing import get_MNIST_data
from src.moving_nca import MovingNCA
from src.utils import get_weights_info

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
# Supression part over

import random

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

parser = argparse.ArgumentParser(
    prog="ProgramName", description="This program runs an optimization.", epilog="Text at the bottom of help"
)
parser.add_argument("-c", "--config", type=str, help="The config file to use", default="config")
parser.add_argument(
    "-cp",
    "--continue_path",
    type=str,
    help="The path to continue the run of. If not specified, starts a new run",
    default=None,
)
parser.add_argument("-v", "--visualize", action="store_true", help="Visualize mode")
parser.add_argument("-s", "--save", action="store_true", help="Save data")
parser.add_argument("-t", "--test_path", type=str, help="If not specified, defaults to training", default=None)
parser.add_argument("-vn", "--vis_num", type=int, help="Number of inferences to visualize", default=1)

args = parser.parse_args()

config.read("config")
mnist_digits = eval(config.dataset.mnist_digits)
loss_function = eval(config.training.loss)
predicting_method = eval(config.training.predicting_method)

# Constants
N_DATAPOINTS = len(mnist_digits) * config.dataset.samples_per_digit
CLASS_CHANNELS = len(mnist_digits)
ACT_CHANNELS = 2 * int(config.network.moving)

# Script wide functions
data_func = get_MNIST_data
kwargs = {
    "MNIST_DIGITS": mnist_digits,
    "SAMPLES_PER_DIGIT": config.dataset.samples_per_digit,
    "verbose": False,
}

# Constants that will be used
NUM_CHANNELS = config.network.img_channels + config.network.hidden_channels + CLASS_CHANNELS
end_of_class = NUM_CHANNELS
input_dim = NUM_CHANNELS
output_dim = config.network.hidden_channels + CLASS_CHANNELS + ACT_CHANNELS


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
    network = MovingNCA.get_instance_with(
        flat_weights,
        size_neo=(N_neo, M_neo),
        size_image=training_data.shape[1:3],
        num_classes=CLASS_CHANNELS,
        num_hidden=config.network.hidden_channels,
        iterations=config.network.iterations,
        current_pos=config.network.current_pos,
        moving=config.network.moving,
        mnist_digits=mnist_digits,
    )

    loss = 0
    accuracy = 0
    visualized = 0
    for sample, (img_raw, expected) in enumerate(zip(training_data, target_data)):
        if not config.training.pool_training or (config.training.pool_training and sample != 0 and sample % 2 == 0):
            network.reset()

        # Code further on requires a 3D image. TODO See if this is necessary
        img_raw = img_raw.reshape(img_raw.shape[0], img_raw.shape[1], 1)

        class_predictions, guesses = network.classify(img_raw, visualize=visualize and (visualized < args.vis_num))

        if visualize and (visualized < args.vis_num):
            visualized += 1

        loss += loss_function(class_predictions, guesses, expected)

        if verbose or return_accuracy:
            belief = np.mean(class_predictions, axis=(0, 1))
            believed = predicting_method(class_predictions)
            actual = np.argmax(expected)
            accuracy += int(believed == actual)
            if verbose:
                print("Expected", expected, "got", belief)

    if verbose:
        print("Accuracy:", np.round(accuracy * 100 / training_data.shape[0], 2), "%")

    if return_accuracy:
        return scale_loss(loss, N_DATAPOINTS), float(accuracy) / float(training_data.shape[0])
    return scale_loss(loss, N_DATAPOINTS)


def run_optimize():
    _, _, weight_amount = get_weights_info(
        MovingNCA(
            num_classes=CLASS_CHANNELS,
            num_hidden=config.network.hidden_channels,
            iterations=config.network.iterations,
            current_pos=config.network.current_pos,
            moving=config.network.moving,
            size_neo=(config.scale.train_n_neo, config.scale.train_m_neo),
            size_image=(28, 28),
        ).weights
    )
    print("\nWeights:", int(weight_amount), "\n")

    # Init solution for the ES to initialize
    init_sol = None
    if config.training.continue_run:
        init_sol = Logger.load_checkpoint(config)
    else:
        init_sol = int(weight_amount.numpy()) * [0.0]
    es = cma.CMAEvolutionStrategy(init_sol, 0.001)  # 0.001

    weight_amount = None
    init_sol = None

    logger_object = None
    generation_numbers = None
    if args.continue_path is not None:
        logger_object = Logger.continue_run(config, args.continue_path, save=args.save)
        start_point = logger_object.data["x_axis"][-1]
        generation_numbers = range(start_point + 1, config.training.maxgen + start_point + 1)
    else:
        logger_object = Logger(config, save=args.save)
        generation_numbers = range(0, config.training.maxgen + 1)

    pool = None
    if config.training.threads > 1:
        print("We're starting pools")
        mp.set_start_method("spawn")
        pool = mp.Pool(config.training.threads)  # , maxtasksperchild=10)

    bestever_score = np.inf
    bestever_weights = None

    try:
        start_run_time = time.time()
        for g in generation_numbers:
            start_time = time.time()
            print()
            print("Generation", g, flush=True)
            solutions = es.ask(number=config.training.popsize)  # , sigma_fac=(((MAXGEN-g)/MAXGEN)*0.9)+0.1)
            training_data, target_data = data_func(**kwargs)

            if pool is None:
                solutions_fitness = [evaluate_nca(s, training_data, target_data) for s in solutions]
            else:
                jobs = [
                    pool.apply_async(
                        evaluate_nca,
                        args=(
                            s,
                            training_data,
                            target_data,
                            False,
                            False,
                            config.scale.train_n_neo,
                            config.scale.train_m_neo,
                        ),
                    )
                    for s in solutions
                ]
                solutions_fitness = [job.get() for job in jobs]
            es.tell(solutions, solutions_fitness)

            if np.min(solutions_fitness) < bestever_score:
                bestever_score = np.min(solutions_fitness)
                bestever_weights = solutions[np.argmin(solutions_fitness)]

            print("Best score:", np.min(solutions_fitness), flush=True)
            print("Bestever score:", bestever_score, flush=True)

            visualize_this_gen = args.visualize and g % config.logging.visualize_interval == 0
            if g % config.logging.plotting_interval == 0 or visualize_this_gen:
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
                    N_neo=config.scale.train_n_neo,
                    M_neo=config.scale.train_m_neo,
                )
                loss_test_size, acc_test_size = evaluate_nca(
                    winner_flat,
                    testing_data,
                    target_data_test,
                    verbose=visualize_this_gen,
                    visualize=visualize_this_gen,
                    return_accuracy=True,
                    N_neo=config.scale.test_n_neo,
                    M_neo=config.scale.test_m_neo,
                )

                testing_data, target_data_test = None, None

                if g % config.logging.plotting_interval == 0:
                    logger_object.store_plotting_data(
                        solutions_fitness,
                        acc_train_size,
                        loss_train_size,
                        acc_test_size,
                        loss_test_size,
                        bestever_score,
                    )

            if args.save and g % config.logging.saving_interval == 0:
                logger_object.save_checkpoint(bestever_weights)
                logger_object.save_plotting_data()

            end_time = time.time()
            print("Generation time:", end_time - start_time, "seconds", flush=True)
        end_run_time = time.time()
        print("The entire run took", end_run_time - start_run_time, "seconds")

    except KeyboardInterrupt:
        input("You've cancelled the run. Press enter.")

    if pool is not None:
        pool.close()

    if args.save:
        logger_object.save_checkpoint(bestever_weights)
        logger_object.save_plotting_data()

    return bestever_weights


if __name__ == "__main__":
    if args.test_path is None:
        winner_flat = run_optimize()
    else:
        winner_flat = Logger.load_checkpoint(args.test_path)

    training_data, target_data = data_func(**kwargs, test=True)

    print("\nEvaluating winner:")
    loss, acc = evaluate_nca(
        winner_flat,
        training_data,
        target_data,
        verbose=True,
        visualize=args.visualize,
        return_accuracy=True,
        N_neo=config.scale.test_n_neo,
        M_neo=config.scale.test_m_neo,
    )

    print("Winner had a loss of", loss, "and an accuracy of", acc, "on test data")
