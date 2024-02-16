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
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
    pixel_wise_L2_and_CE,
    scale_loss,
)
from src.mnist_processing import (
    get_MNIST_data,
    get_MNIST_data_padded,
    get_MNIST_data_resized,
    get_MNIST_data_translated,
)
from src.moving_nca import MovingNCA
from src.utils import get_weights_info

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
# Supression part over

deterministic = False
if deterministic:
    import random

    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)


# Evaluate one solution batched
def evaluate_nca_batch(
    flat_weights,
    training_data,
    target_data,
    moving_nca_kwargs,
    loss_function,
    predicting_method,
    verbose=False,
    visualize=False,
    N_neo=None,
    M_neo=None,
    return_accuracy=False,
    pool_training=False,
    return_confusion=False,
    silenced=0,
):
    """
    Network creation took 0.10552453994750977 # Amounts to 26 minutes extra across 15000 gens
    Reshape took 1.1682510375976562e-05 # Amounts to 1.5 seconds extra across 15000 gens
    Reset took 0.04755043983459473 # Amounts to 11 minutes extra across 15000 gens
    Classify took 0.29085350036621094 # Amounts to 1.2 hours extra across 15000 gens
    Loss took 0.0036323070526123047 # Amounts to 54 seconds extra across 15000 gens
    Predict took 7.152557373046875e-07
    Eval took 0.44785308837890625 # Amounts to 1.99 hours across 15000 gens
    """

    assert pool_training is False, "Batch currently does not support pool training"
    assert visualize is False, "Batch currently does not support visualizing"
    assert silenced == 0, "Batch currently does not support silencing"

    network = MovingNCA.get_instance_with(flat_weights, size_neo=(N_neo, M_neo), **moving_nca_kwargs)

    B, N, M = training_data.shape
    images_raw = training_data.reshape(B, N, M, 1)

    network.reset_batched(B)
    class_predictions, _ = network.classify_batch(images_raw, visualize=False)

    loss = loss_function(class_predictions, None, target_data)

    if return_accuracy:
        beliefs = predicting_method(class_predictions)
        accuracy = np.sum(beliefs == np.argmax(target_data, axis=-1))
        accuracy /= B

        if verbose:
            print("Accuracy:", np.round(accuracy * 100 / B, 2), "%")

    if return_confusion:
        beliefs = predicting_method(class_predictions)
        confusion = np.zeros((len(target_data[0]), len(target_data[0])), dtype=np.int32)
        for b in range(B):
            confusion[np.argmax(target_data[b]), beliefs[b]] += 1

        if return_accuracy:
            return loss, accuracy, confusion
        return loss, confusion

    if return_accuracy:
        return loss, accuracy
    return loss


# Evaluate one solution
def evaluate_nca(
    flat_weights,
    training_data,
    target_data,
    moving_nca_kwargs,
    loss_function,
    predicting_method,
    verbose=False,
    visualize=False,
    N_neo=None,
    M_neo=None,
    return_accuracy=False,
    pool_training=False,
    return_confusion=False,
    silenced=0,
):
    """
    Network creation time: 0.015133857727050781 # Amounts to 225 seconds extra across 15000 gens
    Scale loss time: 7.152557373046875e-07
    Reshape time: 1.1920928955078125e-06 # Amounts to 1.5 seconds extra across 15000 gens
    Classification time: 0.022127866744995117 # Amounts to 9.1 hours extra across 15000 gens
    Loss time: 0.0012400150299072266 # Amounts to 30 minutes extra across 15000 gens
    """
    network = MovingNCA.get_instance_with(flat_weights, size_neo=(N_neo, M_neo), **moving_nca_kwargs)

    loss = 0
    accuracy = 0
    visualized = 0
    for sample, (img_raw, expected) in enumerate(zip(training_data, target_data)):
        if not pool_training or (pool_training and sample != 0 and sample % 2 == 0):
            network.reset()

        # Code further on requires a 3D image. It's not worth fixing, it takes so little time to do this
        img_raw = img_raw.reshape(img_raw.shape[0], img_raw.shape[1], 1)

        class_predictions, guesses = network.classify(
            img_raw, visualize=visualize and (visualized < args.vis_num), silenced=silenced
        )

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

    scaled_loss = scale_loss(loss, training_data.shape[0])
    if return_accuracy:
        return scaled_loss, float(accuracy) / float(training_data.shape[0])
    return scaled_loss


def run_optimize(
    config,
    moving_nca_kwargs,
    loss_function,
    predicting_method,
    data_func,
    data_kwargs,
    continue_path=None,
    save=False,
    sub_folder=None,
):
    # Print amount of weights
    _, _, weight_amount = get_weights_info(
        MovingNCA(size_neo=(config.scale.train_n_neo, config.scale.train_m_neo), **moving_nca_kwargs).weights
    )
    print("\nWeights:", int(weight_amount), "\n")

    # Init solution for the ES to initialize
    init_sol = None
    if continue_path is not None:
        init_sol = Logger.load_checkpoint(continue_path)
    else:
        init_sol = int(weight_amount.numpy()) * [0.0]

    es = cma.CMAEvolutionStrategy(init_sol, config.training.init_sigma)  # 0.001
    if deterministic:
        # CMAEvolutionStrategy does not allow you to set seed in any normal way
        # So I set it here
        np.random.seed(0)

    weight_amount = None
    init_sol = None

    # Init logger
    logger_object = None
    generation_numbers = None
    if continue_path is not None:
        logger_object = Logger.continue_run(config, continue_path, save=save)
        start_point = logger_object.data["x_axis"][-1]
        generation_numbers = range(start_point + 1, config.training.maxgen + start_point + 1)
    else:
        logger_object = Logger(config, save=save, sub_folder=sub_folder)
        generation_numbers = range(0, config.training.maxgen + 1)

    # Init pool
    pool = None
    if config.training.threads > 1:
        print("We're starting pools with", config.training.threads, "threads")
        mp.set_start_method("spawn")
        pool = mp.Pool(config.training.threads)  # , maxtasksperchild=10)

    # Record bestever solution
    bestever_score = np.inf
    bestever_weights = None

    """
    Solutions took 0.022756099700927734
    Training data took 0.00043892860412597656
    Evaluations took 1.0909454822540283
    Tell took 0.0347287654876709
    Bestever took 9.036064147949219e-05
    Plotting took 0.5259389877319336
    Saving took 0.006429910659790039
    """

    try:
        start_run_time = time.time()
        for g in generation_numbers:
            start_time = time.time()
            print()
            print("Generation", g, flush=True)

            # Get candidate solutions based on CMA-ES internal parameters
            solutions = es.ask(number=config.training.popsize)  # , sigma_fac=(((MAXGEN-g)/MAXGEN)*0.9)+0.1)

            # Generate training data for evaluating each candidate solution
            training_data, target_data = data_func(**data_kwargs)

            eval_kwargs = {
                "training_data": training_data,
                "target_data": target_data,
                "moving_nca_kwargs": moving_nca_kwargs,
                "loss_function": loss_function,
                "predicting_method": predicting_method,
                "verbose": False,
                "visualize": False,
                "N_neo": config.scale.train_n_neo,
                "M_neo": config.scale.train_m_neo,
                "return_accuracy": False,
                "pool_training": False,
            }

            # Evaluate each candidate solution
            if pool is None:
                solutions_fitness = [evaluate_nca_batch(s, **eval_kwargs) for s in solutions]
            else:
                jobs = [pool.apply_async(evaluate_nca_batch, args=[s], kwds=eval_kwargs) for s in solutions]
                solutions_fitness = [job.get() for job in jobs]

            # Tell es what the result was. It uses this to update its parameters
            es.tell(solutions, solutions_fitness)

            # Record winner for plotting and the future
            if np.min(solutions_fitness) < bestever_score:
                bestever_score = np.min(solutions_fitness)
                bestever_weights = solutions[np.argmin(solutions_fitness)]

            # Plotting and visualization starts here
            visualize_this_gen = args.visualize and g % config.logging.visualize_interval == 0
            if g % config.logging.plotting_interval == 0 or visualize_this_gen:
                winner_flat = solutions[np.argmin(solutions_fitness)]

                testing_data, target_data_test = data_func(**data_kwargs)

                # Alter kwargs for testing. We don't need to change it back
                eval_kwargs["training_data"] = testing_data
                eval_kwargs["target_data"] = target_data_test
                eval_kwargs["verbose"] = visualize_this_gen
                eval_kwargs["visualize"] = visualize_this_gen
                eval_kwargs["return_accuracy"] = True

                # We don't have to change the neo size because it's already train size
                loss_train_size, acc_train_size = evaluate_nca_batch(winner_flat, **eval_kwargs)

                # Alter specific neo sizes for testing size
                eval_kwargs["N_neo"] = config.scale.test_n_neo
                eval_kwargs["M_neo"] = config.scale.test_m_neo

                loss_test_size, acc_test_size = evaluate_nca_batch(winner_flat, **eval_kwargs)

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

                print("Accuries are train:", acc_train_size, "Test:", acc_test_size, flush=True)
            # Plotting and visualization ends here

            # Do we need to save a little bit of data?
            if save and g % config.logging.saving_interval == 0:
                current_best_weights = solutions[np.argmin(solutions_fitness)]
                logger_object.save_checkpoint(current_best_weights, filename="best_network")
                logger_object.save_checkpoint(bestever_weights, filename="bestever_network")
                logger_object.save_plotting_data()

            # Display results
            print("Current best score:", np.min(solutions_fitness), flush=True)
            print("Bestever score:", bestever_score, flush=True)
            end_time = time.time()
            print("Generation time:", end_time - start_time, "seconds", flush=True)

        end_run_time = time.time()
        print("The entire run took", end_run_time - start_run_time, "seconds")

    except KeyboardInterrupt:
        input("You've cancelled the run. Press enter.")

    # Close pool if it exists
    if pool is not None:
        pool.close()

    # Do we need to save a little bit of data?
    if save:
        current_best_weights = solutions[np.argmin(solutions_fitness)]
        logger_object.save_checkpoint(current_best_weights, filename="best_network")
        logger_object.save_checkpoint(bestever_weights, filename="bestever_network")
        logger_object.save_plotting_data()

    # Winner is best solution. But it could be best individual of last generation
    # Luckily both are saved for future use
    return bestever_weights


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="Main", description="This program runs an optimization.", epilog="Text at the bottom of help"
    )
    parser.add_argument("-c", "--config", type=str, help="The config file to use", default="config")
    parser.add_argument("-sf", "--sub_folder", type=str, help="The sub folder to use", default=None)
    parser.add_argument(
        "-cp",
        "--continue_path",
        type=str,
        help="The path to continue the run of. If not specified, starts a new run",
        default=None,
    )
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize mode")
    parser.add_argument("-s", "--save", action="store_true", help="Save data")
    parser.add_argument("-tp", "--test_path", type=str, help="If not specified, defaults to training", default=None)
    parser.add_argument("-vn", "--vis_num", type=int, help="Number of inferences to visualize", default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Moved to its own function because of size.
    args = parse_args()

    # Read config
    config.read(args.config)
    mnist_digits = eval(config.dataset.mnist_digits)
    loss_function = eval(config.training.loss)
    predicting_method = eval(config.training.predicting_method)

    # This parameter dictionary will be used for all instances of the network
    moving_nca_kwargs = {
        "size_image": (28, 28),  # Changed below if we need to
        "num_classes": len(mnist_digits),
        "num_hidden": config.network.hidden_channels,
        "hidden_neurons": config.network.hidden_neurons,
        "iterations": config.network.iterations,
        "position": config.network.position,
        "moving": config.network.moving,
        "mnist_digits": mnist_digits,
    }

    # Data function and kwargs
    data_func = eval(config.dataset.data_func)
    kwargs = {
        "MNIST_DIGITS": mnist_digits,
        "SAMPLES_PER_DIGIT": config.dataset.samples_per_digit,
        "verbose": False,
    }
    # Taking specific care with the data functions
    if config.dataset.data_func == "get_MNIST_data_resized":
        kwargs["size"] = config.dataset.size
        moving_nca_kwargs["size_image"] = (config.dataset.size, config.dataset.size)
    elif config.dataset.data_func == "get_MNIST_data_translated":
        # Size of translated data "get_MNIST_data_translated" is 70x70, specified in the function
        moving_nca_kwargs["size_image"] = (70, 70)
    elif config.dataset.data_func == "get_MNIST_data_padded":
        # Size of translated data "get_MNIST_data_translated" is 70x70, specified in the function
        moving_nca_kwargs["size_image"] = (56, 56)

    # Should we optimize to get a new winner, or load winner?
    if args.test_path is None:
        winner_flat = run_optimize(
            config=config,
            moving_nca_kwargs=moving_nca_kwargs,
            loss_function=loss_function,
            predicting_method=predicting_method,
            data_func=data_func,
            data_kwargs=kwargs,
            continue_path=args.continue_path,
            save=args.save,
            sub_folder=args.sub_folder,
        )
    else:
        winner_flat = Logger.load_checkpoint(args.test_path)

    # Get test data for new evaluation
    training_data, target_data = data_func(**kwargs, test=True)

    print("\nEvaluating winner:")
    loss, acc, conf = evaluate_nca_batch(
        winner_flat,
        training_data,
        target_data,
        moving_nca_kwargs,
        loss_function,
        predicting_method,
        verbose=False,
        visualize=args.visualize,
        return_accuracy=True,
        N_neo=config.scale.test_n_neo,
        M_neo=config.scale.test_m_neo,
        return_confusion=True,
        silenced=0,
    )

    print("Winner had a loss of", loss, "and an accuracy of", acc, "on test data")
    print("Confusion matrix:")
    print(conf)
