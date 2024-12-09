"""
This is the main file, where the ActiveNCA is trained and tested. You can also use it to visualize episodes
"""

# Numpy uses a lot of threads when running in parallel.
# This didn't play well with multiprocessing and slurm.
# Set OPENBLAS_NUM_THREADS=1 to reduce numpy threads to 1 for increased speed.
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from functools import partial

import numpy as np
from src.active_nca import ActiveNCA

### NB!
# Most imports are moved if __name__ == "__main__"
# This is done because the workers spawned by multiprocessing ended up importing a bunch of stuff
# they didn't use. This might not be a problem, but in the case of cma, simply importing cma can cause the
# threadcount to increase by 14 (not always, setting NUM_THREADS above limits pycma).
# I apologize for the inconvinience.

# This is for testing the code
deterministic = False
if deterministic:
    import random

    import tensorflow as tf

    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)


def get_from_config(config):
    """
    Given a config object, this function returns a dictionary of keyword arguments for the
    moving NCA, the loss function, the predicting method, the data function and the data function
    kwargs. This is used to create a set of keyword arguments for the moving NCA and the data
    function to be used in train and test functions.

    Args:
        config (Config): The configuration object

    Returns:
        dict: A dictionary of keyword arguments for the moving NCA
        function: The loss function to be used
        function: The predicting method to be used
        function: The data function to be used
        dict: A dictionary of keyword arguments for the data function
    """
    # These local imports are also to avoid these imports when running in parallel
    # Although these might not have to be here, maybe they could've been at the top
    from src.data_processing import (
        get_alternating_pattern,
        get_CIFAR_data,
        get_labels,
        get_MNIST_data,
        get_MNIST_data_padded,
        get_MNIST_data_resized,
        get_MNIST_data_translated,
        get_MNIST_fashion_data,
        get_simple_object,
        get_simple_object_translated,
        get_simple_pattern,
    )
    from src.loss import (
        energy,
        global_mean_medians,
        highest_value,
        highest_vote,
        pixel_wise_CE,
        pixel_wise_CE_and_energy,
        pixel_wise_L2,
        pixel_wise_L2_and_CE,
    )

    mnist_digits = eval(config.dataset.mnist_digits)
    predicting_method = eval(config.training.predicting_method)
    hidden_neurons = (
        eval(config.network.hidden_neurons)
        if type(config.network.hidden_neurons) != int
        else [config.network.hidden_neurons]
    )

    # Dealing with loss
    loss_function = eval(config.training.loss)
    if config.training.lambda_energy is not None:
        lambda_energy = float(config.training.lambda_energy)
        # This is kind of ugly, but at this point I didn't want to mess with the code more than this
        loss_function = partial(loss_function, lambda_energy=lambda_energy)

    # This parameter dictionary will be used for all instances of the network
    moving_nca_kwargs = {
        "size_image": (config.dataset.size, config.dataset.size),
        "num_hidden": config.network.hidden_channels,
        "hidden_neurons": hidden_neurons,
        "img_channels": config.network.img_channels,
        "iterations": config.network.iterations,
        "position": str(config.network.position),
        "moving": config.network.moving,
        "mnist_digits": mnist_digits,
        "labels": mnist_digits,
        "activation": config.network.activation_function,
    }

    # Data function and kwargs
    data_func = eval(config.dataset.data_func)
    kwargs = {
        "CLASSES": mnist_digits,
        "SAMPLES_PER_CLASS": config.dataset.samples_per_digit,
        "verbose": False,
        "test": False,
    }
    # Taking specific care with the data functions
    if (
        config.dataset.data_func == "get_MNIST_data_resized"
        or config.dataset.data_func == "get_simple_object"
        or config.dataset.data_func == "get_simple_object_translated"
    ):
        kwargs["size"] = config.dataset.size
    elif config.dataset.data_func == "get_CIFAR_data":
        kwargs["colors"] = config.dataset.colors

    # Get labels for plotting
    moving_nca_kwargs["labels"] = get_labels(data_func, mnist_digits)

    return moving_nca_kwargs, loss_function, predicting_method, data_func, kwargs


def scale_loss(loss, datapoints):
    """
    Scale loss by the number of datapoints in the minibatch. Batch approved.

    Args:
        loss (float): The loss to be scaled.
        datapoints (int): The number of datapoints in the minibatch.

    Returns:
        float: The scaled loss.
    """
    return loss / datapoints


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
    stable=False,
    return_confusion=False,
):
    """
    Evaluate the ActiveNCA for a set of given weights, for a whole minibatch of datapoints.
    Classify batch is used, the faster version.

    Args:
        flat_weights (np.ndarray): The flattened weights of the network to be evaluated.
        training_data (np.ndarray): The input data to be used for evaluation, of shape (n_samples, N, M, n_channels).
        target_data (np.ndarray): The true labels for the input data.
        moving_nca_kwargs (dict): The keyword arguments to be used for the ActiveNCA.
        loss_function (function): The loss function to be used.
        predicting_method (function): The predicting method to be used.
        verbose (bool, optional): Whether to print the accuracy. Defaults to False.
        visualize (bool, optional): Whether to visualize the ActiveNCA. Defaults to False.
        N_neo (int, optional): The number of neurons in the x direction. Defaults to None.
        M_neo (int, optional): The number of neurons in the y direction. Defaults to None.
        return_accuracy (bool, optional): Whether to return the accuracy. Defaults to False.
        pool_training (bool, optional): Whether to use a pool for training. Defaults to False, should not be True for this function.
        stable (bool, optional): Whether the ActiveNCA should be stable. Stable training just applies the loss more often. Defaults to False.
        return_confusion (bool, optional): Whether to return the confusion matrix. Defaults to False.

    Returns:
        float: The loss of the ActiveNCA.
        float: The accuracy of the ActiveNCA, if return_accuracy is True.
        np.ndarray: The confusion matrix of the ActiveNCA, if return_confusion is True.
    """
    if stable:
        # Get number of steps, after each the loss will be applied
        steps = moving_nca_kwargs["iterations"]
    else:
        # Steps is otherwise 1, so loss will only be applied once
        steps = 1

    # Getting network
    network = ActiveNCA.get_instance_with(flat_weights, size_neo=(N_neo, M_neo), **moving_nca_kwargs)

    # Reset network and get classifications
    B = training_data.shape[0]
    network.reset_batched(B)

    loss = 0
    for step in range(steps):
        # All images are classified at once, with respective substrates and perception matrices
        class_predictions, _ = network.classify_batch(training_data, visualize=False, step=step if stable else None)

        # Get loss
        loss += loss_function(class_predictions, None, target_data)

    # Normalize loss if stable, otherwise does nothing (division by 1)
    loss /= steps

    # Calculate and return accuracy if wanted
    if return_accuracy:
        beliefs = predicting_method(class_predictions)
        accuracy = np.sum(beliefs == np.argmax(target_data, axis=-1))
        accuracy /= B

        if verbose:
            print("Accuracy:", np.round(accuracy * 100, 2), "%")

    # Calculate and return confusion matrix if wanted
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
    stable=False,
    return_confusion=False,
):
    """
    Evaluate the ActiveNCA for a set of given weights, for a whole minibatch of datapoints.
    Classify single is used, the slower version. Now it can be trained with a pool, and it can be visualized.

    To visualize, args.vis_num needs to be available, and this is only available from main, typically.

    Args:
        flat_weights (np.ndarray): The flattened weights of the network to be evaluated.
        training_data (np.ndarray): The input data to be used for evaluation, of shape (n_samples, N, M, n_channels).
        target_data (np.ndarray): The true labels for the input data.
        moving_nca_kwargs (dict): The keyword arguments to be used for the ActiveNCA.
        loss_function (function): The loss function to be used.
        predicting_method (function): The predicting method to be used.
        verbose (bool, optional): Whether to print the accuracy. Defaults to False.
        visualize (bool, optional): Whether to visualize the ActiveNCA. Defaults to False.
        N_neo (int, optional): The number of neurons in the x direction. Defaults to None.
        M_neo (int, optional): The number of neurons in the y direction. Defaults to None.
        return_accuracy (bool, optional): Whether to return the accuracy. Defaults to False.
        pool_training (bool, optional): Whether to use a pool for training. Defaults to False.
        stable (bool, optional): Whether the ActiveNCA should be stable. Stable training just applies the loss more often. Defaults to False.
        return_confusion (bool, optional): Whether to return the confusion matrix. Defaults to False.

    Returns:
        float: The loss of the ActiveNCA.
        float: The accuracy of the ActiveNCA, if return_accuracy is True.
        np.ndarray: The confusion matrix of the ActiveNCA, if return_confusion is True.
    """
    if stable:
        # Get number of steps, after each the loss will be applied
        steps = moving_nca_kwargs["iterations"]
    else:
        # Steps is otherwise 1, so loss will only be applied once
        steps = 1

    network = ActiveNCA.get_instance_with(flat_weights, size_neo=(N_neo, M_neo), **moving_nca_kwargs)

    # Initialize confusion matrix
    if return_confusion:
        conf_matrix = np.zeros((len(target_data[0]), len(target_data[0])), dtype=np.float32)

    loss = 0
    accuracy = 0
    visualized = 0
    # For each data sample, classify
    for sample, (img_raw, expected) in enumerate(zip(training_data, target_data)):
        # Network should always be reset, but when training with pool it should only be reset every other time
        if not pool_training or sample % 2 == 0:
            network.reset()

        # Print debugging info
        if visualize and (visualized < args.vis_num):
            print("Correct:", moving_nca_kwargs["labels"][np.argmax(expected)])

        # Classify
        for step in range(steps):
            # Should the step be visualized
            visualize_step = visualize and (visualized < args.vis_num)

            # Classify
            class_predictions, guesses = network.classify(
                img_raw,
                visualize=visualize_step,
                step=step if stable else None,
                correct_label_index=np.argmax(expected) if visualize_step else None,
            )

            # Successful visualization
            if visualize_step and step == steps - 1:
                visualized += 1

            loss += loss_function(class_predictions, guesses, expected)

        # Print debugging info, and calculate accuracy
        if verbose or return_accuracy:
            belief = np.mean(class_predictions, axis=(0, 1))
            believed = predicting_method(class_predictions)
            actual = np.argmax(expected)
            accuracy += int(believed == actual)
            if verbose:
                print("Expected", expected, "got", belief)

            if return_confusion:
                conf_matrix[actual, believed] += 1

    # Accuracy is only normalized by batch size because it is not collected over "steps"
    if return_accuracy:
        accuracy /= training_data.shape[0]

    if verbose:
        print("Accuracy:", np.round(accuracy * 100, 2), "%")

    # Scale loss by batch size and number of steps
    scaled_loss = scale_loss(loss, training_data.shape[0] * steps)

    # Return loss, and accuracy if requested, and confusion matrix if requested
    if return_confusion and return_accuracy:
        return scaled_loss, accuracy, conf_matrix
    elif return_confusion:
        return scaled_loss, conf_matrix
    elif return_accuracy:
        return scaled_loss, accuracy
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
    evaluate_method=evaluate_nca,
):
    """
    This function is the main function for training ActiveNCA.
    It returns the best solution found by the optimization algorithm,
    which is a cma.CMAEvolutionStrategy.

    The function first prints the number of weights in the model, then
    initializes the ES with the given parameters. It then starts a loop over the
    range of generations, where each generation consists of generating a new set
    of candidate solutions using the ES, evaluating each candidate solution using
    the given loss function and predicting method, and then using the results to
    update the ES.

    The function also allows for plotting and visualization of the results. It
    checks if the generation is divisible by the plotting interval, and if so,
    it plots the current best solution and the bestever solution. It also checks
    if the generation is divisible by the saving interval, and if so, it saves
    the current best solution and the bestever solution.

    Args:
        config (localconfig.config): The configuration object.
        moving_nca_kwargs (dict): The keyword arguments for the moving NCA.
        loss_function (function): The loss function to use.
        predicting_method (function): The predicting method to use.
        data_func (function): The function to generate data with.
        data_kwargs (dict): The keyword arguments for the data function.
        continue_path (str, optional): The path to continue from. If not given, a new optimization run is started.
        save (bool, optional): Whether to save the results. If not given, the results are not saved.
        sub_folder (str, optional): The subfolder to save the results in. If not given, the results are saved in the default subfolder.

    Returns:
        list: The best solution found by the optimization algorithm.
    """
    import cma

    # Print amount of weights
    _, _, weight_amount = get_weights_info(
        ActiveNCA(size_neo=(config.scale.train_n_neo, config.scale.train_m_neo), **moving_nca_kwargs).weights
    )
    print("\nWeights:", int(weight_amount), "\n")

    # Init solution for the ES to initialize, or load from checkpoint
    init_sol = None
    if continue_path is not None:
        init_sol = Logger.load_checkpoint(continue_path)
    else:
        init_sol = int(weight_amount) * [0.0]

    es = cma.CMAEvolutionStrategy(init_sol, config.training.init_sigma)
    if deterministic:
        # CMAEvolutionStrategy does not allow you to set seed in any normal way
        # So I set it here
        np.random.seed(0)

    # Probably not needed, but I had issues with garbage collection on cluster, so just in case!
    weight_amount = None
    init_sol = None

    # Init logger, or load from checkpoint
    logger_object = None
    generation_numbers = None  # Range of generations to run (f.ex 0-100, 100-200, etc.)
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
        pool = mp.Pool(config.training.threads)

    # Record bestever solution
    bestever_score = np.inf  # Lower is better, so worst possible is infinity
    bestever_weights = None

    # Start optimization
    try:  # Try that only catches KeyboardInterrupt, not other exceptions. Just because it's nice when debugging to be able to quit the program with ctrl+c
        # Print buffer to not overwhelm logs on cluster
        print_buffer = []

        start_run_time = time.time()
        for g in generation_numbers:
            start_gen_time = time.time()
            print_buffer.append("")
            print_buffer.append(f"Generation {g}")

            # Get candidate solutions based on CMA-ES internal parameters
            solutions = es.ask(number=config.training.popsize)  # , sigma_fac=0.9999

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
                "pool_training": config.training.pool_training,
                "stable": config.training.stable,
            }

            # Evaluate each candidate solution
            if pool is None:
                solutions_fitness = [evaluate_method(s, **eval_kwargs) for s in solutions]
            else:
                jobs = pool.map_async(partial(evaluate_method, **eval_kwargs), solutions)
                solutions_fitness = jobs.get()
                """jobs = [pool.apply_async(evaluate_nca, args=[s], kwds=eval_kwargs) for s in solutions]
                solutions_fitness = [job.get() for job in jobs]"""

            # weight regularization (sorry that it's here, it was added late in the project)
            if config.training.lambda_weight is not None and config.training.lambda_weight > 0.0:
                for i in range(len(solutions)):
                    solutions_fitness[i] += weight_regularization(
                        solutions[i], lambda_weight=config.training.lambda_weight
                    )

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

                # Testing data is not from test set to avoid overfitting (as in, me choosing models that work well on test set)
                # Meanwhile, all testing later is done on test set to make sure it's not overfitted
                # This simply compares the winning network on different data (as CMA-ES will overfit within a generation)
                testing_data, target_data_test = data_func(**data_kwargs)

                # Alter kwargs for testing. We don't need to change it back
                eval_kwargs["training_data"] = testing_data
                eval_kwargs["target_data"] = target_data_test
                eval_kwargs["verbose"] = visualize_this_gen
                eval_kwargs["visualize"] = visualize_this_gen
                eval_kwargs["return_accuracy"] = True

                # We don't have to change the neo size because it's already train size
                loss_train_size, acc_train_size = evaluate_method(winner_flat, **eval_kwargs)

                # Alter specific neo sizes for testing size
                eval_kwargs["N_neo"] = config.scale.test_n_neo
                eval_kwargs["M_neo"] = config.scale.test_m_neo

                loss_test_size, acc_test_size = evaluate_method(winner_flat, **eval_kwargs)

                # Again, probably not needed, but again just for ease of garbage collection
                testing_data, target_data_test = None, None

                # Do we need to store plotting data?
                if g % config.logging.plotting_interval == 0:
                    logger_object.store_plotting_data(
                        solutions_fitness,
                        acc_train_size,
                        loss_train_size,
                        acc_test_size,
                        loss_test_size,
                        bestever_score,
                    )

                print_buffer.append(f"Accuries are train: {acc_train_size} Test: {acc_test_size}")
            # Plotting and visualization ends here

            # Do we need to save a little bit of data?
            if save and g % config.logging.saving_interval == 0:
                current_best_weights = solutions[np.argmin(solutions_fitness)]
                logger_object.save_checkpoint(current_best_weights, filename="best_network")
                logger_object.save_checkpoint(bestever_weights, filename="bestever_network")
                logger_object.save_plotting_data()

            # Buffer results
            print_buffer.append(f"Current best score: {np.min(solutions_fitness)}")
            print_buffer.append(f"Bestever score: {bestever_score}")
            end_gen_time = time.time()
            print_buffer.append(f"Generation time: {end_gen_time - start_gen_time} seconds")

            # Empty buffer?
            if not save or (save and g % config.logging.saving_interval == 0):
                to_print = ""
                for print_line in print_buffer:
                    to_print += print_line + "\n"
                print(to_print)
                print_buffer.clear()

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
    """
    Parse the arguments given to the program.

    Returns:
        argparse.Namespace: An object with all the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="Main",
        description="This program runs an optimization.",
        epilog="Any questions go to mia.kvalsund@gmail.com",
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
    import argparse
    import multiprocessing as mp
    import time
    import warnings
    from functools import partial

    from localconfig import config

    # Suppressing deprecation warnings from numba because it floods
    # the error logs files.
    from numba.core.errors import (
        NumbaDeprecationWarning,
        NumbaPendingDeprecationWarning,
    )
    from src.data_processing import (
        get_CIFAR_data,
        get_labels,
        get_MNIST_data,
        get_MNIST_data_padded,
        get_MNIST_data_resized,
        get_MNIST_data_translated,
        get_MNIST_fashion_data,
        get_simple_object,
        get_simple_object_translated,
        get_simple_pattern,
        get_test_colors_data,
    )
    from src.logger import Logger
    from src.loss import (
        energy,
        global_mean_medians,
        highest_value,
        highest_vote,
        pixel_wise_CE,
        pixel_wise_CE_and_energy,
        pixel_wise_L2,
        pixel_wise_L2_and_CE,
        weight_regularization,
    )
    from src.utils import get_weights_info

    warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
    warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
    # Supression part over

    # Moved to its own function because of size.
    args = parse_args()

    # Read config
    config.read(args.config)

    moving_nca_kwargs, loss_function, predicting_method, data_func, kwargs = get_from_config(config)

    # Dynamically change if we use the fast or slow function
    if config.training.pool_training or args.visualize:
        evaluate_method = evaluate_nca
    else:
        evaluate_method = evaluate_nca_batch

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
            evaluate_method=evaluate_method,
        )
    else:
        winner_flat = Logger.load_checkpoint(args.test_path)

    # Get test data for new evaluation
    kwargs["SAMPLES_PER_CLASS"] = 1
    kwargs["test"] = True

    # Seeds for testing?
    np.random.seed(42)  # 24 for  fashion # 42 for MNIST and CIFAR
    import random

    random.seed(42)

    # Get test data
    training_data, target_data = data_func(**kwargs)

    print("\nEvaluating winner:")
    loss, acc, conf = evaluate_method(
        winner_flat,
        training_data,
        target_data,
        moving_nca_kwargs,
        loss_function,
        predicting_method,
        verbose=False,
        visualize=args.visualize,
        return_accuracy=True,
        N_neo=config.scale.train_n_neo,  # NB: Evaluates on train size
        M_neo=config.scale.train_m_neo,  # NB: Evaluates on train size
        return_confusion=True,
        pool_training=config.training.pool_training,
        stable=config.training.stable,
    )

    print("Winner had a loss of", loss, "and an accuracy of", acc, "on test data")
    print("Confusion matrix:")
    if args.visualize:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(conf / (kwargs["SAMPLES_PER_CLASS"]), annot=True, cmap="plasma")
        plt.xticks(np.arange(len(moving_nca_kwargs["mnist_digits"])), moving_nca_kwargs["labels"])
        plt.yticks(np.arange(len(moving_nca_kwargs["mnist_digits"])), moving_nca_kwargs["labels"])
        plt.ylabel("Real")
        plt.xlabel("Predicted")

        # Below is the code I used to plot an ordered version of the confusion matrix for Fashion-MNIST
        """new_ordering = [3, 0, 2, 4, 6, 8, 1, 5, 7, 9]

        new_conf = np.zeros((10, 10))
        for i in range(10):
            class_i = new_ordering[i]
            for j in range(10):
                class_j = new_ordering[j]
                new_conf[i][j] = conf[class_i][class_j]

        plt.figure()
        sns.heatmap(np.round(new_conf / (kwargs["SAMPLES_PER_CLASS"]), 2), annot=True, cmap="plasma")
        plt.xticks(np.arange(10), np.array(get_labels(data_func, kwargs["CLASSES"]))[new_ordering], rotation=45)
        plt.yticks(np.arange(10), np.array(get_labels(data_func, kwargs["CLASSES"]))[new_ordering], rotation=-45)
        plt.ylabel("Real")
        plt.xlabel("Predicted")"""
        plt.show()
    else:
        print(conf / (kwargs["SAMPLES_PER_CLASS"]))

    print("Winner has", len(winner_flat), "parameters")
