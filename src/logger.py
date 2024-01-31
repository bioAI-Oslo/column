import json
import os
import time

import numpy as np


class LoggerBase:
    def __init__(self, sub_folder=None):
        """
        LoggerBase has useful methods for all my projects.

        Parameters:
            sub_folder (str): The sub-folder to use for the initialization. Defaults to None.
        """
        path = self.get_path(sub_folder)
        self.path = path
        self.make_experiment_folder(path, sub_folder)

    def save_config(self, config) -> None:
        """
        Saves the config of type `localconfig.Config` to a file.

        Parameters:
            config (localconfig.Config): The config object to be saved.
        """
        # There might already be a config file. Add a unique numerical suffix
        additive = 2
        filename = "/config"
        new_filename = filename
        while os.path.isfile(self.path + new_filename):
            new_filename = filename + "_" + str(additive)
            additive += 1

        # We're safe to save the config
        config.save(self.path + new_filename)

    def get_path(self, sub_folder: str = None) -> str:
        """
        Generates a path for an experiment folder.

        Parameters:
            sub_folder (str, optional): The name of a sub-folder to append to the path. Defaults to None.

        Returns:
            str: The generated path for the experiment folder.
        """
        path = "./experiments"
        if sub_folder is not None:
            path += "/" + sub_folder

        # Make a unique experiment folder name by the time and date
        name = f"/{time.localtime().tm_mday}-{time.localtime().tm_mon}-" + str(time.localtime().tm_year)[-2:]
        name += f"_{time.localtime().tm_hour}:{time.localtime().tm_min}"

        path += name
        # F.ex: ./experiments/5-7-23_18:44

        return path

    def make_experiment_folder(self, path: str, sub_folder: str = None) -> None:
        """
        Create an experiment folder at the specified path.

        Parameters:
            path (str): The path where the experiment folder will be created.
            sub_folder (str, optional): The sub-folder within the experiment folder. Defaults to None.
        """
        experiment_dir = "./experiments"
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)
        if sub_folder is not None:
            if not os.path.isdir(experiment_dir + "/" + sub_folder):
                os.mkdir(experiment_dir + "/" + sub_folder)

        # Sometimes, the path is already made. Add a unique numerical suffix
        additive = 2
        new_path = path
        while os.path.isdir(new_path):
            new_path = path + "_" + str(additive)
            additive += 1

        # We are safe to make the folder
        os.mkdir(new_path)
        print("The folder", new_path, "has been made")
        self.path = new_path


class Logger(LoggerBase):
    def __init__(self, config, sub_folder=None, save=False):
        """
        Initialize the object with the given configuration and optional sub-folder and save flag.

        Parameters:
            config (object): The configuration object.
            sub_folder (str, optional): The sub-folder to use. Defaults to None.
            save (bool, optional): Flag to save the configuration. Defaults to False.
        """
        if save:
            super().__init__(sub_folder)
            self.save_config(config)

        self.plotting_interval = config.logging.plotting_interval
        self.generations = config.training.maxgen

        self.data = {
            "x_axis": [],
            "mean_loss_history": [],
            "std_loss_history": [],
            "training_best_loss_history": [],
            "test_accuracy_train_size": [],
            "test_loss_train_size": [],
            "test_accuracy_test_size": [],
            "test_loss_test_size": [],
            "bestever_score_history": [],
        }

    @staticmethod
    def continue_run(config, path, save=False):
        """
        Generate a logger object for continuing a run.

        Parameters:
            config (object): The configuration object for the run.
            path (str): The path where the logger object will be saved.
            save (bool, optional): Whether to save the configuration object. Defaults to False.

        Returns:
            object: The logger object for the continued run.
        """
        # We don't want to save the config because init always makes a unique experiment folder, which is the wrong place to store the new data
        logger_object = Logger(config, save=False)
        logger_object.path = path
        if save:
            logger_object.save_config(config)

        # This will fail if path is not valid
        with open(logger_object.path + "/plotting_data", "r") as file:
            logger_object.data = json.load(file)
            file.close()

        return logger_object

    @staticmethod
    def load_checkpoint(path):
        best_solution = None

        with open(path + "/bestever_network", "r") as file:
            best_solution = json.loads(file.read())
            file.close()

        return best_solution

    def save_checkpoint(self, solution, filename):
        with open(self.path + "/" + filename, "w") as file:
            file.write(json.dumps(list(solution)))
            file.close()

    def store_plotting_data(
        self, fitnesses, acc_train_size, loss_train_size, acc_test_size, loss_test_size, bestever_score
    ):
        mean_fit = np.mean(fitnesses)
        std_fit = np.std(fitnesses)

        self.data["x_axis"].append(
            0 if len(self.data["x_axis"]) == 0 else self.data["x_axis"][-1] + self.plotting_interval
        )

        self.data["mean_loss_history"].append(mean_fit)
        self.data["std_loss_history"].append(std_fit)
        self.data["training_best_loss_history"].append(np.min(fitnesses))

        self.data["test_accuracy_train_size"].append(acc_train_size)
        self.data["test_loss_train_size"].append(loss_train_size)

        self.data["test_accuracy_test_size"].append(acc_test_size)
        self.data["test_loss_test_size"].append(loss_test_size)

        self.data["bestever_score_history"].append(bestever_score)

    def save_plotting_data(self):
        with open(self.path + "/plotting_data", "w") as file:
            json.dump(self.data, file)
            file.close()

    def save_to_file(self):
        self.save_plotting_data()
