"""
Train a CNN on current data_func. The data_func should be one of the functions in src/data_processing.py
Various test functions are also included

This is the main CNN file.
"""

import functools
import json
import os
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from src.data_processing import (
    get_CIFAR_data,
    get_max_samples_balanced,
    get_MNIST_data,
    get_MNIST_data_padded,
    get_MNIST_data_resized,
    get_MNIST_fashion_data,
)
from src.utils import get_unique_lists, get_weights_info
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tqdm import tqdm

# Globals
CLASSES = (0, 1, 2, 3, 4)
trained_image_size = 28  # 28x28
img_channels = 1  # 1 or 3, 3 if colors, 1 if grayscale

# Data
data_func = get_MNIST_data
kwargs = {
    "CLASSES": CLASSES,
    "verbose": False,
    "colors": True if img_channels == 3 else False,
}

# Because all datasets are required to be balanced, we need to find the largest dataset possible for a given data_func
samples_per_digit = get_max_samples_balanced(data_func, **kwargs, test=False)
samples_per_digit_test = get_max_samples_balanced(data_func, **kwargs, test=True)


# The CNN model to be used. It is a convolutional neural network, very simple, but works well
class CNN(tf.keras.Model):
    def __init__(self, digits=5, img_dim=(trained_image_size, trained_image_size, img_channels)):
        super().__init__()

        self.dmodel = tf.keras.Sequential(
            [
                Conv2D(10, 3, padding="valid", input_shape=img_dim, activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(10, 3, padding="valid", activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(100, activation="relu"),
                Dense(digits, activation="softmax"),
            ]
        )

        N, M, O = img_dim
        out = self(tf.zeros([1, N, M, O]))  # dummy calls to build the model

        _, _, weight_amount = get_weights_info(self.weights)
        print("Weights:", int(weight_amount))

    # @tf.function
    def call(self, x):
        return self.dmodel(x)


def plot_history(history):
    """
    Plots the training history of a model.

    This function takes the history object returned by a Keras model's `fit` method
    and plots the training loss and categorical accuracy over epochs.

    Args:
        history: A Keras History object containing the training loss and accuracy
                information.
    """
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["categorical_accuracy"], label="accuracy")
    plt.legend()
    plt.show()


def get_model(saved_model):
    """
    Loads a pre-trained CNN model.

    This function initializes a CNN model with the specified number of output digits,
    compiles it with Adam optimizer and categorical crossentropy loss, and loads
    the weights from a saved model checkpoint.

    Args:
        saved_model (str): The file path to the saved model weights.

    Returns:
        model (tf.keras.Model): The compiled CNN model with loaded weights.
    """
    model = CNN(digits=len(CLASSES))
    model.compile(
        optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    model.load_weights(saved_model)
    return model


def print_mean_score(path):
    """
    Prints the mean accuracy score of all models in the given directory tree.

    This function is used to quickly get the mean accuracy score of all models
    in a directory tree. The mean accuracy score is calculated by loading each
    model, evaluating it on the test data, and then taking the mean of all those
    scores.

    Args:
        path (str): The directory path to look for models in.
    """
    # Buffering the prints because the tensorflow functions are very verbose
    to_print = []
    for mode, sample_number in zip(["train", "test"], [samples_per_digit, samples_per_digit_test]):
        # Get data
        images, labels = data_func(CLASSES=CLASSES, SAMPLES_PER_CLASS=sample_number, verbose=False, test=mode == "test")

        # Get scores
        all_scores = {}

        for sub_path in os.listdir(path):
            if os.path.isdir(path + sub_path):
                saved_model = path + sub_path + "/checkpoint_model.ckpt"
                model = get_model(saved_model)

                test_loss, test_acc = model.evaluate(images, labels)
                all_scores[sub_path] = test_acc

        # Buffer the prints
        to_print.append(mode.capitalize() + ": " + str(np.mean(list(all_scores.values()))))

    # Print
    for line in to_print:
        print(line)
    input()  # Again, verbosity is high, so just makign sure I will see the prints


def plot_state_damage(silencing=True, type="square", plot=False):
    """
    Analyzes the effect of state damage on CNN model accuracy.

    This function evaluates the impact of silencing or altering
    feature maps in a trained CNN model for a given dataset,
    using different damage methods (e.g., random or square).
    The results are plotted and saved as a JSON file for further analysis.

    Args:
        silencing (bool): If True, feature maps are silenced; if False,
                          they are altered using bias and ReLU.
        type (str): The method of selecting cells to alter
                    ("random" or "square").
        plot (bool): If True, plots the accuracy scores against
                     the percentage of silenced cells.
    """
    # Import damage methods
    from zero_shot_damage import (
        alter_divisible,
        flip_values,
        sample_randomly,
        sample_rectangular,
        sample_squarely,
        set_to_random,
        set_to_zero,
    )

    # The folder to plot from
    path = "./experiments/cnn/mnist5/"

    # Get data
    images, labels = data_func(CLASSES=CLASSES, SAMPLES_PER_CLASS=40, verbose=False, test=True)
    true_digits = np.argmax(labels, axis=1)

    # The sizes to test
    test_sizes_percentages = np.linspace(0, 1, 11)

    # Get scores
    all_scores = {}

    for sub_path in os.listdir(path):
        if os.path.isdir(path + sub_path):
            saved_model = path + sub_path + "/checkpoint_model.ckpt"
            model = get_model(saved_model)

            if silencing is False:
                bias = model.weights[1].numpy()
                bias_relu = tf.keras.activations.relu(bias).numpy()

            # Get test sizes
            _, N, M, O = model.dmodel.layers[1].input_shape
            test_sizes = np.round(np.array(test_sizes_percentages, dtype=float) * (N * M)).astype(int)

            # Get accuracies
            accuracies = []
            for test_size in test_sizes:
                agreed = 0
                for image, true in zip(images, true_digits):  # For all samples

                    # Run the sample through the first layer to get the first feature map
                    output = np.array(
                        model.dmodel.layers[0](image.reshape(1, trained_image_size, trained_image_size, img_channels))
                    )

                    # Get the indexes of the feature map to silence
                    if type == "random":
                        x_indexes, y_indexes = sample_randomly(test_size, N, M)
                    elif type == "square":
                        x_indexes, y_indexes = sample_squarely(test_size, N, M)
                    else:
                        # raise exception
                        raise Exception(f'Invalid type "{type}"')

                    # Silence the feature map
                    for x in x_indexes:
                        for y in y_indexes:
                            if silencing:
                                output[:, x - 1, y - 1, :] = 0
                            else:
                                # If input was 0 (no whisker), then only bias would be used, as well as relu.
                                output[:, x - 1, y - 1, :] = bias_relu

                    # Run the sample through the rest of the layers
                    for layer in model.dmodel.layers[1:]:
                        output = layer(output)

                    # Get the prediction
                    pred = np.argmax(output)
                    if true == pred:
                        agreed += 1

                print(test_size, agreed / len(true_digits))
                accuracies.append(agreed / len(true_digits))

            all_scores[sub_path] = accuracies

    # Plot
    if plot:
        plt.figure()
        for key, value in all_scores.items():
            plt.plot(test_sizes_percentages, value, label=key)

        ax = plt.gca()
        ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
        ax.set_xticks(test_sizes_percentages, np.round(test_sizes_percentages * 100))
        ax.set_ylabel("Retained accuracy (%)")
        ax.set_xlabel("Randomly silenced cells (%)")

        plt.show()

    # Save
    all_scores["test_sizes"] = test_sizes_percentages.tolist()
    name_add = "silencing" if silencing else "no_silencing"
    json.dump(all_scores, open(path + f"/{type}_{name_add}_robustness.json", "w"))


def main():
    """
    Train a CNN on current data_func.

    The CNN is trained on current data_func with a custom loss function which is a combination of categorical cross-entropy and an L2 regularization term.
    The model is trained for 30 epochs with a batch size of 200.
    The model is then evaluated on the test set and the test accuracy is printed.
    The model is then saved to a checkpoint file.

    Returns:
        float: The test accuracy of the model.
    """
    train_batches = 200
    train_epochs = 30

    # Change this so it corresponds with the globals
    superfolder = "mnist5"

    # Make a unique experiment folder name by the time and date
    name = f"{time.localtime().tm_mday}-{time.localtime().tm_mon}-" + str(time.localtime().tm_year)[-2:]
    name += f"_{time.localtime().tm_hour}:{time.localtime().tm_min}"

    # Sometimes, the path is already made. Add a unique numerical suffix
    additive = 2
    new_name = name
    while os.path.isdir(f"./experiments/cnn/{superfolder}/{new_name}"):
        new_name = name + "_" + str(additive)
        additive += 1

    saved_model = f"./experiments/cnn/{superfolder}/{new_name}/checkpoint_model.ckpt"

    # Get data
    train_x, train_y = data_func(**kwargs, SAMPLES_PER_CLASS=samples_per_digit)
    test_x, test_y = data_func(**kwargs, SAMPLES_PER_CLASS=samples_per_digit_test, test=True)

    model = CNN(digits=len(CLASSES))

    # Create a custom loss function that corresponds with the energy loss function in the paper
    class EnergyLoss(tf.keras.losses.Loss):
        def __init__(self, rate=0.01):
            super().__init__()
            self.rate = rate

        def call(self, y_true, y_pred):
            return np.sum(y_pred**2) * self.rate

    # Make sure it doesn't use softmax, because we already do it in the model
    model.compile(
        optimizers.Adam(lr=1e-4),
        loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=False), EnergyLoss(rate=0.0001)],
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # Train
    history = model.fit(train_x, train_y, batch_size=train_batches, epochs=train_epochs)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_x, test_y)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    # Plot
    plot_history(history)

    # Save
    model.save_weights(saved_model)
    return test_acc


def confusion_matrix(path):
    """
    Loads a CNN model and evaluates it on the test data, then computes the
    confusion matrix and plots it.

    Args:
        path (str): The path to the model checkpoint file.
    """
    model = get_model(path)
    test_x, test_y = data_func(**kwargs, SAMPLES_PER_CLASS=samples_per_digit_test, test=True)

    # Get predictions
    predictions = model.predict(test_x)
    predictions = np.argmax(predictions, axis=1)
    test_y = np.argmax(test_y, axis=1)

    # Compute confusion matrix
    confusion = np.zeros((len(CLASSES), len(CLASSES)))
    for pred, real in zip(predictions, test_y):
        confusion[pred, real] += 1

    # Normalize
    confusion /= samples_per_digit_test

    # Plot
    plt.title("CNN confusion matrix")
    sns.heatmap(confusion, annot=True, cmap="plasma")
    plt.ylabel("Guess")
    plt.xlabel("Answer")
    plt.show()


if __name__ == "__main__":
    path = "./experiments/cnn/mnist5/"
    print_mean_score(path)
