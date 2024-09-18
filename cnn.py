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

CLASSES = (0, 1, 2)
trained_image_size = 28
img_channels = 1
fashion = False

# Data
data_func = get_MNIST_data
kwargs = {
    "CLASSES": CLASSES,
    "verbose": False,
    "colors": True if img_channels == 3 else False,
}


samples_per_digit = get_max_samples_balanced(data_func, **kwargs, test=False)

samples_per_digit_test = get_max_samples_balanced(data_func, **kwargs, test=True)


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

        """self.dmodel = tf.keras.Sequential(
            [
                Conv2D(32, 4, padding="valid", input_shape=img_dim, activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(32, 4, padding="valid", activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(digits, activation="relu"),
            ]
        )"""

        N, M, O = img_dim
        out = self(tf.zeros([1, N, M, O]))  # dummy calls to build the model

        _, _, weight_amount = get_weights_info(self.weights)
        print("Weights:", int(weight_amount))

    # @tf.function
    def call(self, x):
        return self.dmodel(x)


def plot_history(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["categorical_accuracy"], label="accuracy")
    plt.legend()
    plt.show()


def resize_for_CNN(size, mnist_digits):
    # First, get normal resized data that the MovingNCA could handle well
    data_func = get_MNIST_data_resized
    test_x, test_y = data_func(
        CLASSES=mnist_digits, SAMPLES_PER_CLASS=samples_per_digit_test, size=size, verbose=False, test=True
    )

    # Then, resize to again be 56x56, with whatever information loss was incurred
    resized_x_data = []
    for img in test_x:
        resized_x_data.append(
            cv2.resize(img, (trained_image_size, trained_image_size), interpolation=cv2.INTER_NEAREST)
        )

    return np.array(resized_x_data), test_y


def get_model(saved_model):
    model = CNN(digits=len(CLASSES))
    model.compile(
        optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    model.load_weights(saved_model)
    return model


def plot_image_scale_always_full_size():
    model = get_model()

    losses = []
    accuracies = []

    start_size = 7

    for size in range(start_size, trained_image_size + 1):
        # Get data
        test_x, test_y = resize_for_CNN(size, CLASSES)

        # Evaluate model at this size
        test_loss, test_acc = model.evaluate(test_x, test_y)
        losses.append(test_loss)
        accuracies.append(test_acc)

    plt.plot(range(start_size, trained_image_size + 1), losses, label="Loss")
    plt.plot(range(start_size, trained_image_size + 1), accuracies, label="Accuracy")
    plt.legend()
    plt.show()


def plot_state_damage():
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
    path = "./experiments/cnn/mnist3/"

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
                    x_indexes, y_indexes = sample_squarely(test_size, N, M)

                    # Silence the feature map
                    for x in x_indexes:
                        for y in y_indexes:
                            output[:, x - 1, y - 1, :] = 0

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
    json.dump(all_scores, open(path + "/square_silencing_robustness.json", "w"))


def plot_image_noise_robustness():
    from zero_shot_noise_perturbations import add_noise

    # The folder to plot from
    path = "./experiments/cnn/mnist3/"

    # The noise to test
    to_test = np.linspace(0, 1.0, 11)
    NUM_DATA = 40

    # Get data
    images, labels = data_func(CLASSES=CLASSES, SAMPLES_PER_CLASS=NUM_DATA, verbose=False, test=True)
    true_digits = np.argmax(labels, axis=1)

    # Get scores
    all_scores = {}

    for sub_path in os.listdir(path):
        if os.path.isdir(path + sub_path):
            saved_model = path + sub_path + "/checkpoint_model.ckpt"
            model = get_model(saved_model)

            accuracies = []

            for noise in to_test:

                # Add noise
                images_noisy = add_noise(images, noise)

                # Get accuracies
                predictions = model.predict(images_noisy)

                acc = tf.keras.metrics.CategoricalAccuracy()
                acc.update_state(labels, predictions)
                accuracies.append(float(acc.result().numpy()))

            all_scores[sub_path] = list(accuracies)

    # Plot
    for key, value in all_scores.items():
        plt.plot(to_test, value, label=key)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 1.1, 0.1), range(0, 110, 10))
    ax.set_xticks(to_test)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Randomly silenced cells (%)")

    plt.show()

    print(all_scores)

    # Save
    all_scores["test_sizes"] = to_test.tolist()
    json.dump(all_scores, open(path + "/image_noise_robustness.json", "w"))


def plot_image_scale():
    model = get_model()

    losses = []
    accuracies = []

    for size in range(7, trained_image_size + 1):
        # Get data
        test_x, test_y = get_MNIST_data_resized(
            CLASSES=CLASSES, SAMPLES_PER_CLASS=samples_per_digit_test, size=size, verbose=False, test=True
        )

        # Pad with zeros so that the network will accept it
        diff = trained_image_size - size
        diff_x, diff_y = int(np.floor(diff / 2)), int(np.ceil(diff / 2))

        test_x = np.pad(test_x, ((0, 0), (diff_x, diff_y), (diff_x, diff_y)), "constant")

        # Evaluate model at this size
        test_loss, test_acc = model.evaluate(test_x, test_y)
        losses.append(test_loss)
        accuracies.append(test_acc)

    plt.plot(range(7, trained_image_size + 1), losses, label="Loss")
    plt.plot(range(7, trained_image_size + 1), accuracies, label="Accuracy")
    plt.legend()
    plt.show()


def plot_picture_damage():
    model = get_model()

    test_x, test_y = data_func(**kwargs, SAMPLES_PER_CLASS=1, test=True)
    B, N, M, _ = test_x.shape

    radius = 10
    scores = np.zeros((N, M))

    for x in tqdm(range(N)):
        for y in range(M):
            # Silencing a circle
            x_list, y_list = [], []
            for i in range(N):
                for j in range(M):
                    if np.sqrt((i - x) ** 2 + (j - y) ** 2) < radius:
                        x_list.append(i)
                        y_list.append(j)

            x_list = np.array(x_list)
            y_list = np.array(y_list)

            # Setting the circle to zero
            training_data_copy = deepcopy(test_x)
            training_data_copy[:, x_list, y_list] = 0

            # Evaluate model at this size
            test_loss, test_acc = model.evaluate(training_data_copy, test_y, verbose=0)

            scores[x, y] = test_acc

    im = plt.imshow(scores * 100, vmin=0.0, vmax=100)
    cb = plt.colorbar(im, ax=[plt.gca()], location="right")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    train_batches = 200
    train_epochs = 30

    # Make a unique experiment folder name by the time and date
    name = f"{time.localtime().tm_mday}-{time.localtime().tm_mon}-" + str(time.localtime().tm_year)[-2:]
    name += f"_{time.localtime().tm_hour}:{time.localtime().tm_min}"

    # Sometimes, the path is already made. Add a unique numerical suffix
    additive = 2
    new_name = name
    while os.path.isdir(f"./experiments/cnn/mnist3/{new_name}"):
        new_name = name + "_" + str(additive)
        additive += 1

    saved_model = f"./experiments/cnn/mnist3/{new_name}/checkpoint_model.ckpt"

    # Script wide functions
    train_x, train_y = data_func(**kwargs, SAMPLES_PER_CLASS=samples_per_digit)
    test_x, test_y = data_func(**kwargs, SAMPLES_PER_CLASS=samples_per_digit_test, test=True)

    model = CNN(digits=len(CLASSES))

    class EnergyLoss(tf.keras.losses.Loss):
        def __init__(self, rate=0.01):
            super().__init__()
            self.rate = rate

        def call(self, y_true, y_pred):
            return np.sum(y_pred**2) * self.rate

    model.compile(
        optimizers.Adam(lr=1e-4),
        loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=False), EnergyLoss(rate=0.0001)],
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    history = model.fit(train_x, train_y, batch_size=train_batches, epochs=train_epochs)

    test_loss, test_acc = model.evaluate(test_x, test_y)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    plot_history(history)

    model.save_weights(saved_model)
    return test_acc


def confusion_matrix():
    model = get_model()
    test_x, test_y = data_func(**kwargs, SAMPLES_PER_CLASS=samples_per_digit_test, test=True)

    predictions = model.predict(test_x)
    predictions = np.argmax(predictions, axis=1)
    test_y = np.argmax(test_y, axis=1)

    confusion = np.zeros((len(CLASSES), len(CLASSES)))
    for pred, real in zip(predictions, test_y):
        confusion[pred, real] += 1

    confusion /= samples_per_digit_test

    plt.title("CNN confusion matrix")
    sns.heatmap(confusion, annot=True, cmap="plasma")
    plt.ylabel("Guess")
    plt.xlabel("Answer")
    plt.show()


def find_easiest_digits():
    global CLASSES
    global kwargs
    global samples_per_digit

    combinations = get_unique_lists(
        [1, 2, 3, 4, 5, 6, 7, 8], 5
    )  # Removes (automobile/truck 1/9 and plane/bird 0/2 choice)
    print("Found", len(combinations), "combinations")

    accuracies = []
    for i, list_i in enumerate(combinations):
        print(i, ":", list_i)
        CLASSES = list_i
        kwargs = {
            "CLASSES": CLASSES,
            "verbose": False,
            "colors": True if img_channels == 3 else False,
        }

        if "MNIST" in str(data_func):
            samples_per_digit = get_max_samples_balanced(CLASSES=CLASSES, test=False, fashion=fashion)
            print(samples_per_digit)
        elif "CIFAR" in str(data_func):
            samples_per_digit = get_max_samples_balanced_cifar(CLASSES=CLASSES, test=False, colors=kwargs["colors"])
        else:
            print("OOOPS")

        acc = main()
        accuracies.append(acc)
        print("Acc:", acc, "\n")

    scores_dict = {}
    for list_i, acc in zip(combinations, accuracies):
        scores_dict[str(list_i)] = acc

    sorted_dict = dict(sorted(scores_dict.items(), key=lambda item: item[1]))

    for key, item in sorted_dict.items():
        print(key, ":", item)


if __name__ == "__main__":
    plot_image_noise_robustness()
