import functools
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from src.cifar_processing import get_CIFAR_data
from src.mnist_processing import (
    get_max_samples_balanced,
    get_MNIST_data,
    get_MNIST_data_padded,
    get_MNIST_data_resized,
)
from src.utils import get_unique_lists, get_weights_info
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tqdm import tqdm

MNIST_DIGITS = (0, 9)
trained_image_size = 32
img_channels = 3
fashion = False

saved_model = "./experiments/cnn/checkpoint_model_cifar.ckpt"

# Data
data_func = get_CIFAR_data
kwargs = {
    "MNIST_DIGITS": MNIST_DIGITS,
    "verbose": False,
    "colors": True if img_channels == 3 else False,
}

samples_per_digit = get_max_samples_balanced(MNIST_DIGITS=MNIST_DIGITS, test=False, fashion=fashion)

samples_per_digit_test = get_max_samples_balanced(MNIST_DIGITS=MNIST_DIGITS, test=True, fashion=fashion)


class CNN(tf.keras.Model):
    def __init__(self, digits=5, img_dim=(trained_image_size, trained_image_size, img_channels)):
        super().__init__()

        self.dmodel = tf.keras.Sequential(
            [
                Conv2D(3, 3, padding="valid", input_shape=img_dim, activation="linear"),
                Conv2D(3, 3, padding="valid", activation="linear"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(3, 3, padding="valid", activation="linear"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(digits),
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
    plt.plot(history.history["loss"])
    plt.plot(history.history["categorical_accuracy"])
    plt.show()


def resize_for_CNN(size, mnist_digits):
    # First, get normal resized data that the MovingNCA could handle well
    data_func = get_MNIST_data_resized
    test_x, test_y = data_func(
        MNIST_DIGITS=mnist_digits, SAMPLES_PER_DIGIT=samples_per_digit_test, size=size, verbose=False, test=True
    )

    # Then, resize to again be 56x56, with whatever information loss was incurred
    resized_x_data = []
    for img in test_x:
        resized_x_data.append(
            cv2.resize(img, (trained_image_size, trained_image_size), interpolation=cv2.INTER_NEAREST)
        )

    return np.array(resized_x_data), test_y


def get_model():
    model = CNN(digits=len(MNIST_DIGITS))
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
        test_x, test_y = resize_for_CNN(size, MNIST_DIGITS)

        # Evaluate model at this size
        test_loss, test_acc = model.evaluate(test_x, test_y)
        losses.append(test_loss)
        accuracies.append(test_acc)

    plt.plot(range(start_size, trained_image_size + 1), losses, label="Loss")
    plt.plot(range(start_size, trained_image_size + 1), accuracies, label="Accuracy")
    plt.legend()
    plt.show()


def plot_weights_dense():
    model = get_model()

    dense_layer = np.array(model.weights[0]).T
    print(dense_layer.shape)

    for i in range(len(MNIST_DIGITS)):
        reshapen = np.reshape(dense_layer[i], (trained_image_size, trained_image_size))
        plt.subplot(1, len(MNIST_DIGITS), i + 1)
        plt.imshow(reshapen)

    plt.show()


def plot_weights_conv():
    model = get_model()

    first_conv_layer = np.array(model.weights[0]).T

    for i in range(len(first_conv_layer)):
        plt.subplot(4, len(MNIST_DIGITS), i + 1)
        plt.imshow(first_conv_layer[i, 0])

    second_conv_layer = np.array(model.weights[2]).T

    for i in range(len(second_conv_layer)):
        plt.subplot(4, len(MNIST_DIGITS), i + 1 + len(MNIST_DIGITS))
        plt.imshow(second_conv_layer[i])

    third_conv_layer = np.array(model.weights[4]).T

    for i in range(len(third_conv_layer)):
        plt.subplot(4, len(MNIST_DIGITS), i + 1 + len(MNIST_DIGITS) * 2)
        plt.imshow(third_conv_layer[i])

    dense_layer = np.array(model.weights[6]).T

    for i in range(len(MNIST_DIGITS)):
        reshapen = np.reshape(dense_layer[i], (5, 5, 3))
        plt.subplot(4, len(MNIST_DIGITS), i + 1 + len(MNIST_DIGITS) * 3)
        plt.imshow(reshapen)

    plt.show()


def plot_feature_maps():
    model = get_model()

    middle_images = []
    image = data_func(MNIST_DIGITS=MNIST_DIGITS, SAMPLES_PER_DIGIT=samples_per_digit_test, verbose=False, test=True)[0][
        0
    ]

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    middle_images.append(image)

    answer = None
    for layer in model.dmodel.layers:
        image = layer(image)
        print(layer.name)
        if layer.name == "dense":
            answer = image
        elif layer.name != "flatten":
            middle_images.append(image)

    for img in middle_images:
        plt.subplot(1, len(middle_images), middle_images.index(img) + 1)
        plt.imshow(img[0])

    print("Answer:", answer[0])
    plt.show()


def plot_image_scale():
    model = get_model()

    losses = []
    accuracies = []

    for size in range(7, trained_image_size + 1):
        # Get data
        test_x, test_y = get_MNIST_data_resized(
            MNIST_DIGITS=MNIST_DIGITS, SAMPLES_PER_DIGIT=samples_per_digit_test, size=size, verbose=False, test=True
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

    test_x, test_y = data_func(**kwargs, SAMPLES_PER_DIGIT=samples_per_digit_test, test=True)
    B, N, M = test_x.shape

    radius = 5
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

    plt.imshow(scores)
    plt.show()


def main():
    train_epochs = 100
    train_batches = 200

    # Script wide functions
    train_x, train_y = data_func(**kwargs, SAMPLES_PER_DIGIT=samples_per_digit)
    test_x, test_y = data_func(**kwargs, SAMPLES_PER_DIGIT=samples_per_digit_test, test=True)

    model = CNN(digits=len(MNIST_DIGITS))
    model.compile(
        optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
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
    test_x, test_y = data_func(**kwargs, SAMPLES_PER_DIGIT=samples_per_digit_test, test=True)

    predictions = model.predict(test_x)
    predictions = np.argmax(predictions, axis=1)
    test_y = np.argmax(test_y, axis=1)

    confusion = np.zeros((len(MNIST_DIGITS), len(MNIST_DIGITS)))
    for pred, real in zip(predictions, test_y):
        confusion[pred, real] += 1

    confusion /= samples_per_digit_test

    plt.title("CNN confusion matrix")
    sns.heatmap(confusion, annot=True, cmap="plasma")
    plt.ylabel("Guess")
    plt.xlabel("Answer")
    plt.show()


def find_easiest_digits():
    global MNIST_DIGITS
    global kwargs
    global samples_per_digit

    combinations = get_unique_lists([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5)
    print("Found", len(combinations), "combinations")

    accuracies = []
    for i, list_i in enumerate(combinations):
        print(i, ":", list_i)
        MNIST_DIGITS = list_i
        kwargs = {
            "MNIST_DIGITS": MNIST_DIGITS,
            "verbose": False,
        }

        samples_per_digit = get_max_samples_balanced(MNIST_DIGITS=MNIST_DIGITS, test=False, fashion=fashion)

        acc = main()
        accuracies.append(acc)

    scores_dict = {}
    for list_i, acc in zip(combinations, accuracies):
        scores_dict[str(list_i)] = acc

    sorted_dict = dict(sorted(scores_dict.items(), key=lambda item: item[1]))

    for key, item in sorted_dict.items():
        print(key, ":", item)


if __name__ == "__main__":
    find_easiest_digits()
