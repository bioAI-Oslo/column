import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from src.mnist_processing import (
    get_MNIST_data,
    get_MNIST_data_padded,
    get_MNIST_data_resized,
)
from src.utils import get_weights_info
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.dmodel = tf.keras.Sequential(
            [
                Conv2D(3, 5, padding="valid", input_shape=(56, 56, 1), activation="linear"),  # 56 -> 52
                Conv2D(3, 5, padding="valid", activation="linear"),  # 52 -> 48
                MaxPooling2D(pool_size=(2, 2)),  # 48 -> 24
                Conv2D(3, 3, padding="valid", activation="linear"),  # 24 -> 22
                MaxPooling2D(pool_size=(2, 2)),  # 22 -> 11
                Flatten(),
                Dense(3),
            ]
        )

        out = self(tf.zeros([1, 56, 56]))  # dummy calls to build the model

        _, _, weight_amount = get_weights_info(self.weights)
        print("Weights:", int(weight_amount))

    # @tf.function
    def call(self, x):
        return self.dmodel(x)


# Digit 0 has 5923 samples
# Digit 1 has 6742 samples
# Digit 2 has 5958 samples
# Digit 3 has 6131 samples
# Digit 4 has 5842 samples
# Digit 5 has 5421 samples # Smallest sample size
# Digit 6 has 5918 samples
# Digit 7 has 6265 samples
# Digit 8 has 5851 samples
# Digit 9 has 5949 samples

# And for test set:
# Digit 0 has 980 samples
# Digit 1 has 1135 samples
# Digit 2 has 1032 samples
# Digit 3 has 1010 samples
# Digit 4 has 982 samples
# Digit 5 has 892 samples # Smallest sample size
# Digit 6 has 958 samples
# Digit 7 has 1028 samples
# Digit 8 has 974 samples
# Digit 9 has 1009 samples


def plot_history():
    plt.plot(history.history["loss"])
    plt.plot(history.history["categorical_accuracy"])
    plt.show()


def plot_image_scale():
    model = CNN()
    model.compile(
        optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    model.load_weights("./experimnets/cnn/checkpoint_model_28trained.ckpt")

    image_size = 56

    # Data
    data_func = get_MNIST_data_resized
    MNIST_DIGITS = (0, 1, 2)
    SAMPLES_PER_DIGIT = 5400

    losses = []
    accuracies = []

    for size in range(7, image_size + 1):
        test_x, test_y = data_func(
            MNIST_DIGITS=MNIST_DIGITS, SAMPLES_PER_DIGIT=892, size=size, verbose=False, test=True
        )

        diff = image_size - size
        diff_x, diff_y = int(np.floor(diff / 2)), int(np.ceil(diff / 2))

        test_x = np.pad(test_x, ((0, 0), (diff_x, diff_y), (diff_x, diff_y)), "constant")

        if size == 28 or size == 56:
            plt.imshow(test_x[np.random.randint(0, len(test_x))])
            plt.show()

        test_loss, test_acc = model.evaluate(test_x, test_y)
        losses.append(test_loss)
        accuracies.append(test_acc)

    plt.plot(range(7, image_size + 1), losses, label="Loss")
    plt.plot(range(7, image_size + 1), accuracies, label="Accuracy")
    plt.legend()
    plt.show()


def main():
    train_epochs = 20
    train_batches = 200

    # Data
    data_func = get_MNIST_data_padded
    MNIST_DIGITS = (0, 1, 2)
    SAMPLES_PER_DIGIT = 5400

    # Script wide functions
    train_x, train_y = data_func(MNIST_DIGITS=MNIST_DIGITS, SAMPLES_PER_DIGIT=SAMPLES_PER_DIGIT, verbose=False)
    test_x, test_y = data_func(MNIST_DIGITS=MNIST_DIGITS, SAMPLES_PER_DIGIT=892, verbose=False, test=True)

    model = CNN()
    model.compile(
        optimizers.Adam(lr=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    history = model.fit(train_x, train_y, batch_size=train_batches, epochs=train_epochs)

    test_loss, test_acc = model.evaluate(test_x, test_y)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

    model.save_weights("./experimnets/cnn/checkpoint_model_28trained.ckpt")


if __name__ == "__main__":
    plot_image_scale()
