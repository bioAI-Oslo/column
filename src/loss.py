import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from numba import jit


def pixel_wise_CE_and_energy(img, guesses, expected):
    # Batch approved
    return pixel_wise_CE(img, guesses, expected) + energy(img, guesses, expected)


def energy(img, guesses, expected):
    N, M, O = img.shape[-3:]
    # Any size of cortex and any amount of classes will yield the equivalent of 0.0001 for a 15*15 cortex with 5 classes
    # For historic reasons, sadly. Just consider it a fancy value instead of 0.0001
    # regulator = 225 / (N * M) * 0.0001 # Previous regulator, used on most of the currently aquired data. Same as the one below for all 5 class data
    regulator = 1125 / (N * M * O) * 0.0001

    # Batch approved
    if len(img.shape) == 4:
        B = img.shape[0]
        return (np.sum(img**2) * regulator) / B
    elif len(img.shape) == 3:
        # We scale outside because the batch size is currently unknown
        return np.sum(img**2) * regulator
    else:
        print("oopsie")


def pixel_wise_L2_and_CE(img, guesses, expected):
    # Batch approved
    return pixel_wise_L2(img, guesses, expected) / 2 + pixel_wise_CE(img, guesses, expected) / 2


def get_expected_and_predicted(img, expected):
    # Batch approved
    if len(img.shape) == 4:
        B, N, M, O = img.shape
        predicted = np.reshape(img, (B * N * M, O))
        out = np.empty((B * N * M, O), expected[0].dtype)
        for b in range(B):
            out[b * N * M : (b + 1) * N * M] = expected[b]
        expected = out
    elif len(img.shape) == 3:
        N, M, O = img.shape
        predicted = np.reshape(img, (N * M, O))
        out = np.empty((N * M, O), expected.dtype)
        expected = [expected for _ in range(len(predicted))]
    else:
        print("oopsie")
    return expected, predicted


def pixel_wise_L2(img, guesses, expected):
    # Batch approved
    expected, predicted = get_expected_and_predicted(img, expected)
    return float(tf.keras.losses.MeanSquaredError()(expected, predicted))


def pixel_wise_CE(img, guesses, expected):
    # Batch approved
    expected, predicted = get_expected_and_predicted(img, expected)
    return float(
        tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO)(
            expected, predicted
        )
    )
    """B, N, M, O = img.shape
    loss = 0
    for b in range(B):
        loss += float(
            tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO)(
                expected[b * N * M : (b + 1) * N * M], predicted[b * N * M : (b + 1) * N * M]
            )
        )
    return loss / len(img)"""


def global_mean_medians(img, guesses, expected):
    N, M, O = img.shape
    length = N * M
    interval_start = length // 2 - length // 6
    interval_end = length // 2 + length // 6
    mean_medians = []
    for channel in range(O):
        channel_values = img[:, :, channel].flatten()
        channel_values = np.sort(channel_values)
        interval = channel_values[interval_start:interval_end]
        mean_medians.append(np.mean(interval))

    return np.sum(((np.array(expected) - np.array(mean_medians)) ** 2))


def scale_loss(loss, datapoints):
    # Batch approved
    return loss / datapoints


def highest_value(class_images):
    # Batch approved
    belief = np.mean(class_images, axis=(-3, -2))
    numerical_belief = np.argmax(belief, axis=-1)
    return numerical_belief


def highest_vote(class_images):
    votes = np.zeros(class_images.shape[-1])
    for x in range(len(class_images)):
        for y in range(len(class_images[0])):
            votes[np.argmax(class_images[x, y])] += 1
    return np.argmax(votes)


if __name__ == "__main__":
    loss_to_test = pixel_wise_CE_and_energy

    expected = [0, 1, 0]
    predicted = np.random.rand(10, 10, 3)

    losses = []
    for _ in range(100):
        loss = loss_to_test(predicted, None, expected)
        losses.append(loss)
        predicted[:, :, 0] -= predicted[:, :, 0] / 10
        predicted[:, :, 1] += (1 - predicted[:, :, 1]) / 10
        predicted[:, :, 2] -= predicted[:, :, 2] / 10

    plt.plot(losses)
    plt.show()
