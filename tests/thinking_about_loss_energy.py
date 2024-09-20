import numpy as np
from matplotlib import pyplot as plt


def energy(img, guesses, expected, regulator=0.0001):
    N, M, O = img.shape[-3:]
    # Any size of cortex and any amount of classes will yield the equivalent of 0.0001 for a 15*15 cortex with 5 classes
    # For historic reasons, sadly. Just consider it a fancy value instead of 0.0001
    # regulator = 225 / (N * M) * 0.0001 # Previous regulator, used on most of the currently aquired data. Same as the one below for all 5 class data
    # Batch approved
    if len(img.shape) == 4:
        B = img.shape[0]
        return (np.sum(img**2) * regulator) / B
    elif len(img.shape) == 3:
        # We scale outside because the batch size is currently unknown
        return np.sum(img**2) * regulator
    else:
        print("oopsie")


def pixel_wise_CE_and_energy(img, guesses, expected):
    # Batch approved
    return pixel_wise_CE(img, guesses, expected) + energy(img, guesses, expected)


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


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def CE(y_pred, expected):
    return -np.sum(expected * np.log(softmax(y_pred) + 1e-10), axis=-1)


def pixel_wise_CE(img, guesses, expected):

    # Batch approved
    expected, predicted = get_expected_and_predicted(img, expected)

    return np.mean(CE(predicted, expected))


if __name__ == "__main__":
    N, M = 28, 28

    CE_values = []
    energy_values = []

    to_test = list(range(2, 10 + 1))

    for C in to_test:

        ideal_i = 0
        ideal_total = 1000
        ideal_energy = 1000
        ideal_CE = 1000
        for i in np.linspace(0, 10, 100):
            img = np.zeros((N, M, C))
            img[:, :, 0] = i / 2
            img[:, :, 1:] = -i / 2
            expected = np.array([1 if i == 0 else 0 for i in range(C)])

            CE_value = pixel_wise_CE(img, None, expected)
            energy_value_C = energy(img, None, expected, regulator=1125 / (N * M * C) * 0.0001)

            loss_total = CE_value + energy_value_C
            if loss_total < ideal_total:
                ideal_i = i
                ideal_total = loss_total
                ideal_energy = energy_value_C
                ideal_CE = CE_value

        print(f"ideal i: {ideal_i}, CE: {ideal_CE}, energy: {ideal_energy}")
        CE_values.append(ideal_CE)
        energy_values.append(ideal_energy)

    plt.plot(to_test, CE_values, label="CE")
    plt.plot(to_test, energy_values, label="Energy")
    plt.plot(to_test, np.array(CE_values) + np.array(energy_values), label="Total")
    plt.legend()
    plt.show()
