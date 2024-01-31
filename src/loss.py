import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def pixel_wise_CE_and_energy(img, guesses, expected):
    return pixel_wise_CE(img, guesses, expected) + energy(img, guesses, expected)


def energy(img, guesses, expected):
    return np.sum(img**2) * 0.0001


def pixel_wise_L2_and_CE(img, guesses, expected):
    return pixel_wise_L2(img, guesses, expected) / 2 + pixel_wise_CE(img, guesses, expected) / 2


def pixel_wise_L2(img, guesses, expected):
    """Alternative implementation
    N, M, O = img.shape
    predicted = np.reshape(img, (N * M, O))
    expected = [expected for _ in range(len(predicted))]
    return float(tf.keras.losses.MeanSquaredError()(expected, predicted))"""
    N, M, O = img.shape
    loss = 0
    for guess in np.reshape(img, (N * M, O)):
        loss += np.sum(((np.array(expected) - guess) ** 2)) / (N * M)
    return loss


def pixel_wise_CE(img, guesses, expected):
    N, M, O = img.shape
    predicted = np.reshape(img, (N * M, O))
    expected = [expected for _ in range(len(predicted))]
    return float(tf.keras.losses.CategoricalCrossentropy(from_logits=True)(expected, predicted))


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


def final_guess_wise_L2(img, guesses, expected):
    N, M = img.shape[:2]
    loss = 0
    for guess in guesses:
        loss += np.sum(((np.array(expected) - guess) ** 2)) / (2 * (N) * (M))
    return loss


def scale_loss(loss, datapoints):
    return loss / datapoints


def highest_value(class_images):
    belief = np.mean(class_images, axis=(0, 1))
    return np.argmax(belief)


def highest_vote(class_images):
    votes = np.zeros(class_images.shape[-1])
    for x in range(len(class_images)):
        for y in range(len(class_images[0])):
            votes[np.argmax(class_images[x, y])] += 1
    return np.argmax(votes)


def entropy_seeking_loss(perceptions, image):
    """entropy_img = global_entropy_image(image)
    entropy = 0

    for x in range(len(perceptions)):
        for y in range(len(perceptions[0])):
            x_p, y_p = perceptions[x,y]
            entropy += np.sum(entropy_img[x_p:x_p+3, y_p:y_p+3])"""

    return 0


def neighborhood_histogram(img):
    N, M = img.shape

    hist = {}
    for x in range(N - 2):
        for y in range(M - 2):
            neighborhood = img[x : x + 3, y : y + 3]
            if str(neighborhood) in hist:
                hist[str(neighborhood)] += 1 / (N * M)
            else:
                hist[str(neighborhood)] = 1 / (N * M)

    return hist


def neighborhood_entropy(img):
    img = img * 2 / np.max(img)
    img = np.round(img).astype(int)
    hist = neighborhood_histogram(img)
    N, M = img.shape

    new_img = np.zeros((N - 2, M - 2))
    for x in range(N - 2):
        for y in range(M - 2):
            neighborhood = img[x : x + 3, y : y + 3]
            p_i = hist[str(neighborhood)]
            e = np.log2(1 / p_i)
            new_img[x, y] = e

    return new_img


def neighborhood_information(img):
    N, M = img.shape

    new_img = np.zeros((N - 2, M - 2))
    for x in range(N - 2):
        for y in range(M - 2):
            neighborhood = img[x : x + 3, y : y + 3]
            e = (np.max(neighborhood) - np.min(neighborhood)) ** 2
            new_img[x, y] = e

    return new_img


def information_loss(perceptions, image):
    new_img = neighborhood_information(image)
    new_img = new_img / np.max(new_img)

    loss = 0
    for row in perceptions:
        for x_p, y_p in row:
            loss += 1 - new_img[x_p, y_p]

    loss = loss / (len(perceptions) * len(perceptions[0]))

    return loss


if __name__ == "__main__":
    expected = [0, 1, 0]

    predicted = np.random.rand(10, 10, 3)

    losses = []
    for _ in range(100):
        loss = global_mean_medians(predicted, None, expected)
        losses.append(loss)
        predicted[:, :, 0] -= predicted[:, :, 0] / 10
        predicted[:, :, 1] += (1 - predicted[:, :, 1]) / 10
        predicted[:, :, 2] -= predicted[:, :, 2] / 10

    plt.plot(losses)
    plt.show()
