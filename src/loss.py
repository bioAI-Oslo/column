import numpy as np
import tensorflow as tf


def pixel_wise_L2(img, guesses, expected):
    N, M, O = img.shape
    loss = 0
    for guess in np.reshape(img, (N * M, O)):
        loss += np.sum(((np.array(expected) - guess) ** 2)) / (N * M)
    return loss


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
    expected = np.array([0, 0, 1, 0, 0])

    loss_small = pixel_wise_L2(np.ones((5, 5, 5)), None, expected) * 25.0 / 25.0
    loss_big = pixel_wise_L2(np.ones((26, 26, 5)), None, expected) * 25.0 / 25.0
    # They are the same: 4.000000000000001 3.999999999999942
    print(loss_small, loss_big)
    """from gen_images import get_MNIST_data
    import matplotlib.pyplot as plt

    perceptions = [[[28//4, 28//4], [28//4, 28*3//4]],[[28*3//4, 28//4], [28*3//4, 28*3//4]]]

    highlight = [[[0,0,0,0] for _ in range(28)] for _ in range(28)]
    for row in perceptions:
        for (x_p, y_p) in row:
            highlight[x_p][y_p] = [1,0,1,1]

    images, labels = get_MNIST_data(MNIST_DIGITS=(0,1,2,3,4,5,6,7,8,9), SAMPLES_PER_DIGIT=2, verbose=False)
    for i in range(20):
        img = images[i]

        new_img = neighborhood_information(img)
        print(information_loss(perceptions, img))

        plt.subplot(1,2,1)
        plt.imshow(img, cmap="gray")
        plt.imshow(highlight)
        plt.subplot(1,2,2)
        plt.imshow(new_img, cmap="gray")
        plt.imshow(highlight)
        plt.show()"""
