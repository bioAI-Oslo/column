import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist, mnist
from src.utils import shuffle, translate

sorted_X_train = None
sorted_X_test = None


def get_MNIST_data_padded(MNIST_DIGITS=(3, 4), SAMPLES_PER_DIGIT=10, verbose=False, test=False):
    training_data, target_data = get_MNIST_data(MNIST_DIGITS, SAMPLES_PER_DIGIT, verbose, test)

    diff_x, diff_y = 14, 14
    training_data = np.pad(training_data, ((0, 0), (diff_x, diff_y), (diff_x, diff_y)), "constant")

    return training_data, target_data


def get_MNIST_data_resized(MNIST_DIGITS=(3, 4), SAMPLES_PER_DIGIT=10, size=56, verbose=False, test=False):
    training_data, target_data = get_MNIST_data(MNIST_DIGITS, SAMPLES_PER_DIGIT, verbose, test)

    resized_x_data = []
    for img in training_data:
        resized_x_data.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))

    return np.array(resized_x_data), target_data


def get_MNIST_data_translated(MNIST_DIGITS=(3, 4), SAMPLES_PER_DIGIT=10, verbose=False, test=False):
    training_data, target_data = get_MNIST_data(MNIST_DIGITS, SAMPLES_PER_DIGIT, verbose, test)

    training_data = translate(training_data, new_length=(70, 70))

    return training_data, target_data


def get_max_samples_balanced(MNIST_DIGITS, test=False, fashion=False):
    # To ensure nothing breaks, load a little dataset to initialize the data like normal.
    get_MNIST_data(MNIST_DIGITS=MNIST_DIGITS, SAMPLES_PER_DIGIT=1, verbose=False, test=test, fashion=fashion)

    # Then, check the smallest amount of data available
    if test:
        min_len = min(len(sorted_X_test[i]) for i in range(len(MNIST_DIGITS)))
    else:
        min_len = min(len(sorted_X_train[i]) for i in range(len(MNIST_DIGITS)))

    # And return that for a balanced dataset
    return min_len


def get_MNIST_fashion_data(MNIST_DIGITS=(3, 4), SAMPLES_PER_DIGIT=10, verbose=False, test=False):
    return get_MNIST_data(MNIST_DIGITS, SAMPLES_PER_DIGIT, verbose, test, fashion=True)


def get_MNIST_data(MNIST_DIGITS=(3, 4), SAMPLES_PER_DIGIT=10, verbose=False, test=False, fashion=False):
    global sorted_X_train
    global sorted_X_test
    if verbose:
        print("Getting", "training" if not test else "test", "data")
    if not test and sorted_X_train is None:
        if verbose:
            print("Initializing MNIST training data")
        sorted_X_train = initalize_MNIST_reduced_digits(MNIST_DIGITS, test=False, fashion=fashion)
    elif test and sorted_X_test is None:
        if verbose:
            print("Initializing MNIST test data")
        sorted_X_test = initalize_MNIST_reduced_digits(MNIST_DIGITS, test=True, fashion=fashion)

    sorted_X = sorted_X_train if not test else sorted_X_test

    N_digits = len(MNIST_DIGITS)

    # Getting random samples of every digit
    train_X = []
    train_y = []
    for i in range(N_digits):
        one_hot = [1.0 if x == i else 0.0 for x in range(N_digits)]
        for _ in range(SAMPLES_PER_DIGIT):
            index = random.randrange(len(sorted_X[i]))
            train_X.append(sorted_X[i][index])
            train_y.append(one_hot)

    training_data, target_data = shuffle(train_X, train_y)

    if verbose:
        print("Returning the training set")
    return training_data, target_data


def initalize_MNIST_reduced_digits(MNIST_DIGITS=(3, 4), test=False, fashion=False):
    # Loading
    if fashion:
        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    else:
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
    x = train_X if not test else test_X
    y = train_y if not test else test_y

    # Scaling to [0,1]
    # NB: If scaling by training specific data, use training scaler for test data
    x_scaled = x / 255

    # get indexes of digits to include
    where_digits = []
    for digit in MNIST_DIGITS:
        where_digits.append(np.where(y == digit))

    # Making x-lists of every digit
    sorted_X_internal = []
    for i in range(len(MNIST_DIGITS)):
        sorted_X_internal.append(x_scaled[where_digits[i]])

    return sorted_X_internal


def _test_MNIST_dataset():
    X_data, y_data = get_MNIST_data(MNIST_DIGITS=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), SAMPLES_PER_DIGIT=3, verbose=True)

    for img, lab in zip(X_data, y_data):
        plt.figure()
        plt.imshow(img)
        plt.title(str(lab))

    plt.show()


def _test_MNIST_dataset_time():
    start_time = time.time()
    X_data, y_data = get_MNIST_data(MNIST_DIGITS=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), SAMPLES_PER_DIGIT=10)
    print("Initial load:", time.time() - start_time)

    start_time = time.time()
    X_data, y_data = get_MNIST_data(MNIST_DIGITS=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), SAMPLES_PER_DIGIT=10)
    print("Subsequent load:", time.time() - start_time)

    times = 0
    N_times = 100
    for _ in range(N_times):
        start_time = time.time()
        X_data, y_data = get_MNIST_data(MNIST_DIGITS=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), SAMPLES_PER_DIGIT=10)
        times += time.time() - start_time
    print("Average time after load:", times / N_times)


if __name__ == "__main__":
    np.random.seed(4)
    images, labels = get_MNIST_data_resized(
        MNIST_DIGITS=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), SAMPLES_PER_DIGIT=1, size=15, verbose=False, test=False
    )

    for img, lab in zip(images, labels):
        plt.figure()
        plt.imshow(img)
        plt.title(str(lab))

    plt.show()
