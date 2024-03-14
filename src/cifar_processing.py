import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from utils import shuffle

sorted_X_train = None
sorted_X_test = None


def get_max_samples_balanced_cifar(MNIST_DIGITS, test=False):
    # To ensure nothing breaks, load a little dataset to initialize the data like normal.
    get_CIFAR_data(MNIST_DIGITS=MNIST_DIGITS, SAMPLES_PER_DIGIT=1, verbose=False, test=test)

    # Then, check the smallest amount of data available
    if test:
        min_len = min(len(sorted_X_test[i]) for i in range(len(MNIST_DIGITS)))
    else:
        min_len = min(len(sorted_X_train[i]) for i in range(len(MNIST_DIGITS)))

    # And return that for a balanced dataset
    return min_len


def get_CIFAR_data(MNIST_DIGITS=(3, 4), SAMPLES_PER_DIGIT=10, verbose=False, test=False, colors=False):
    global sorted_X_train
    global sorted_X_test
    if verbose:
        print("Getting", "training" if not test else "test", "data")
    if not test and sorted_X_train is None:
        if verbose:
            print("Initializing CIFAR training data")
        sorted_X_train = initalize_CIFAR_reduced_classes(MNIST_DIGITS, test=False, colors=colors)
    elif test and sorted_X_test is None:
        if verbose:
            print("Initializing CIFAR test data")
        sorted_X_test = initalize_CIFAR_reduced_classes(MNIST_DIGITS, test=True, colors=colors)

    sorted_X = sorted_X_train if not test else sorted_X_test

    N_classes = len(MNIST_DIGITS)

    # Getting random samples of every class
    train_X = []
    train_y = []
    for i in range(N_classes):
        one_hot = [1.0 if x == i else 0.0 for x in range(N_classes)]
        for _ in range(SAMPLES_PER_DIGIT):
            index = random.randrange(len(sorted_X[i]))
            train_X.append(sorted_X[i][index])
            train_y.append(one_hot)
            # if verbose:
            #    print(index, "out of", len(sorted_X[i]))

    training_data, target_data = shuffle(train_X, train_y)

    if verbose:
        print("Returning the training set")
    return training_data, target_data


def initalize_CIFAR_reduced_classes(MNIST_DIGITS=(3, 4), test=False, colors=False):
    # Loading
    data = cifar10.load_data()
    (train_X, train_y), (test_X, test_y) = data
    x = train_X if not test else test_X
    y = train_y if not test else test_y

    # Thsi data has shape (len(y), 1), so reshaping it to accomodate the following code.
    y = np.reshape(y, (len(y)))

    # Converting to grayscale
    if not colors:
        x_list = []
        for img in x:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            N, M = img.shape
            img = img.reshape((N, M, 1))
            x_list.append(img)
        x = np.array(x_list)

    # Scaling to [0,1]
    # NB: If scaling by training specific data, use training scaler for test data
    x_scaled = x / 255

    # get indexes of digits to include
    where_classes = []
    for class_i in MNIST_DIGITS:
        where_classes.append(np.where(y == class_i))

    # Making x-lists of every digit
    sorted_X_internal = []
    for i in range(len(MNIST_DIGITS)):
        sorted_X_internal.append(x_scaled[where_classes[i]])

    return sorted_X_internal


def _test_dataset_func(data_func, kwargs):
    X_data, y_data = data_func(**kwargs)

    for img, lab in zip(X_data, y_data):
        print("Shape image:", img.shape)
        plt.figure()
        plt.imshow(img)
        plt.title(str(lab))

    plt.show()


def _test_dataset_func_time(data_func, kwargs):
    start_time = time.time()
    X_data, y_data = data_func(**kwargs)
    print("Initial load:", time.time() - start_time)

    start_time = time.time()
    X_data, y_data = data_func(**kwargs)
    print("Subsequent load:", time.time() - start_time)

    times = 0
    N_times = 100
    for _ in range(N_times):
        start_time = time.time()
        X_data, y_data = data_func(**kwargs)
        times += time.time() - start_time
    print("Average time after load:", times / N_times)


if __name__ == "__main__":
    data_func = get_CIFAR_data
    kwargs = {
        "MNIST_DIGITS": (0, 1, 2, 8, 9),
        "SAMPLES_PER_DIGIT": 3,
        "verbose": False,
        "test": False,
        "colors": True,
    }
    _test_dataset_func(data_func, kwargs)
