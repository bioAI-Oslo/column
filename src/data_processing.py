import random
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10, fashion_mnist, mnist
from src.utils import shuffle, translate

# Global storage of the dataset to make fetching data faster
sorted_X_train = None
sorted_X_test = None
loaded_classes = None
loaded_data_name = None


def get_labels(data_func, classes):
    """Returns the names of the classes for the dataset"""
    # Taking specific care with the data functions
    if data_func == get_CIFAR_data:
        possibles = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
        return [possibles[i] for i in classes]
    elif data_func == get_MNIST_fashion_data:
        possibles = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
        return [possibles[i] for i in classes]
    else:
        return [str(i) for i in classes]


def get_max_samples_balanced(data_func, **kwargs):
    """A Function to find the largest dataset with balanced classes that can be with data_func and CLASSES"""
    # To ensure nothing breaks, load a little dataset to initialize the data like normal.
    # This means the dataset is loaded
    kwargs_copied = deepcopy(kwargs)
    kwargs_copied["SAMPLES_PER_CLASS"] = 1
    data_func(**kwargs_copied)

    # Then, check the smallest amount of data available
    sorted_X = sorted_X_test if kwargs["test"] else sorted_X_train
    min_len = min(len(sorted_X[i]) for i in range(len(kwargs["CLASSES"])))

    # And return that for a balanced dataset
    return min_len


def get_MNIST_data_padded(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False):
    training_data, target_data = get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, digits=True)

    diff_x, diff_y = 14, 14
    training_data = np.pad(training_data, ((0, 0), (diff_x, diff_y), (diff_x, diff_y), (0, 0)), "constant")

    return training_data, target_data


def get_MNIST_data_resized(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, size=56, verbose=False, test=False):
    training_data, target_data = get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, digits=True)

    resized_x_data = []
    for img in training_data:
        resized_x_data.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))

    return np.array(resized_x_data), target_data


def get_MNIST_data_translated(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False):
    training_data, target_data = get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, digits=True)

    training_data = translate(training_data, new_length=(70, 70))

    return training_data, target_data


def get_MNIST_fashion_data(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False):
    return get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, fashion=True)


def get_CIFAR_data(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False, colors=False):
    return get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, cifar=True, colors=colors)


def get_MNIST_data(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False):
    return get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, digits=True)


def get_data(
    CLASSES=(3, 4),
    SAMPLES_PER_CLASS=10,
    verbose=False,
    test=False,
    digits=False,
    fashion=False,
    cifar=False,
    colors=False,
):
    global sorted_X_train
    global sorted_X_test
    global loaded_classes
    global loaded_data_name

    name = ("CIFAR" if colors else "CIFAR gray") if cifar else "MNIST" if fashion is False else "Fashion MNIST"

    # Do the loaded digits correspond to the CLASSES?
    if loaded_classes is None or CLASSES != loaded_classes or loaded_data_name != name:
        # Reload
        loaded_data_name = name
        loaded_classes = CLASSES
        sorted_X_train = None
        sorted_X_test = None
        if verbose:
            print(f"Loading new {name} data")

    if verbose:
        print("Getting", "training" if not test else "test", name, "data")
    if not test and (sorted_X_train is None):
        if verbose:
            print(f"Initializing {name} training data")
        sorted_X_train = initalize_reduced_data(
            CLASSES, test=False, digits=digits, fashion=fashion, cifar=cifar, colors=colors
        )
    elif test and (sorted_X_test is None):
        if verbose:
            print(f"Initializing {name} test data")
        sorted_X_test = initalize_reduced_data(
            CLASSES, test=True, digits=digits, fashion=fashion, cifar=cifar, colors=colors
        )

    sorted_X = sorted_X_train if not test else sorted_X_test

    N_digits = len(CLASSES)

    # Getting random samples of every digit
    train_X = []
    train_y = []
    for i in range(N_digits):
        one_hot = [1.0 if x == i else 0.0 for x in range(N_digits)]
        for _ in range(SAMPLES_PER_CLASS):
            index = random.randrange(len(sorted_X[i]))
            train_X.append(sorted_X[i][index])
            train_y.append(one_hot)

    training_data, target_data = shuffle(train_X, train_y)

    if verbose:
        print("Returning the training set")
    return training_data, target_data


def initalize_reduced_data(CLASSES=(3, 4), test=False, digits=False, fashion=False, cifar=False, colors=False):
    # Loading
    if cifar:
        data = cifar10.load_data()
        (train_X, train_y), (test_X, test_y) = data
        x = train_X if not test else test_X
        y = train_y if not test else test_y

        # This data has shape (len(y), 1), so reshaping it to accomodate the following code.
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
    else:
        if fashion:
            (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
        else:
            (train_X, train_y), (test_X, test_y) = mnist.load_data()
        x = train_X if not test else test_X
        y = train_y if not test else test_y

        # Reshaping it to accomodate the network
        N, M = x[0].shape
        x = x.reshape((len(x), N, M, 1))

    # Scaling to [0,1]
    # NB: If scaling by training specific data, use training scaler for test data
    x_scaled = x / 255

    # get indexes of digits to include
    where_digits = []
    for digit in CLASSES:
        where_digits.append(np.where(y == digit))

    # Making x-lists of every digit
    sorted_X_internal = []
    for i in range(len(CLASSES)):
        sorted_X_internal.append(x_scaled[where_digits[i]])

    return sorted_X_internal


def _test_dataset_func(data_func, kwargs):
    labels = get_labels(data_func, kwargs["CLASSES"])
    X_data, y_data = data_func(**kwargs)
    print("\n\n\nShapes", X_data.shape, y_data.shape, "\n\n\n")

    for img, lab in zip(X_data, y_data):
        label = labels[np.argmax(lab)]
        plt.figure()
        plt.imshow(img)
        plt.title(str(lab) + " " + label)

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
    data_func = get_MNIST_data
    kwargs = {
        "CLASSES": (0, 1, 2, 8, 9),
        "SAMPLES_PER_CLASS": 3,
        "verbose": True,
        "test": True,
    }

    print(get_max_samples_balanced(data_func, **kwargs))
    _test_dataset_func(data_func, kwargs)

    kwargs["CLASSES"] = (3, 4, 5, 6, 7)
    print(get_max_samples_balanced(data_func, **kwargs))
    _test_dataset_func(data_func, kwargs)

    data_func = get_MNIST_fashion_data
    print(get_max_samples_balanced(data_func, **kwargs))
    _test_dataset_func(data_func, kwargs)

    data_func = get_CIFAR_data
    kwargs["colors"] = False
    print(get_max_samples_balanced(data_func, **kwargs))
    _test_dataset_func(data_func, kwargs)

    kwargs["colors"] = True
    print(get_max_samples_balanced(data_func, **kwargs))
    _test_dataset_func(data_func, kwargs)
