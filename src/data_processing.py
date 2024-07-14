import random
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10, fashion_mnist, mnist
from src.plotting_utils import get_plotting_ticks
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
    elif data_func == get_simple_object:
        possibles = ["Cup", "Knife", "Bowl"]
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


def get_simple_pattern(verbose=False, **kwargs):
    """Only for quick testing"""

    if verbose:
        print("Generating simple pattern")

    class_0 = np.zeros((10, 10, 1))
    class_1 = np.zeros((10, 10, 1))

    class_0[[i for i in range(0, 10, 2)], :, 0] = 1.0
    class_0[:, [i for i in range(0, 10, 2)], 0] = 1.0

    class_1[[i for i in range(0, 10, 2)], ::2, 0] = 1.0
    class_1[1::2, [i for i in range(1, 10, 2)], 0] = 1.0

    training_data = np.array([class_0, class_1])
    target_data = np.array([[1, 0], [0, 1]], dtype=np.float32)

    return training_data, target_data


def get_simple_object(verbose=False, size=18, **kwargs):
    """Only for quick testing"""

    def mug_image_gen(N, M):
        mug_image = np.zeros((N, M, 1))

        mug_image[3 * N // 10 : 8 * N // 10, 2 * M // 10 : 6 * M // 10, 0] = 1.0
        mug_image[4 * N // 10 : 7 * N // 10, 6 * M // 10 : 8 * M // 10, 0] = 1.0
        mug_image[5 * N // 10 : 6 * N // 10, 6 * M // 10 : 7 * M // 10, 0] = 0.0

        return mug_image

    def bowl_image_gen(N, M):
        bowl_image = np.zeros((N, M, 1))

        r_i = 2 * N // 10
        r_o = 4 * N // 10
        for x in range(N // 2, N):
            for y in range(M):
                if r_i**2 <= (x - N / 2) ** 2 + (y - M / 2) ** 2 <= r_o**2:
                    bowl_image[x - N // 4, y, 0] = 1.0

        return bowl_image

    def knife_image_gen(N, M):
        knife_image = np.zeros((N, M, 1))

        knife_image[1 * N // 10 : 9 * N // 10, 5 * M // 10 : 6 * M // 10, 0] = 1.0
        knife_image[1 * N // 10 : 6 * N // 10, 4 * M // 10 : 6 * M // 10, 0] = 1.0

        return knife_image

    if verbose:
        print("Generating moving pattern")

    N, M = 10, 10

    cup_pattern = mug_image_gen(N, M)
    knife_pattern = knife_image_gen(N, M)
    bowl_pattern = bowl_image_gen(N, M)

    if size > 10:
        pad_factor = (size - 10) // 2
        modulo = (size - 10) % 2

        pad_shape = ((pad_factor, pad_factor + modulo), (pad_factor, pad_factor + modulo), (0, 0))

        cup_pattern = np.pad(cup_pattern, pad_shape, "constant")
        knife_pattern = np.pad(knife_pattern, pad_shape, "constant")
        bowl_pattern = np.pad(bowl_pattern, pad_shape, "constant")

    training_data = np.array([cup_pattern, knife_pattern, bowl_pattern])
    target_data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    return training_data, target_data


def get_simple_object_translated(verbose=False, SAMPLES_PER_CLASS=10, size=18, **kwargs):

    # Get the OG data with 10x10
    training_data, target_data = get_simple_object(verbose=verbose, size=10, **kwargs)

    # Expand the amount to satisy the sample requirement specified by user
    training_data = np.array([training_data[i] for _ in range(SAMPLES_PER_CLASS) for i in range(len(training_data))])
    target_data = np.array([target_data[i] for _ in range(SAMPLES_PER_CLASS) for i in range(len(target_data))])

    # Translate training set
    if size > 10:
        training_data = translate(training_data, new_length=(size, size))

    # I know it actually doesn't matter if I shuffle or not but I'd rather avoid any potential problems (f.ex. with training CNNs)
    training_data, target_data = shuffle(training_data, target_data)

    return training_data, target_data


# Not supported anywhere
def get_simple_object_resized(verbose=False, SAMPLES_PER_CLASS=10, size=18, **kwargs):
    training_data, target_data = get_simple_object(verbose=verbose, size=10, **kwargs)

    if size > 10:
        training_data = np.array([cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) for img in training_data])

    return training_data, target_data


def get_MNIST_data_padded(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False, **kwargs):
    training_data, target_data = get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, digits=True)

    diff_x, diff_y = 14, 14
    training_data = np.pad(training_data, ((0, 0), (diff_x, diff_y), (diff_x, diff_y), (0, 0)), "constant")

    return training_data, target_data


def get_MNIST_data_resized(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, size=56, verbose=False, test=False, **kwargs):
    training_data, target_data = get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, digits=True)

    resized_x_data = []
    for img in training_data:
        resized_x_data.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))  # Lost the last dimension!
        resized_x_data[-1] = np.expand_dims(resized_x_data[-1], axis=2)

    return np.array(resized_x_data), target_data


def get_MNIST_data_translated(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False, **kwargs):
    training_data, target_data = get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, digits=True)

    training_data = translate(training_data, new_length=(70, 70))

    return training_data, target_data


def get_MNIST_fashion_data(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False, **kwargs):
    return get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, fashion=True)


def get_CIFAR_data(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False, colors=False, **kwargs):
    return get_data(CLASSES, SAMPLES_PER_CLASS, verbose, test, cifar=True, colors=colors)


def get_MNIST_data(CLASSES=(3, 4), SAMPLES_PER_CLASS=10, verbose=False, test=False, **kwargs):
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
    **kwargs,
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


def _plot_dataset(data_func, kwargs):
    labels = get_labels(data_func, kwargs["CLASSES"])
    kwargs["SAMPLES_PER_CLASS"] = 1
    X_data, y_data = data_func(**kwargs)

    print("\n\n\nShapes", X_data.shape, y_data.shape, "\n\n\n")

    plt.subplots(ncols=len(labels), sharey=True, sharex=True)
    ax = None
    for img, lab in zip(X_data, y_data):
        label = labels[np.argmax(lab)]
        plt.subplot(1, len(labels), np.argmax(lab) + 1)
        if not kwargs["colors"]:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title("Class:" + " " + label)
        xticks, yticks = get_plotting_ticks(img)
        plt.xticks(xticks[0], xticks[1])
        plt.yticks(yticks[0], yticks[1])

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
    np.random.seed(0)
    random.seed(0)

    data_func = get_MNIST_data_resized
    kwargs = {
        "CLASSES": (0, 1, 2, 3, 4),
        "SAMPLES_PER_CLASS": 1,
        "verbose": True,
        "test": True,
        "colors": False,
        "size": 10,
    }

    # print(get_max_samples_balanced(data_func, **kwargs))
    _plot_dataset(data_func, kwargs)
