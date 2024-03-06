import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10


def shuffle(X_data, y_data):
    temp = list(zip(X_data, y_data))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    training_data, target_data = np.array(res1), np.array(res2)

    return training_data, target_data


sorted_X_train = None
sorted_X_test = None


def get_CIFAR_data(MNIST_DIGITS=(3, 4), SAMPLES_PER_DIGIT=10, verbose=False, test=False):
    global sorted_X_train
    global sorted_X_test
    if verbose:
        print("Getting", "training" if not test else "test", "data")
    if not test and sorted_X_train is None:
        if verbose:
            print("Initializing CIFAR training data")
        sorted_X_train = initalize_CIFAR_reduced_classes(MNIST_DIGITS, test=False)
    elif test and sorted_X_test is None:
        if verbose:
            print("Initializing CIFAR test data")
        sorted_X_test = initalize_CIFAR_reduced_classes(MNIST_DIGITS, test=True)

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


def initalize_CIFAR_reduced_classes(MNIST_DIGITS=(3, 4), test=False):
    # Loading
    data = cifar10.load_data()
    (train_X, train_y), (test_X, test_y) = data
    x = train_X if not test else test_X
    y = train_y if not test else test_y

    # Thsi data has shape (len(y), 1), so reshaping it to accomodate the following code.
    y = np.reshape(y, (len(y)))

    # Converting to grayscale
    x_list = []
    for img in x:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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


if __name__ == "__main__":
    # 0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    x_data, y_data = get_CIFAR_data(MNIST_DIGITS=(0, 1, 2, 3, 4), SAMPLES_PER_DIGIT=4, verbose=True, test=False)

    for img, lab in zip(x_data, y_data):
        plt.figure()
        plt.imshow(img)
        plt.title(str(lab))

    plt.show()
