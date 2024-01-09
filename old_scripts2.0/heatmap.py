import copy
import multiprocessing as mp

import cma
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_generator import get_data
from main import *
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

from column.src.animate import animate
from column.src.loss import (
    entropy_seeking_loss,
    final_guess_wise_L2,
    pixel_wise_L2,
    scale_loss,
)
from column.src.mnist_processing import get_dataset, get_MNIST_data
from column.src.perception_matrix import get_perception_matrix
from column.src.utils import (
    add_channels,
    get_flat_weights,
    get_model_weights,
    get_weights_info,
    load_checkpoint,
    save_checkpoint,
)


def heatmap(images, perceptions_through_time, outputs_through_time, HIDDEN_CHANNELS, CLASS_CHANNELS, ACT_CHANNELS):
    ITERATIONS = len(images)
    N, M, O = images[0].shape

    input = []
    output = []
    actions = []
    for iter in range(1, ITERATIONS):
        if np.random.rand() < 0.2:
            img, perceptions, outputs = images[iter - 1], perceptions_through_time[iter - 1], outputs_through_time[iter]

            for x in range(N - 2):
                for y in range(M - 2):
                    x_p, y_p = perceptions[x, y]
                    perc = img[x_p : x_p + 3, y_p : y_p + 3, :1]
                    comms = img[x : x + 3, y : y + 3, 1:]
                    input.append(np.concatenate((perc, comms), axis=2))

                    x_p, y_p = outputs[x, y, -ACT_CHANNELS:]
                    x_a = custom_round(x_p)
                    y_a = custom_round(y_p)

                    actions.append([x_a, y_a])

    actions_direction = np.array([[np.zeros((3, 3, O)) for _ in range(3)] for _ in range(3)])
    for i in range(len(actions)):
        x_a, y_a = actions[i]

        actions_direction[1 + x_a, 1 + y_a] += input[i]

    for ch in range(O):
        plt.figure()
        plt.suptitle(
            "Class " + str(MNIST_DIGITS[ch - 4]) if ch > 3 else "Hidden channel " + str(ch - 1) if ch != 0 else "Image"
        )
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.title(str(-1 + i // 3) + " " + str(-1 + i % 3))
            plt.imshow(actions_direction[i // 3, i % 3, :, :, ch])  # , vmax=np.max(actions_direction[:,:,:,:,ch]))
    plt.show()


winner_flat = load_checkpoint()
network = NeuralNetwork_TF.get_instance_with(winner_flat)

kwargs["SAMPLES_PER_DIGIT"] = 100
training_data, target_data = data_func(**kwargs)
training_data = training_data[:]
target_data = target_data[:]

N, M = training_data.shape[1:3]
N_active, M_active = N - 2, M - 2
N_neo, M_neo = N_active, M_active

loss = 0
accuracy = 0

images = []
perceptions_through_time = []
outputs_through_time = []

for img_raw, expected in zip(training_data, target_data):
    img = add_channels(np.array([img_raw]), NUM_CHANNELS - IMG_CHANNELS)[0]

    # perceptions
    perceptions = get_perception_matrix(N_active, M_active, N_neo, M_neo)

    guesses = None
    for _ in range(ITERATIONS):
        input = []
        for x in range(N_neo):
            for y in range(M_neo):
                x_p, y_p = perceptions[x, y]
                perc = img[x_p : x_p + 3, y_p : y_p + 3, :1]
                comms = img[x : x + 3, y : y + 3, 1:]
                input.append(np.concatenate((perc, comms), axis=2))

        guesses = network(np.array(input)).numpy()
        outputs = np.reshape(guesses[:, :], (N_neo, M_neo, output_dim))

        img[1 : 1 + N_neo, 1 : 1 + M_neo, 1:] = (
            img[1 : 1 + N_neo, 1 : 1 + M_neo, 1:] + outputs[:, :, : end_of_class - IMG_CHANNELS]
        )

        if True:
            for x in range(N_neo):
                for y in range(M_neo):
                    action = outputs[x, y, -ACT_CHANNELS:]
                    perception = perceptions[x, y]
                    perceptions[x, y] = add_action(perception, action, N_active, M_active)

        images.append(copy.deepcopy(img))
        perceptions_through_time.append(copy.deepcopy(perceptions))
        outputs_through_time.append(copy.deepcopy(outputs))

    class_predictions = img[1 : 1 + N_neo, 1 : 1 + M_neo, -CLASS_CHANNELS:]

    loss += loss_function(class_predictions, guesses, expected)

    belief = np.mean(class_predictions, axis=(0, 1))
    believed = np.argmax(belief)
    actual = np.argmax(expected)
    accuracy += int(believed == actual)
    print("Expected", expected, "got", belief)

heatmap(images, perceptions_through_time, outputs_through_time, HIDDEN_CHANNELS, CLASS_CHANNELS, ACT_CHANNELS)

print(loss, float(accuracy) / float(training_data.shape[0]))
