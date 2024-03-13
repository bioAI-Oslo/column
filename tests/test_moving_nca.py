import numpy as np
import pytest
from matplotlib import pyplot as plt

# from ..src.moving_nca import collect_input_batched


def get_perception_matrix(N, M, N_cells, M_cells):
    """
    Look at get_perception_matrix_mine for the original version.
    It is also easier to understand, but it is slower.
    These two functions do the same.
    """
    if N_cells == M_cells == 1:
        return np.array([[[N // 2, M // 2]]])
    x_new = np.arange(N_cells)
    y_new = np.arange(M_cells)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    coords = np.stack((X_new.ravel(), Y_new.ravel())).T

    Sx = (N_cells - 1) / (N - 1)  # Scaling factor x
    Sy = (M_cells - 1) / (M - 1)  # Scaling factor y
    A = np.array([[Sx, 0], [0, Sy]])  # Scaling matrix
    A_inv = np.linalg.inv(A)  # Inverse gives us the backwards scaling

    coords_old = np.round(np.dot(coords, A_inv.T)).astype(int)
    perceptions = coords_old.reshape(N_cells, M_cells, 2)

    # ChatGPT did not realize which order of indexes I had and messed this up for me. The bastard.
    return np.transpose(perceptions, (1, 0, 2))


def collect_input_batched(input, images, state_batched, perceptions_batched, position, N_neo, M_neo):
    B, N, M, _ = images.shape

    for x in range(N_neo):
        for y in range(M_neo):
            for b in range(B):
                x_p, y_p = perceptions_batched[b, x, y].T
                perc = images[b, x_p : x_p + 3, y_p : y_p + 3, :]
                comms = state_batched[b, x : x + 3, y : y + 3, :]

                dummy = np.concatenate((perc, comms), axis=-1)
                dummy_flat = dummy.flatten()
                input[b * N_neo * M_neo + x * M_neo + y, : len(dummy_flat)] = dummy_flat

                # When position is None, the input is just the perception and comms
                if position == "current":
                    input[b * N_neo * M_neo + x * M_neo + y, -2] = (x_p - N // 2) / (N // 2)
                    input[b * N_neo * M_neo + x * M_neo + y, -1] = (y_p - M // 2) / (M // 2)
                elif position == "initial":
                    input[b * N_neo * M_neo + x * M_neo + y, -2] = (float(x) * N / float(N_neo) - N // 2) / (N // 2)
                    input[b * N_neo * M_neo + x * M_neo + y, -1] = (float(y) * M / float(M_neo) - M // 2) / (M // 2)


def test_collect_input_batched():

    CHANNELS = 4
    B = 2
    N, M = 7, 7
    N_neo, M_neo = 5, 5

    images = np.zeros((B, N, M, 1))

    for b in range(B):
        for x in range(N):
            for y in range(M):
                images[b, x, y, 0] = b + x + y

    # 0 1 2 3   1 2 3 4
    # 1 2 3 4   2 3 4 5
    # 2 3 4 5   3 4 5 6
    # 3 4 5 6   4 5 6 7 ->

    state_batched = np.zeros((B, N, M, CHANNELS))
    perceptions_batched = np.array([get_perception_matrix(N - 2, M - 2, N_neo, M_neo) for _ in range(B)])
    input = np.empty((B * N_neo * M_neo, 3 * 3 * (CHANNELS + 1) + 2))
    position = "current"

    collect_input_batched(input, images, state_batched, perceptions_batched, position, N_neo, M_neo)

    for i in range(len(input)):
        print(input[i, :-2:5])


test_collect_input_batched()
