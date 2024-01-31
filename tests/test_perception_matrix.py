import numpy as np
import pytest

from ..src.perception_matrix import get_perception_matrix, get_perception_matrix_mine


def test_basic_function1():
    N, M = 5, 5
    N_cells, M_cells = 5, 5
    matrix = get_perception_matrix(N, M, N_cells, M_cells)

    for x, x_p in zip(range(N_cells), range(0, N_cells, 1)):
        for y, y_p in zip(range(M_cells), range(0, M_cells, 1)):
            assert np.all(matrix[x, y] == [x_p, y_p])


def test_basic_function2():
    N, M = 5, 5
    N_cells, M_cells = 3, 3
    matrix = get_perception_matrix(N, M, N_cells, M_cells)

    for x, x_p in zip(range(N_cells), range(0, N_cells, 2)):
        for y, y_p in zip(range(M_cells), range(0, M_cells, 2)):
            assert np.all(matrix[x, y] == [x_p, y_p])


def test_basic_function_complicated():
    N, M = 7, 7
    N_cells, M_cells = 3, 3
    matrix = get_perception_matrix(N, M, N_cells, M_cells)

    assert np.all(matrix[0, 0:3] == [[0, 0], [0, 3], [0, 6]]), matrix[0, 1:3]
    assert np.all(matrix[1, 0:3] == [[3, 0], [3, 3], [3, 6]]), matrix[0, 1:3]
    assert np.all(matrix[2, 0:3] == [[6, 0], [6, 3], [6, 6]]), matrix[0, 1:3]


def test_basic_function_complicated2():
    N, M = 8, 8
    N_cells, M_cells = 3, 3
    matrix = get_perception_matrix(N, M, N_cells, M_cells)

    assert np.all(matrix[0, 0:3] == [[0, 0], [0, 4], [0, 7]]), matrix[0, 1:3]
    assert np.all(matrix[1, 0:3] == [[4, 0], [4, 4], [4, 7]]), matrix[0, 1:3]
    assert np.all(matrix[2, 0:3] == [[7, 0], [7, 4], [7, 7]]), matrix[0, 1:3]


@pytest.mark.parametrize(
    "N, M, N_cells, M_cells", [(26, 26, 26, 26), (26, 26, 15, 15), (26, 26, 5, 5), (26, 26, 1, 1), (30, 30, 15, 15)]
)
def test_get_perception_matrix_shape(N, M, N_cells, M_cells):
    matrix = get_perception_matrix(N, M, N_cells, M_cells)
    assert matrix.shape == (N_cells, M_cells, 2)


@pytest.mark.parametrize(
    "N, M, N_cells, M_cells", [(26, 26, 26, 26), (26, 26, 15, 15), (26, 26, 5, 5), (26, 26, 1, 1), (30, 30, 15, 15)]
)
def test_perception_matrix_equal(N, M, N_cells, M_cells):
    # N is active cells of the image
    # N_cells is number of networks (cells) in the neocortex
    # They can be equal, but N >= N_cells
    matrix1 = get_perception_matrix(N, M, N_cells, M_cells)
    matrix2 = get_perception_matrix_mine(N, M, N_cells, M_cells)

    assert np.all(matrix1 == matrix2), f"Failed on {N}, {M}, {N_cells}, {M_cells}"
