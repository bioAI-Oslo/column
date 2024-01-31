import numpy as np


def get_perception_matrix_mine(N, M, N_cells, M_cells):
    if N_cells == M_cells == 1:
        return np.array([[[N // 2, M // 2]]])
    Sx = (N_cells - 1) / (N - 1)  # Scaling factor x
    Sy = (M_cells - 1) / (M - 1)  # Scaling factor y
    A = np.array([[Sx, 0], [0, Sy]])  # Scaling matrix
    A_inv = np.linalg.inv(A)  # Inverse gives us the bakwards scaling

    perceptions = np.zeros((N_cells, M_cells, 2), dtype=int)

    for x_new in range(N_cells):
        for y_new in range(M_cells):
            x_old, y_old = A_inv @ np.array([x_new, y_new])
            perceptions[x_new, y_new] = [int(np.round(x_old)), int(np.round(y_old))]
    return perceptions


def get_perception_matrix_NxM(N, M):
    perceptions = np.zeros((N - 2, M - 2, 2), dtype=int)
    for x in range(N - 2):
        for y in range(M - 2):
            perceptions[x, y] = [x, y]
    return perceptions


# Optimized with ChatGPTs help
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


if __name__ == "__main__":
    N, M = 4, 4
    N_cells, M_cells = 3, 3

    perceptions_GPT = get_perception_matrix(N, M, N_cells, M_cells)

    # Plotting
    a = np.random.rand(N + 2, M + 2) * 0.1
    import matplotlib.pyplot as plt

    for row in perceptions_GPT:
        for x, y in row:
            a[x, y] += 1

    plt.imshow(a)
    plt.show()

    """perceptions_mine = get_perception_matrix_mine(N, M, N_cells, M_cells)

    for x in range(N_cells):
        for y in range(M_cells):
            print(perceptions_GPT[x, y], end=" ")
            print(perceptions_mine[x, y])"""
