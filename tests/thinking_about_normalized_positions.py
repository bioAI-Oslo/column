import cmcrameri.cm as cmc
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


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


N, M = 7, 7
N_active, M_active = N - 2, M - 2
N_neo, M_neo = 5, 5

perceptions = get_perception_matrix(N_active, M_active, N_neo, M_neo)

normalized_positions = np.zeros((N_neo, M_neo, 2))

for x in range(N_neo):
    for y in range(M_neo):
        x_p, y_p = perceptions[x, y]
        normalized_positions[x, y, -2] = (x_p / (N_active - 1)) * 2 - 1
        normalized_positions[x, y, -1] = (y_p / (M_active - 1)) * 2 - 1
        print(x_p)

import seaborn as sns

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "",
    colors=[
        "#FFD958",
        "#EF792A",
        "#DC267F",
        "#2C2463",
        "#DC267F",
        "#EF792A",
        "#FFD958",
    ],
)  # Modified Plasma palette to be more colorfriendly (but idk if I succeeded). Also diverging

plt.subplot(131)
sns.heatmap(
    normalized_positions[:, :, 0],
    annot=True,
    cbar=False,
    square=True,
    cmap=cmap,
    vmin=-1,
    vmax=1,
)
plt.subplot(132)

sns.heatmap(
    abs(normalized_positions[:, :, 0]) + abs(normalized_positions[:, :, 1]),
    annot=True,
    cbar=False,
    square=True,
    cmap=cmap,
    vmin=-2,
    vmax=2,
)


plt.subplot(133)
sns.heatmap(
    normalized_positions[:, :, 1],
    annot=True,
    cbar=False,
    square=True,
    cmap=cmap,
    vmin=-1,
    vmax=1,
)
plt.show()
