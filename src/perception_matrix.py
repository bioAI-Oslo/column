"""Perception matrix for the network, the logic is tested and shown here. 
Note that only get_perception_matrix is used, all other versions are included for documentation 
and to keep previous versions of the function."""

import numpy as np


def get_perception_matrix_mine(N_active, M_active, N_neo, M_neo):
    """The original version of the general get_perception_matrix function. It is not used anymore because it is slow.
    It is kept to show the logic of the function easier.

    Args:
        N_active (int): Height of the image - (kernel_size - 1) (for MNIST with kernel size 3, N=26)
        M_active (int): Width of the image
        N_neo (int): Height of the perception matrix
        M_neo (int): Width of the perception matrix

    Returns:
        np.array: The perception matrix of size (N_neo, M_neo, 2)
    """
    if N_neo == M_neo == 1:
        return np.array([[[N_active // 2, M_active // 2]]])

    # The scaling matrix
    Sx = (N_neo - 1) / (N_active - 1)  # Scaling factor x
    Sy = (M_neo - 1) / (M_active - 1)  # Scaling factor y
    A = np.array([[Sx, 0], [0, Sy]])  # Scaling matrix
    A_inv = np.linalg.inv(A)  # Inverse gives us the bakwards scaling

    # The perception matrix
    perceptions = np.zeros((N_neo, M_neo, 2), dtype=int)

    # Looping through the perception matrix to calculate new positions
    for x_new in range(N_neo):
        for y_new in range(M_neo):
            x_old, y_old = A_inv @ np.array([x_new, y_new])
            perceptions[x_new, y_new] = [int(np.round(x_old)), int(np.round(y_old))]

    return perceptions


def get_perception_matrix_NxM(N, M):
    """The original version of this function used in this project. It is not used anymore because it does not generalize to more than
    the image size for neo. See get_perception_matrix for the actually used version.

    Args:
        N (int): Height of the image (for MNIST, N=28)
        M (int): Width of the image

    Returns:
        np.ndarray: The perception matrix of shape (N-2, M-2, 2)

    """
    perceptions = np.zeros((N - 2, M - 2, 2), dtype=int)
    for x in range(N - 2):
        for y in range(M - 2):
            perceptions[x, y] = [x, y]
    return perceptions


# Optimized with ChatGPTs help. It has been substantially altered.
def get_perception_matrix_old(N_active, M_active, N_neo, M_neo):
    """
    This function is not used anymore. See get_perception_matrix for the new version.
    """
    if N_neo == M_neo == 1:
        return np.array([[[N_active // 2, M_active // 2]]])
    x_new = np.arange(N_neo)
    y_new = np.arange(M_neo)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    coords = np.stack((X_new.ravel(), Y_new.ravel())).T

    Sx = (N_neo - 1) / (N_active - 1)  # Scaling factor x
    Sy = (M_neo - 1) / (M_active - 1)  # Scaling factor y
    A = np.array([[Sx, 0], [0, Sy]])  # Scaling matrix
    A_inv = np.linalg.inv(A)  # Inverse gives us the backwards scaling

    coords_old = np.round(np.dot(coords, A_inv.T)).astype(int)
    perceptions = coords_old.reshape(N_neo, M_neo, 2)

    # ChatGPT did not realize which order of indexes I had and messed this up for me. The bastard.
    return np.transpose(perceptions, (1, 0, 2))


# Optimized with ChatGPTs help. It has been substantially altered.
def get_perception_matrix(N_active, M_active, N_neo, M_neo):
    """
    A function to make a matrix of coordinates that shows how the neocortex maps to the image.
    F.ex. for a 28x28 image, kernel size 3x3, and therefore active cells 26x26,and 3x3 neocortex, the matrix is of shape (3, 3, 2) and looks like this:
    [[[0,0], [0,12], [0,25]],
    [[12,0], [12,12], [12,25]],
    [[25,0], [25,12], [25,25]]]

    Args:
        N_active (int): Height of the image - (kernel_size - 1) (for MNIST with kernel size 3, N=28, N_active=26)
        M_active (int): Width of the image
        N_neo (int): Height of the perception matrix
        M_neo (int): Width of the perception matrix

    Returns:
        np.array: The perception matrix of size (N_neo, M_neo, 2)

    The difference between get_perception_matrix_old and this one is that the matrix A is inverted in the old version,
    while here, I have already inverted it by hand (which was simple really, so why not).

    Look at get_perception_matrix_mine for the original version.
    It is also easier to understand, but it is slower.
    These two functions do the same.
    """
    # Handling the case of only one network. It maps to the middle.
    if N_neo == M_neo == 1:
        return np.array([[[N_active // 2, M_active // 2]]])

    # For a 3x3 neo example:
    # x_new = 0, 1, 2
    # y_new = 0, 1, 2
    # X_new, Y_new = [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    # coords = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]
    x_new = np.arange(N_neo)
    y_new = np.arange(M_neo)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    coords = np.stack((X_new.ravel(), Y_new.ravel())).T

    # Making a scaling matrix
    # Example contd, with N_active = 26, M_active = 26:
    # A = [[12.5, 0], [0, 12.5]]
    Sx = (N_active - 1) / (N_neo - 1)  # Scaling factor x
    Sy = (M_active - 1) / (M_neo - 1)  # Scaling factor y
    A = np.array([[Sx, 0], [0, Sy]])  # Scaling matrix

    # Example contd:
    # coords_old = [[0, 0], [12, 0], [25, 0], [0, 12], [12, 12], [25, 12], [0, 25], [12, 25], [25, 25]]
    # perceptions = [[[0, 0], [12, 0], [25, 0]], [[0, 12], [12, 12], [25, 12]], [[0, 25], [12, 25], [25, 25]]]
    coords_old = np.round(np.dot(coords, A)).astype(
        int
    )  # While I'm not sure if A needs to be transposed actually, it also does not matter because the other elements are 0
    perceptions = coords_old.reshape(N_neo, M_neo, 2)

    # At this point, the top middle network maps to the left center of the image. This is not what we want. We must transpose.
    # By shifting axes 0 and 1, we flip the matrix around the diagonal. Now top middle maps to top middle.
    # Example contd:
    # return [[[0, 0], [0, 12], [0, 25]], [[12, 0], [12, 12], [12, 25]], [[25, 0], [25, 12], [25, 25]]]
    return np.transpose(perceptions, (1, 0, 2))


if __name__ == "__main__":
    """Main is for testing purposes only"""
    # Importing plotting libraries that don't need to be a part of the files dependencies
    import matplotlib.pyplot as plt
    import seaborn as sns

    def _get_mapping_matrix(N, M, N_active, M_active, N_neo, M_neo, func):
        """
        To get one matrix of size (N, M) that has 0 everywhere, except for where the networks will be positioned
        in the image. Here, the number of the network, from 1 to N_neo * M_neo, is placed in the mapping matrix.

        Args:
            N (int): Image height
            M (int): Image width
            N_active (int): Image height - (kernel_size - 1). The # of possible positions for a kernel
            M_active (int): Image width - (kernel_size - 1). The # of possible positions for a kernel
            N_neo (int): Number of rows in the perception matrix
            M_neo (int): Number of columns in the perception matrix
            func (func): The function that returns the perception matrix

        Returns:
            np.ndarray: The mapping matrix of size (N, M)
        """
        # Get the perception matrix
        perceptions = func(N_active, M_active, N_neo, M_neo)

        # Matrix for plotting the method
        a = np.zeros((N, M))
        neo = np.zeros((N_neo, M_neo))

        # counter is which network gets this position
        counter = 1
        for i in range(N_neo):
            for j in range(M_neo):
                # Get the new positions
                x, y = perceptions[i, j]
                # Increment the plotting matrix to show that network [counter] gets position [x_new, y_new]
                a[x, y] += counter
                # Save the position in the neo matrix
                neo[i, j] = counter

                counter += 1
        return a, neo

    def _test_old_new_and_original_matrices():
        """Tests the old, the new, and the original adaptable perception matrices.
        Tests include visualizing to see if they are correct, and an assertion that they are the same."""
        # Testing for a 28x28 image
        N_active, M_active = 26, 26
        N, M = N_active + 2, M_active + 2

        # Testing many different neo sizes
        for i in range(1, 32):
            N_neo, M_neo = i, i

            # Matrixes for plotting for each method
            a_new, neo_new = _get_mapping_matrix(N, M, N_active, M_active, N_neo, M_neo, get_perception_matrix)
            a_old, neo_old = _get_mapping_matrix(N, M, N_active, M_active, N_neo, M_neo, get_perception_matrix_old)
            a_slow, neo_slow = _get_mapping_matrix(N, M, N_active, M_active, N_neo, M_neo, get_perception_matrix_mine)

            # Plot all the mappings, new and old and slow
            for i, ((a, neo), name) in enumerate(
                zip([(a_new, neo_new), (a_old, neo_old), (a_slow, neo_slow)], ["New", "Old", "Slow"])
            ):
                # Plot the mapping matrix
                plt.figure()
                plt.subplot(121)
                # Make a dark backdrop for visual purposes. Otherwise it would just say 0.
                plt.imshow(np.zeros((N, M)), cmap="plasma", vmin=0, vmax=1)
                # Mask to get rid of all the zeros
                sns.heatmap(a, annot=True, cmap="plasma", cbar=False, square=True, mask=a == 0)
                plt.xlabel("Image axis 2")
                plt.ylabel("Image axis 1")
                plt.title(f"{name} mapping matrix")

                # Plot the neo matrix
                plt.subplot(122)
                sns.heatmap(neo, annot=True, cmap="plasma", cbar=False, square=True)
                plt.xlabel("Neo axis 2")
                plt.ylabel("Neo axis 1")
                plt.title(f"{name} neo matrix")

            plt.show()

            # Assert that all methods do indeed produce the same mapping
            assert np.all(a_new == a_old) and np.all(a_new == a_slow)

    def _plot_example_from_paper():
        """Plots the example from the paper"""
        # Testing for a 7x7 image
        N_active, M_active = 5, 5
        N, M = N_active + 2, M_active + 2
        N_neo, M_neo = 3, 3

        a, n = _get_mapping_matrix(N, M, N_active, M_active, N_neo, M_neo, get_perception_matrix)

        # Plot the mapping matrix
        plt.figure()
        sns.heatmap(a, annot=True, cmap="plasma", cbar=False, square=True)
        plt.xlabel("Image axis 2")
        plt.ylabel("Image axis 1")
        plt.title("Mapping matrix")

        # Plot the neo matrix
        plt.figure()
        sns.heatmap(n, annot=True, cmap="plasma", cbar=False, square=True)
        plt.xlabel("Neo axis 2")
        plt.ylabel("Neo axis 1")
        plt.title("Neo matrix")

        plt.show()

    # Put the test under here
    _test_old_new_and_original_matrices()
