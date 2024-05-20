"""A little script to get to know plt.Rectangle

Because I have a bad dicipline in labelling axes in matrixes, plt.Rectangle made me confused.
Basically, I call its y-axis the x-axis. 

So under, I try to show myself its peculiarities."""

import matplotlib.pyplot as plt
import numpy as np

N = 10  # Size of image

image = np.zeros((N, N, 3))

# Image where the lower left is greem and the upper right is blue
# Lower right corner is white
for i in range(N):
    for j in range(N):
        image[i][j][0] = 1.0 * i * j / (N - 1) ** 2
        image[i][j][1] = 1.0 * i / (N - 1)
        image[i][j][2] = 1.0 * j / (N - 1)

plt.ylabel("First image axis:\nFor me: x-axis\nFor plt: y-axis")
plt.xlabel("Second image axis:\nFor me: y-axis\nFor plt: x-axis")

ax = plt.gca()
# Upper left corner (origin)
# Minus 0.5 because pixel center is (0,0), and we want to include the whole pixel (0,0)
rect = plt.Rectangle((0.0 - 0.5, 0.0 - 0.5), 3, 3, fill=False, color="mediumvioletred", linewidth=1)
ax.add_patch(rect)

# Upper right corner
rect = plt.Rectangle((7.0 - 0.5, 0.0 - 0.5), 3, 3, fill=False, color="green", linewidth=1)
ax.add_patch(rect)

# Lower left corner
rect = plt.Rectangle((0.0 - 0.5, 7.0 - 0.5), 3, 3, fill=False, color="black", linewidth=1)
ax.add_patch(rect)

plt.imshow(image)


### Now to show the problem:

plt.figure()

# How I normally do it:
plt.subplot(121)
ax = plt.gca()

image = np.zeros((N, N, 3))
for x in range(N):
    for y in range(N):
        color = (1.0 * x * y / (N - 1) ** 2, 1.0 * x / (N - 1), 1.0 * y / (N - 1))
        # OBS: the indexing is wrong this way
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, color=color, linewidth=1)
        ax.add_patch(rect)

plt.imshow(image)
plt.title("Wrong indexing")

# Fixing it: The indexing colors should correspond with the above colored image gradient
plt.subplot(122)
ax = plt.gca()

image = np.zeros((N, N, 3))
for x in range(N):
    for y in range(N):
        color = (1.0 * x * y / (N - 1) ** 2, 1.0 * x / (N - 1), 1.0 * y / (N - 1))
        # OBS: now correct indexing
        rect = plt.Rectangle((y - 0.5, x - 0.5), 1, 1, fill=False, color=color, linewidth=1)
        ax.add_patch(rect)

plt.title("Correct indexing")
plt.imshow(image)
plt.show()
