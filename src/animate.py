"""File to animate the system as it solves a classification"""

import gc

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def highlight_cell(x, y, ax=None, last_rect=None, **kwargs):
    """
    Highlight the cell at x, y on the axis ax with a rectangle.

    From here:
    https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
    ImportanceOfBeingErnest: https://stackoverflow.com/users/4124317/importanceofbeingernest

    Origin of plt.Rectangle is upper left corner, just like plt.imshow
    I tested and found
    Rectangle's (0,20) is down from origin, (20,0) is right of origin
    plt.imshow's (0,20) is right and (20,0) is down
    Therefore, I switch x and y for Rectangle
    rect = plt.Rectangle((y-.5, x-.5), 1,1, fill=False, **kwargs)

    Args:
        x, y (int): The coordinates of the cell to highlight.
        ax (matplotlib.axes.Axes, optional): The axis on which to draw the rectangle. If None, use the current axis.
        last_rect (matplotlib.patches.Rectangle, optional): The previous rectangle to remove. If None, do nothing.
        **kwargs: Additional keyword arguments to pass to matplotlib.patches.Rectangle.

    Returns:
        matplotlib.patches.Rectangle: The created rectangle.
    """
    rect = plt.Rectangle((y - 1.5, x - 1.5), 3, 3, fill=False, alpha=0.4, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    if last_rect is not None:  # Garbage collection was necessary because of memory leak
        last_rect.remove()
        del last_rect
    return rect


def animate(
    images, states, actions, perceptions_through_time, hidden_channels, class_channels, labels, correct_label_index
):
    """
    Animate the classification process of the Active NCA model.

    Parameters:
        images (list of np.ndarrays): The images to classify.
        states (list of np.ndarrays): The states of the model at each step.
        actions (list of np.ndarrays or None): The actions taken by the model at each step. If None, the actions are not plotted.
        perceptions_through_time (list of np.ndarrays): The perceptions of the model at each step.
        hidden_channels (int): The number of hidden channels of the model.
        class_channels (int): The number of class channels of the model.
        labels (list of str): The labels of the classes.
        correct_label_index (int): The index of the correct label.
    """
    # Set up the figure, sized according to the number of classes
    if class_channels == 10:
        fig = plt.figure(figsize=(15, 5))
        plt.subplots_adjust(top=0.8, bottom=0, hspace=0)
    elif class_channels == 4:
        fig = plt.figure(figsize=(10, 5))
        plt.subplots_adjust(top=0.8, bottom=0, hspace=0)
    elif class_channels == 3:
        fig = plt.figure(figsize=(10, 5))
        plt.subplots_adjust(top=0.8, bottom=0, hspace=0)
    elif class_channels == 5:
        fig = plt.figure(figsize=(10, 5))
        plt.subplots_adjust(top=0.8, bottom=0, hspace=0)
    else:
        fig = plt.figure()

    super_ax = plt.gca()
    super_ax.axis("off")

    # Converting to numpy makes indexing easier later on
    states = np.array(states)
    actions = np.array(actions) if actions is not None else None

    # Get amount of images in the horizontal direction
    max_images_on_line = max(hidden_channels + 1 + (2 * (actions is not None)), class_channels)

    # IBM design library palette
    """cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["#f17720", "#ffa630", "#ffffff", "#00a7e1", "#0474ba"]
    )"""
    cmap = "RdBu"

    # Plot image
    ax_img = fig.add_subplot(2, max_images_on_line, 1)
    if len(images[0][0][0]) == 1:
        im = ax_img.imshow(images[0], cmap="RdBu", vmin=-1, vmax=1)
    else:
        im = ax_img.imshow(images[0])
    ax_img.set_title("Image")
    ax_img.set_yticks([0, images[0].shape[0]], [0, images[0].shape[0]])
    ax_img.set_xticks([0, images[0].shape[1]], [0, images[0].shape[1]])

    # Plot hidden channels
    im_hidden_list = []
    for j in range(hidden_channels):
        ax_hidden = fig.add_subplot(2, max_images_on_line, 2 + j)
        im_hidden = ax_hidden.imshow(states[0, :, :, j], cmap=cmap, vmin=-1, vmax=1)
        im_hidden_list.append(im_hidden)
        ax_hidden.set_title("Hidden\nchannel " + str(j + 1))
        ax_hidden.set_yticks([])
        ax_hidden.set_xticks([1, states[0].shape[1] - 2], [1, states[0].shape[1] - 2])
    # cb = plt.colorbar(im_hidden, ax=[ax_hidden], location="right")

    # Plot actions (always just 2)
    if not actions is None:
        im_action_list = []
        for j in range(2):
            ax_action = fig.add_subplot(2, max_images_on_line, 2 + hidden_channels + j)
            im_action = ax_action.imshow(actions[0, :, :, j], cmap="RdBu", vmin=-0.0007, vmax=0.0007)
            im_action_list.append(im_action)
            ax_action.set_title("Up/Down" if j == 0 else "Left/Right")
            ax_action.set_yticks([])
            ax_action.set_xticks([1, actions[0].shape[1] - 2], [1, actions[0].shape[1] - 2])
        # cb = plt.colorbar(im_action, ax=[ax_action], location="right")

    # Show class channels
    im_class_list = []
    ax_class_list = []
    for j in range(class_channels):
        ax_class = fig.add_subplot(2, max_images_on_line, 1 + max_images_on_line + j)
        im_class = ax_class.imshow(states[0, :, :, hidden_channels + j], cmap=cmap, vmin=-1, vmax=1)
        im_class_list.append(im_class)
        ax_class_list.append(ax_class)
        ax_class.set_title("Class\n" + str(labels[j]))
        ax_class.set_yticks([])
        ax_class.set_xticks([1, states[0].shape[1] - 2], [1, states[0].shape[1] - 2])
    # cb = plt.colorbar(im_class, ax=[ax_class], location="right")

    # Keeping track of the little squares because they need careful attention to actually be deleted
    # Otherwise they stay around and cause the process to break
    last_rects = []

    def animate_func(i):
        # Set title to indicate current belief
        state = states[i]
        believed = np.argmax(np.mean(state[1:-1, 1:-1, -class_channels:], axis=(0, 1)))
        super_ax.set_title(
            f"Timestep: {i}\nCurrent classification: "
            + str(labels[believed])
            + "\nGround truth: "
            + str(labels[correct_label_index]),
            pad=20,
            ha="left",
            x=0,
        )

        # Set image, hidden channels, actions, and class channels
        im.set_array(images[i])

        for j in range(hidden_channels):
            im_hidden_list[j].set_array(states[i, :, :, j])

        if not actions is None:
            for j in range(len(im_action_list)):
                im_action_list[j].set_array(actions[i, :, :, j])

        for j in range(class_channels):
            im_class_list[j].set_array(states[i, :, :, hidden_channels + j])
            if believed == j:
                if j == correct_label_index:
                    ax_class_list[j].set_title("Class\n" + str(labels[j]), color="#30C14E", weight="bold")
                else:
                    ax_class_list[j].set_title("Class\n" + str(labels[j]), color="#DC2381", weight="bold")
            else:
                ax_class_list[j].set_title("Class\n" + str(labels[j]), color="black")

        ### Update the little squares indicating perceptive fields
        perceptions = perceptions_through_time[i]

        # This is a bit of a hack, but it works
        last_rects_i = None if len(last_rects) == 0 else last_rects[-1]
        last_rects.clear()

        # Add the little squares
        last_rects.append([])
        for x in range(0, len(perceptions)):
            last_rects[-1].append([])
            for y in range(0, len(perceptions[0])):
                # Perceptions point to the beginning of the field, but plotted squares need to know the middle of the field
                x_p = perceptions[x, y, 0] + 1
                y_p = perceptions[x, y, 1] + 1

                last_rect = None if last_rects_i is None else last_rects_i[x][y]
                last_rect = highlight_cell(
                    x_p, y_p, ax=ax_img, color="mediumvioletred", linewidth=1, last_rect=last_rect
                )
                last_rects[-1][-1].append(last_rect)

        if not actions is None:
            return [im, *im_hidden_list, *im_action_list, *im_class_list]
        return [im, *im_hidden_list, *im_class_list]

    # Animate
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=list(range(len(images))),
        interval=100,  # in ms
    )

    plt.show()

    # This is probably not needed, but I kept it after I was stuck with a memory leak
    del last_rects
    del im
    del anim

    plt.clf()
    plt.close("all")
    gc.collect()
