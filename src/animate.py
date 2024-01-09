import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gc

def pad(action, N, M):
    N_neo, M_neo, N_actions = action.shape

    padded = np.zeros((N,M,N_actions))
    for i in range(N_actions):
        padded[1:N_neo+1, 1:M_neo+1, i] = action[:,:,i]

    return padded

# Assumes there is only 1 image channel (grayscale)
def transform(img, action, hidden_channels, class_channels):
    N, M, N_channels = img.shape
    action = pad(action, N, M)
    concat = np.zeros((N*2, M*(max(1+hidden_channels, class_channels)+1)))

    for plot_index in range(1+hidden_channels):
        concat[:N,M*plot_index:M*(plot_index+1)] = img[:,:,plot_index]

    for plot_index in range(class_channels):
        concat[N:,M*plot_index:M*(plot_index+1)] = img[:,:,1+hidden_channels+plot_index]

    for plot_index in range(2):
        concat[N*plot_index:N*plot_index+N,-M:] = action[:,:,plot_index]*1000

    return concat

# From here:
# https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
# ImportanceOfBeingErnest: https://stackoverflow.com/users/4124317/importanceofbeingernest
def highlight_cell(x,y, M, ax=None, last_rect=None, **kwargs):
    # Origin of plt.Rectangle is upper left corner, just like plt.imshow
    # I tested and found
    # Rectangle's (0,20) is down from origin, (20,0) is right of origin
    # plt.imshow's (0,20) is right and (20,0) is down
    # Therefore, I switch x and y for Rectangle
    #rect = plt.Rectangle((y-.5, x-.5), 1,1, fill=False, **kwargs)
    rect = plt.Rectangle((y-1.5, x-1.5), 3,3, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    if last_rect is not None:
        last_rect.remove()
        del last_rect
    return rect

def animate(images, perceptions_through_time, actions, hidden_channels, class_channels, mnist_digits):
    a = transform(images[0], actions[0], hidden_channels, class_channels)

    N, M = a.shape
    fig = plt.figure( figsize=(20,8) )
    ax = fig.add_subplot(111)

    im = plt.imshow(a, cmap="RdBu", interpolation='none', aspect='auto', vmin=-1, vmax=1)
    cb = plt.colorbar(im,ax=[ax],location='right')

    one_length = N // 2
    plt.yticks([N//4, N*3//4], ["Image", "Belief"])

    plt.xticks(
        [one_length*i+one_length//2 for i in range(class_channels)],
        ["Class "+str(mnist_digits[i]) for i in range(class_channels)]
    )

    last_rects = []
    def animate_func(i):
        print(i)
        im.set_array(transform(images[i], actions[i], hidden_channels, class_channels))

        img = images[i]
        believed = np.argmax(np.mean(img[1:-1,1:-1,-class_channels:], axis=(0,1)))
        ax = plt.gca()
        ax.set_title(f"{i} Believed class: " + str(mnist_digits[believed]))

        perceptions = perceptions_through_time[i]

        last_rects_i = None if len(last_rects) == 0 else last_rects[-1]
        last_rects.clear()

        last_rects.append([])
        for x in range(0,len(perceptions)):
            last_rects[-1].append([])
            for y in range(0,len(perceptions[0])):
                x_p = perceptions[x,y,0]+1
                y_p = perceptions[x,y,1]+1

                last_rect = None if last_rects_i is None else last_rects_i[x][y]
                last_rect = highlight_cell(x_p, y_p, M, color="mediumvioletred", linewidth=3, last_rect=last_rect)
                last_rects[-1][-1].append(last_rect)

        return [im]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames = list(range(len(images))),
        interval = 100, # in ms
    )

    plt.show()

    del last_rects
    del im
    del anim

    plt.clf()
    plt.close("all")
    gc.collect()
