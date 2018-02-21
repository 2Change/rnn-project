from __future__ import division

import numpy as np
import scipy, scipy.misc
from math import sqrt, floor, ceil


def imresize(images, size, mode='F'):
    """
    Resize image(s)

    :param images: input image(s) - 4D array for multiple images
    :param size: output size
    :return: resized image
    """

    def _imresize(img, size):
        return scipy.misc.imresize(img, size, mode=mode)

    # scipy.imresize accepts shape (H,W) or (H,W,3)
    # if only one channel is used, I have to remove it
    one_channel = (images.shape[-1] == 1)

    if one_channel:
        images = np.squeeze(images)

    # multiple images
    if len(images.shape) == 4 or (len(images.shape) == 3 and images.shape[-1] != 3):
        images = np.array([_imresize(img, size) for img in images])
    # single images
    else:
        images = _imresize(images, size)

    # if one channel was used, add it again
    if one_channel:
        images = np.expand_dims(images, axis=len(images.shape))

    return images



def to_grid(images, nrows=None, ncols=None):

    """

    Transforms a list of images into a grid of images
    The number of rows or cols of the output grid can be specified;
    if not, the dimension of the grid is as close to sqrt(nb_images) as possible.
    Throws an exception if both nrows and ncols are specified but
    nrows * ncols < number of images


    :param images: a 4D numpy array with shape = (N, width, height, channels)
    :param nrows: (optional) required number of rows
    :param ncols: (optional) required number of cols
    :return: a 3D numpy array with shape = (new_rows, new_cols, channels)

    """

    n, h, w, ch = images.shape
    if nrows is None and ncols is None:
        s = sqrt(n)
        nrows, ncols = int(floor(s)), int(ceil(s))

        if nrows*ncols < n:
            nrows = nrows + 1  #ceil

    elif nrows is None and ncols is not None:
        nrows = int(ceil(n / ncols))
    elif nrows is not None and ncols is None:
        ncols = int(ceil(n / nrows))
    else:  # both are defined
        if ncols * nrows < n:
            raise ValueError('not enough cells to store ' + str(n) + ' images')

    if ncols * nrows > n:
        # padding
        images = np.concatenate([images, np.zeros((ncols*nrows - n, h,w,ch))])

    result = images.reshape((nrows, ncols, h, w, ch)) \
        .swapaxes(1, 2) \
        .reshape((h * nrows, w * ncols, ch))

    return result


def plotgrid(images, nrows=None, ncols=None, show=True, to_file=None):

    """

    Plot input images as a grid into the current pyplot ax.

    :param images: see to_grid
    :param nrows: see to_grid
    :param ncols: see to_grid
    :param show: (default True) calls plt.show()
    :param to_file: (optional) if specified, save the grid to the specified location

    """
    import matplotlib.pyplot as plt


    if len(images.shape) == 3:
        if images.shape[-1] > 3:
            # multiple images with 1 or 3 channels
            # the assumption is that I won't plot images with (h,w) <= 3 (seems reasonable)
            images = np.expand_dims(images, 3)
        else:
            # single image with 1 or 3 channels
            images = np.expand_dims(images, 0)

    _, h, w, _ = images.shape
    result = to_grid(images, nrows, ncols)
    nrows, ncols, channels = result.shape[0] // h, result.shape[1] // w, result.shape[2]
    ax = plt.gca()

    ax.xaxis.set_ticks_position('top')

    if channels == 1:
        result = result.reshape(result.shape[0], result.shape[1])
        plt.imshow(result, cmap='gray')
    else:
        plt.imshow(result)

    plt.xticks(np.arange(ncols)*w, np.arange(ncols))
    plt.yticks(np.arange(nrows)*h, np.arange(nrows)*ncols)

    if to_file:
        plt.savefig(to_file)

    if show:
        plt.show()


def cv2_imshow(filename, title='some image', waitKey=None):

    """
    Show an image file into a OpenCV window.
    OpenCV seems the only library that can handle 'real time' image rendering
    (matplotlib.pyplot or scipy are too slow)

    :param filename: filename of the image
    :param title: title of the opencv window (same title -> same window)

    :param waitKey: (optional) if specified, calls cv2.waitKey with the specified amount of time

    """

    import cv2


    img = cv2.imread(filename, 0)
    cv2.imshow(title, img)

    if waitKey:
        cv2.waitKey(waitKey)


def savegrid(images, nrows=None, ncols=None, to_file='grid.png'):

    """

    Save input image as a grid

    :param images: see to_grid
    :param nrows: see to_grid
    :param ncols: see to_grid
    :param to_file: filename for output image

    """

    grid = to_grid(images, nrows, ncols)
    nrows, ncols, channels = grid.shape

    if channels == 1:
        grid = grid.reshape(nrows, ncols)

    scipy.misc.imsave(to_file, grid)
