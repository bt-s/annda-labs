#!/usr/bin/python3

"""helper.py Helper functions for data preparation and visualization purposes."""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
import matplotlib.pyplot as plt


def plot_images(images, ixs=None):
    """Plots images from arrays

    Args:
        images (np.ndarray): (x, y) array where,
            - x = number of images
            - y = flattened shape of image x
        ixs (list): Indices of specific images
    """
    if ixs != None:
        for ix in ixs:
            plt.imshow(images[ix].reshape(32, 32))
            plt.show()

    else:
        for i in images:
            plt.imshow(i.reshape(32, 32))
            plt.show()




