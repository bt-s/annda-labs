#!/usr/bin/python3

"""helper.py Helper functions for data preparation and visualization purposes."""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import time
import numpy as np
import matplotlib.pyplot as plt


def get_pict_data(fname):
    """Gets us the data

    Args:
        fname (str): Path to file

    Returns:
        images (np.ndarray): 11 images of 32x32 pixels
    """
    with open(fname, 'r') as f:
        return np.asarray(f.read().split(',')).reshape((11, 1024)).astype(int)


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




