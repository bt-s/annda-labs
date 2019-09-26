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


def add_noise(x, noise_level):
    """Adds noise to a pattern

    Args:
        x (np.ndarray): The input pattern
        noise_level (float): The fraction of noise to be added to x

    Return:
        x_prime (np.ndarray): The input pattern corrupted with noise
    """
    mask = np.full(len(x), False)
    mask[:int(len(x)*noise_level)] = True
    np.random.shuffle(mask)
    x_prime = np.where(mask, -x, x).reshape((1, x.shape[0]))

    return x_prime

