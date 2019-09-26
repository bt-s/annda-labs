#!/usr/bin/python3

"""data.py File containing functions to load data"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np


def load_X():
    """ Loads X as per assignment sheet

    Returns:
        X (np.ndarray)
    """
    x1 = np.asarray([-1, -1, 1, -1, 1, -1, -1, 1]).reshape((1, 8))
    x2 = np.asarray([-1, -1, -1, -1, -1, 1, -1, -1]).reshape((1, 8))
    x3 = np.asarray([-1, 1, 1, -1, -1, 1, -1, 1]).reshape((1, 8))
    X = np.vstack([x1, x2, x3])

    return X


def load_Xd():
    """ Loads Xd as per assignment sheet

    Returns:
        Xd (np.ndarray)
    """
    x1d = np.asarray([1, -1, 1, -1, 1, -1, -1, 1]).reshape((1, 8))
    x2d = np.asarray([1, 1, -1, -1, -1, 1, -1, -1]).reshape((1, 8))
    x3d = np.asarray([1, 1, 1, -1, 1, 1, -1, 1]).reshape((1, 8))
    Xd = np.vstack([x1d, x2d, x3d])

    return Xd


def load_Xmd():
    """ Loads Xmd as per assignment sheet

    Returns:
        Xmd (np.ndarray)
    """
    # More dissimilar inputs
    x1md = [1, 1, -1, 1, -1, -1, -1, 1] # 4 out of 8 dissimilar
    x2md = [1, 1, 1, 1, 1, 1, -1, -1]   # 5 out of 8 dissimilar
    x3md = [1, -1, -1, 1, 1, 1, -1, 1]  # 5 out of 8 dissimilar
    Xmd = np.vstack((x1md, x2md, x3md))

    return Xmd


def get_pict_data(fname):
    """Gets us the data

    Args:
        fname (str): Path to file

    Returns:
        images (np.ndarray): 11 images of 32x32 pixels
    """
    with open(fname, 'r') as f:
        return np.asarray(f.read().split(',')).reshape((11, 1024)).astype(int)


def create_random_patterns(n, sparsity=.5):
    """Creates n 32x32 random patterns

    Args:
        n (int): The number of random patterns
        sparsity (float): Describes the ratio of nodes 1 vs -1

    Returns:
       X np.ndarray: Containing the n random patterns
    """
    X = np.zeros((n, 1024))
    for i in range(n):
        p = np.full(1024, 1)
        p[:int(len(p)*sparsity)] = -1
        np.random.shuffle(p)
        X[i] = p.reshape((1, 1024))

    return X


