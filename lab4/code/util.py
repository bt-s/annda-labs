#!/usr/bin/python3

"""util.py - Containing utitility functions for lab 4.

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology

Note: Part of this code was provided by the course coordinators.
"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def sigmoid(support):
    """ Sigmoid activation function that finds probabilities to turn ON each unit

    Args:
        support (np.ndarray): (size of mini-batch, size of layer)

    Returns:
        (np.ndarray): on_probabilities (size of mini-batch, size of layer)
    """
    return 1. / (1. + np.exp(-support))


def softmax(support):
    """Softmax activation function that finds probabilities of each category

    Args:
        support (np.ndarray): (size of mini-batch, number of categories)

    Returns:
        (np.ndarray): on_probabilities (size of mini-batch, number of categories)
    """
    return np.exp(support - np.sum(support, axis=1)[:, None])


def sample_binary(on_probabilities):
    """Sample activations ON=1 (OFF=0) from probabilities sigmoid probabilities

    Args:
        on_probabilities: (np.ndarray): activation probabilities shape=(size of mini-batch, size of layer)

    Returns:
        (np.ndarray): activations (size of mini-batch, size of layer)
    """
    return 1. * (on_probabilities >= np.random.random_sample(
        size=on_probabilities.shape))


def sample_categorical(probabilities):
    """Sample one-hot activations from a categorical probabilities

    Args:
        probabilities (np.ndarray): activation probabilities shape=(size of mini-batch, number of categories)

    Returns:
        activations (np.ndarray): one hot encoded argmax probability (size of mini-batch, number of categories)
    """
    cumsum = np.cumsum(probabilities,axis=1)
    rand = np.random.random_sample(size=probabilities.shape[0])[:, None]
    activations = np.zeros(probabilities.shape)
    activations[range(probabilities.shape[0]),
            np.argmax((cumsum >= rand), axis=1)] = 1
    return activations


def load_idxfile(filename):
    """Load idx file format.

    Args:
        filename (str)

    Returns:
        data (np.ndarray)

    Note: For more information : http://yann.lecun.com/exdb/mnist/
    """
    with open(filename,'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0 :
           raise Exception('Invalid idx file: unexpected magic number!')
        dtype, ndim = ord(_file.read(1)), ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder(
            '>')).reshape(shape)

    return data


def read_mnist(dim=[28,28], n_train=60000, n_test=1000):
    """Read mnist train and test data.

    Args:
        dim (list): Dimensions of the images
        n_train (int): Number of training images
        n_test (int): Number of test images

    Returns:
        train_imgs (np.ndarray): The training images
        train_lbls_1hot (np.ndarray): One-hot-encoded training labels
        test_imgs (np.ndarray): The test images
        test_lbls_1hot (np.ndarray): One-hot-encoded test labels

    Note: Images are normalized to be in range [0,1]
    """
    train_imgs = load_idxfile("../data/train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1,dim[0]*dim[1])

    train_lbls = load_idxfile("../data/train-labels-idx1-ubyte")
    train_lbls_1hot = np.zeros((len(train_lbls),10),dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)),train_lbls] = 1.

    test_imgs = load_idxfile("../data/t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1,dim[0]*dim[1])

    test_lbls = load_idxfile("../data/t10k-labels-idx1-ubyte")
    test_lbls_1hot = np.zeros((len(test_lbls),10),dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)),test_lbls] = 1.

    return train_imgs[:n_train], train_lbls_1hot[:n_train], test_imgs[:n_test], \
            test_lbls_1hot[:n_test]


def viz_rf(weights, it, grid):
    """Visualize receptive fields and save

    Args:
        weights (np.ndarray): The weight matrix
        it (int): The current iteration
        grid (list): the dimensions of the tgrid for the plot
    """
    fig, axs = plt.subplots(grid[0], grid[1], figsize=(grid[1], grid[0]))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    imax = abs(weights).max()
    for x in range(grid[0]):
        for y in range(grid[1]):
            axs[x,y].set_xticks([]);
            axs[x,y].set_yticks([]);
            axs[x,y].imshow(weights[:,:,y+grid[1]*x], cmap="bwr", vmin=-imax,
                    vmax=imax, interpolation=None)
    plt.savefig("plots_and_animations/rf.iter%06d.png"%it)
    plt.close('all')


def stitch_video(fig, imgs):
    """Stitches a list of images and returns an animation object

    Args:
        fig (matplotlib.figure.Figure): Plot
        imgs (list): List of images

    Returns:
        (matplotlib.animation.ArtistAnimation): The stitched video
    """
    return animation.ArtistAnimation(fig, imgs, interval=100, blit=True,
            repeat=False)
