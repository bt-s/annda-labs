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


def create_histogram(x, bins, title="", xlabel="", ylabel="",
        normalized=False, fname="", save_fig=False):
    """Plots a histogram

    Args:
        x (np.ndarray OR list): Input to be plotted
        bins (int): Number of bins
        title (str)
        xlabel (str)
        ylabel (str)
        normalized (bool): Whether to normalize the y-axis
        fname (str): File name for saving
        save_fig (bool): Whether to save the plot
    """
    plt.hist(x, density=normalized, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_fig: plt.savefig(fname)
    plt.show()


def plot_digit(d_arr, label):
    """Plot a handwritten digit based on its numpy array

    Args:
        d_arr (np.ndarray): The array representing the digit
        label (int): The label of the handwritten digit
    """
    two_d = (np.reshape(d_arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, cmap='gray', interpolation='nearest')
    plt.title(f'Digit: {label}')
    plt.show()


def plot_reconstruction_err(errors, fname="", save_fig=False):
    """Plots the reconstruction errors over epochs

    Args:
        fname (str): File name for saving
        save_fig (bool): Whether to save the plot
    """

    for i, _ in enumerate(errors):
        plt.plot(range(len(errors[0])), errors[i],
                label=f"Reconstruction Error Layer {i+1}")

    plt.legend()
    plt.title("The reconstruction errors over the epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Reconstruction error")
    if save_fig: plt.savefig(fname)
    plt.show()
