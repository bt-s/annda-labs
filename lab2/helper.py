#!/usr/bin/python3

"""helper.py Helper functions for data preparation and visualization purposes."""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
import matplotlib.pyplot as plt


def square(x):
    """Box envelope for the sine wave

    Args:
        x (float): A data point

    Returns:
        (int)
    """
    return 1 if np.sin(x) >= 0 else -1


def generate_data(n, step_size, random=False, data_range=(0, 2*np.pi),
        sin2x=False, square2x=False):
    """Generates toy data

    Args:
        n (int): Number of data points to be generated (if applicable)
        step_size (float): The step size (if applicable)
        random (bool): Flag to specify whether to sample randomly
        data_range (tuple): Data range
        sin2x (bool): Flag to specify whether to generate data from sin(2x)
                      function
        square2x (bool): Flag to specify whether to generate data from
                         square(2x) function

    Returns:
        x (np.ndarray): Array of input data points
        y (np.ndarray): Array of output data points
    """
    if random:
        # Sample randomly from data range
        x = np.random.uniform(data_range[0], data_range[1], size=(n))
    else:
        # Sample with a stepsize
        x = np.arange(data_range[0], data_range[1], step_size)

    if sin2x:
        y = np.sin(2*x)

    elif square2x:
        y = np.fromiter(map(square, x), dtype=np.int)

    return x, y


def plot_1d_funcs(input_seqs, output_seqs, names, title="", fname="",
        save_plot=False):
    """Generates a plot of a 1D function

    Args:
        input_seqs (list): List of input sequences (np.ndarray)
        output_seqs (list): List of output sequences (np.ndarray)
        names (list): Names of the sequences (str)
        title (str): The title of the plot
        fname (str): File name for saving
        save_plot (bool): Flag to specify whether to save the plot

    Returns:
        None
    """
    plt.xlabel("x"), plt.ylabel("y")
    plt.title(title)

    for input_seq, output_seq, name in zip(input_seqs, output_seqs, names):
        plt.scatter(input_seq, output_seq, color='red')
        plt.plot(input_seq, output_seq, label=f'{name}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
