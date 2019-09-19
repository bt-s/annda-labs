
"""helper.py Helper functions for data preparation and visualization purposes."""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import time
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
        sin2x=False, square2x=False, noise=False, noise_level=(0, 0.3)):
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
        noise (bool): Specifies whether to corrupt the data using Gaussian noise
        noise_level (tuple): Contains mean and variance of Gaussian noise

    Returns:
        x (np.ndarray): Array of input data points
        y (np.ndarray): Array of output data points
    """
    if random:
        # Sample randomly from data range
        x = np.random.uniform(data_range[0], data_range[1], size=(n))
        if noise:
            x += np.random.normal(noise_level[0], noise_level[1], x.shape)

    else:
        # Sample with a stepsize
        x = np.arange(data_range[0], data_range[1], step_size)
        if noise:
            x += np.random.normal(noise_level[0], noise_level[1], x.shape)

    if sin2x:
        y = np.sin(2*x)
        if noise:
            y += np.random.normal(noise_level[0], noise_level[1], y.shape)

    elif square2x:
        y = np.fromiter(map(square, x), dtype=np.int)
        if noise:
            y += np.random.normal(noise_level[0], noise_level[1], y.shape)

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
    if save_plot: plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_error_vs_rbfunits(errors, all_rbf_units, title="", fname="",
        save_plot=False):
    """Plot the error versus the number of RBF units

    Args:
       errors (list): List of error values
       all_rbf_units (list): List of number of units
       title (str): Plot title
       fname (str): File name
       save_plot (bool): Decides whether to save the plot

    Returns:
        None
    """
    plt.xlabel("RBF units"), plt.ylabel("error")
    plt.title(title)
    plt.plot(all_rbf_units, errors, label="error")
    plt.legend(loc='best')
    plt.tight_layout()
    if save_plot: plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_cities(X, names, title="", fname="", save_plot=False):
    """Generates a plot of a 1D function

    Args:
        X (np.ndarray): Array of coordinates
        names (list): Names of the cities
        title (str): The title of the plot
        fname (str): File name for saving
        save_plot (bool): Flag to specify whether to save the plot

    Returns:
        None
    """
    plt.xlabel('x'), plt.ylabel('y')
    plt.title(title)
    for x, y, name in zip(X[:, 0], X[:, 1], names):
        plt.scatter(x, y, label=f'{name}')
        plt.text(x+.01, y+.01, name, fontsize=9)
    plt.legend(loc='best')
    plt.tight_layout()
    if save_plot: plt.savefig(fname, bbox_inches='tight')
    plt.show()


def get_animal_data(fname):
    """Gets us the animal data

    Args:
        fname (str): Path to file

    Returns:
        props (np.ndarray): Array of binary integers representing animal
                            characteristics
    """
    with open(fname, 'r') as f:
        return np.asarray(f.read().split(',')).reshape((32, 84)).astype(int)


def get_animal_names(fname):
    """Gets us the animal names

    Args:
        fname (str): Path to file

    Returns:
        names (list): Containing the animal names
    """
    with open(fname, 'r') as f:
        return f.read().replace("'", "").split()


def get_cities(fname):
    """Gets us the locations of the cities

    Args:
        fname (str): Path to file

    Returns:
        (np.ndarray): Numpy array containing the locations of the cities
    """
    with open(fname, 'r') as f:
        cities, k = [], 1
        with open('data_lab2/cities.dat', 'r') as f:
            for line in f:
                if (k > 4):
                    for word in line.split():
                        cities.append(float(word[:-1]))
                k += 1

        return np.array(cities).reshape((10, 2))

def get_votes(fname):
    pass

