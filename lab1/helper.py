#!/usr/bin/python3

"""helper.py Helper functions for data preparation and visualization purposes."""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def generate_data(n, mA, sigmaA, mB, sigmaB, special_case=False):
    """Generates toy data

    Args:
        n (int): Number of data points to be generated per class
        mA (np.ndarray): Means of classA
        sigmaA (float): Variabnce of classA
        mB (np.ndarray): Means of classB
        sigmaB (float): Variance of classB
        special_case (bool): Flag to specify the special case of non-linearly
                             separable data at 3.1.3

    Returns:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB

    Note: a row in the lab description is a column in the code here, and vice
    versa. This simplifies shuffling, and doesn't have any adverse side-effects.
    """
    classA, classB = np.zeros((n, 2)), np.zeros((n, 2))

    if special_case:
        classA[:, 0] = np.hstack((np.random.randn(1, round(0.5*n))
            * sigmaA - mA[0], np.random.randn(1, round(0.5*n))
            * sigmaA - mA[0]))
    else:
        classA[:, 0] = np.random.randn(1, n) * sigmaA + mA[0]

    classA[:, 1] = np.random.randn(1, n) * sigmaA + mA[1]
    classB[:, 0] = np.random.randn(1, n) * sigmaB + mB[0]
    classB[:, 1] = np.random.randn(1, n) * sigmaB + mB[1]

    return classA, classB


def subsample_data(classA, classB, percA, percB):
    """Subsample from classA and classB by percentages

    Args:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB
        percA (int): What percentage should be randomly selected from classA
        percB (int): What percentage should be randomly selected from classB

    Returns:
        classA (np.ndarray): Subsampled data points belonging to classA
        classB (np.ndarray): Subsampled data points belonging to classB
    """
    sizeA = round(classA.shape[0] / 100 * percA)
    sizeB = round(classB.shape[0] / 100 * percB)
    classA = classA[np.random.randint(classA.shape[0], size=sizeA), :]
    classB = classB[np.random.randint(classB.shape[0], size=sizeB), :]

    return classA, classB


def create_training_examples_and_targets(classA, classB):
    """Transforms toy data to trianing examples and targets

    Args:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB

    Returns:
        X (np.ndarray): Training data including bias term
        t (np.ndarray): Target vector

    """
    # Get number of data points per class
    n = classA.shape[0]

    # Add an bias row to the matrices representing the two different classes
    classA = np.hstack((classA, np.ones((n, 1))))
    classB = np.hstack((classB, np.ones((n, 1))))

    # Store the training data in a big matrix
    X = np.vstack((classA, classB))

    # Create a targets vector where classA = -1 and classB = 1
    t = np.vstack((np.ones((n, 1))*-1, np.ones((n, 1))))

    # Shuffle the training data and targets in a consistent manner
    X, t = shuffle(X, t, random_state=0)

    return X, t


def create_data_scatter_plot(classA, classB, linearly_separable=False):
    """Creates a scatter plot of the input data

    Args:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB
        linearly_separable (bool): Flag to specify whether data is linearly
                                   separable

    Returns:
        None
    """
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")

    if linearly_separable:
        plt.title("Linearly separable data")
    else:
        plt.title("Linearly inseparable data")

    plt.show()


def decision_boundary_animation(classA, classB, x, W, title, bias=True):
    """Draws the decision boundary

    Args:
        classA (np.ndarray): The data corresponding to class A
        classB (np.ndarray): The data corresponding to class B
        x (np.ndarray): A linspace
        W (np.ndarrat): The weight vector
        title (str): Plot title
        bias (bool): Flag to determine whether to use the bias weight

    Returns:
        None
    """
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")

    if bias:
        y = -(W[0]*x + W[2])/W[1] # y will coincide with x2 in the plot
        plt.title(title)
    else:
        y = -(W[0]*x)/W[1]
        plt.title(title + " without bias")

    plt.plot(x, y, '-b', label="line")
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.show()


def approx_decision_boundary_animation(classA, classB, net, title):
    """Draws the approximated decision boundary e.g. network output = 0

    Args:
        classA (np.ndarray): The data corresponding to class A
        classB (np.ndarray): The data corresponding to class B
        x (np.ndarray): A linspace
        net (np.ndarrat): The network object.
        title (str): Plot title

    Returns:
        None
    """
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")
    res = np.linspace(-2, 2, 1000)
    xlist, ylist = np.meshgrid(res, res)
    linspace = np.vstack((np.ravel(xlist), np.ravel(ylist)))
    linspace = np.vstack((linspace, np.ones((1, len(linspace[0])))))
    linspace = np.transpose(linspace)
    Z = net.predict(net.forward_pass(linspace)[1])
    Z = np.reshape(Z, (len(xlist), len(xlist[0])))
    plt.contour(res, res, Z, [0], color='black')
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.title(title)
    plt.show()
