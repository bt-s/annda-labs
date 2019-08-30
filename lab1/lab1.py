#!/usr/bin/python3

"""lab1.py Containing the code for lab1

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(42)

# Flags to decide which part of  the program should be run
SHOW_DATA_SCATTER_PLOT = True
APPLY_DELTA_RULE = True

### Classification with a single-layer perceptron
## 3.1.1 Generation of linearly-separable data
n = 100
mA, sigmaA = [1.0, 1.0], 0.4
mB, sigmaB = [-1.0, -0.5], 0.4

# Note: a row in the lab description is a column in the code here, and vice versa
# This simplifies shuffling, and doesn't have any adverse side-effects
classA, classB = np.zeros((n, 2)), np.zeros((n, 2))
classA[:, 0] = np.random.randn(1, n) * sigmaA + mA[0]
classA[:, 1] = np.random.randn(1, n) * sigmaA + mA[1]
classB[:, 0] = np.random.randn(1, n) * sigmaB + mB[0]
classB[:, 1] = np.random.randn(1, n) * sigmaB + mB[1]

# Create a scatter plot of the data
if SHOW_DATA_SCATTER_PLOT:
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.xlabel("x1"), plt.ylabel("x2")
    plt.title("Linearly separable data")
    print(type(plt))
    plt.show()

## 3.1.2 Classification with a single-layer perceptron and analysis
# Add an bias row to the matrices representing the two different classes
classA = np.hstack((classA, np.ones((n, 1))))
classB = np.hstack((classB, np.ones((n, 1))))

# Store the training data in a big matrix
X = np.vstack((classA, classB))

# Create a targets vector where classA = -1 and classB = 1
t = np.vstack((np.ones((n, 1))*-1, np.ones((n, 1))))

# Shuffle the training data and targets in a consistent manner
X, t = shuffle(X, t, random_state=0)

# Initialize the weight matrix by randomly sampling from a normal distribution
W = np.random.rand(3, 1)

# Delta Rule:
# Set the learning rate and number of epochs
eta, epochs = 0.001, 15
# Create a linspace for drawing a line
x = np.linspace(-2, 2, 100)


def decision_boundary_animation(classA, classB, x, W):
    """Draws the decision boundary

    Args:
        classA (np.ndarray): The data corresponding to class A
        classB (np.ndarray): The data corresponding to class B
        x (np.ndarray): A linspace
        W (np.ndarrat): The weight vector

    Returns:
        None
    """
    y = W[0]*x + W[1]*x + W[2]
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.plot(x, y, '-b', label="line")
    plt.show()


# The Delta Rule algorithm
if APPLY_DELTA_RULE:
    for e in range(epochs):
        dW = - eta * X.T @ (X@W - t) # Delta rule
        W += dW
        decision_boundary_animation(classA, classB, x, W)

