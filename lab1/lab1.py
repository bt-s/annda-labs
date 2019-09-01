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
APPLY_PERCEPTRON_LEARNING_RULE = True


def generate_data(n, mA, sigmaA, mB, sigmaB):
    """Generates toy data

    Args:
        n (int): Number of data points to be generated per class
        mA (float): Mean of classA
        sigmaA (float): Variabnce of classA
        mB (float): Mean of classB
        sigmaB (float): Variabnce of classB

    Returns:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB

    Note: a row in the lab description is a column in the code here, and vice versa.
    This simplifies shuffling, and doesn't have any adverse side-effects
    """
    classA, classB = np.zeros((n, 2)), np.zeros((n, 2))
    classA[:, 0] = np.random.randn(1, n) * sigmaA + mA[0]
    classA[:, 1] = np.random.randn(1, n) * sigmaA + mA[1]
    classB[:, 0] = np.random.randn(1, n) * sigmaB + mB[0]
    classB[:, 1] = np.random.randn(1, n) * sigmaB + mB[1]

    return classA, classB


def create_data_scatter_plot(classA, classB):
    """Creates a scatter plot of the input data

    Args:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB

    Returns:
        None
    """
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.xlabel("x1"), plt.ylabel("x2")
    plt.title("Linearly separable data")
    plt.show()


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


def decision_boundary_animation(classA, classB, x, W, title):
    """Draws the decision boundary

    Args:
        classA (np.ndarray): The data corresponding to class A
        classB (np.ndarray): The data corresponding to class B
        x (np.ndarray): A linspace
        W (np.ndarrat): The weight vector
        title (str): Plot title

    Returns:
        None
    """
    y = W[0]*x + W[1]*x + W[2]
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.plot(x, y, '-b', label="line")
    plt.title(title)
    plt.show()


class DeltaClassifier:
    """The delta learning classifier"""
    def __init__(self, epochs=20, eta=0.001):
        """Class constructor

        Args:
            epochs (int): Number of training epochs
            eta (float): The learning rate
        """
        self.epochs = epochs
        self.eta = eta
        self.W = np.random.rand(3, 1) # Initialize the weights

    def train(self, X, t, animate=False):
        """Train the classifier

        Args:
            X (np.ndarray): The training data
            y (np.ndarray): The targets vector
            animate (bool): Flag to determine whether to plot the decision
                            boundary on each epoch.

        Returns:
            None
        """

        for e in range(self.epochs):
            dW = - self.eta * X.T @ (X@self.W - t) # Delta rule
            self.W += dW # Update the weight vector

            if animate:
                decision_boundary_animation(classA, classB, np.linspace(-2, 2, 100),
                    self.W, title="Delta Learning Decision Boundary")


class Perceptron:
    """The perceptron learning classifier"""
    def __init__(self, epochs=20, eta=0.001):
        """Class constructor

        Args:
            epochs (int): Number of training epochs
            eta (float): The learning rate
        """
        self.epochs = epochs
        self.eta = eta
        self.W = np.random.rand(3, 1) # Initialize the weights


    def predict(self, X, threshold=0):
        """Perceptron prediction function

        Args:
            X (np.ndarray): Data point to be predicted
            threshold (float): Threshold for perceptron activation function

        Returns:
            activation (int): Perceptron output (0 or 1)
        """
        summation = X @ self.W
        if summation > 0: return 1
        else: return 0


    def train(self, X, T, animate):
        """Perceptron learning training function

        Args:
            X (np.ndarray): Training data
            T (np.ndarray): Targets
            animate (bool): Flag to determine whether to plot the decision
                            boundary on each epoch.
        """
        for _ in range(self.epochs):
            if animate:
                decision_boundary_animation(classA, classB, np.linspace(-2, 2, 100),
                        self.W, title="Perceptron Learning Decision Boundary")
            for x, t in zip(X, T):
                # t_pred must be of shape (1,)
                t_pred = np.asarray(self.predict(x)).reshape(1,)
                # Reshape required for correct broadcasting
                self.W += self.eta * ((t - t_pred) * x).reshape(3,1)


# Generate toy-data
classA, classB = generate_data(n=100, mA=[1.0, 1.0], sigmaA=0.4,
        mB=[-1.0, -0.5], sigmaB=0.4)

# Transform data to training examples and targets
X, t = create_training_examples_and_targets(classA, classB)

if SHOW_DATA_SCATTER_PLOT:
    create_data_scatter_plot(classA, classB)

if APPLY_DELTA_RULE:
    delta_learning = DeltaClassifier()
    delta_learning.train(X, t, animate=True)

if APPLY_PERCEPTRON_LEARNING_RULE:
    perceptron = Perceptron()
    perceptron.train(X, t, animate=True)

