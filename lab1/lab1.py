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
SHOW_DATA_SCATTER_PLOT = False
APPLY_DELTA_RULE_BATCH = False
APPLY_DELTA_RULE_SEQUENTIAL = False
APPLY_PERCEPTRON_LEARNING_RULE = False
APPLY_TWO_LAYER_PERCEPTRON_LEARNING_RULE = True


def generate_data(n, mA, sigmaA, mB, sigmaB):
    """Generates toy data

    Args:
        n (int): Number of data points to be generated per class
        mA (np.ndarray): Means of classA
        sigmaA (float): Variabnce of classA
        mB (np.ndarray): Means of classB
        sigmaB (float): Variance of classB

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
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")
    plt.title("Linearly separable data")
    plt.show()


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
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")
    y = -(W[0]*x + W[2])/W[1] # Will coincide with x2 in the plot
    plt.plot(x, y, '-b', label="line")
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.title(title)
    plt.show()


class DeltaClassifier:
    """The delta learning classifier"""
    def __init__(self, epochs=200, eta=0.001):
        """Class constructor

        Args:
            epochs (int): Number of training epochs
            eta (float): The learning rate
        """
        self.epochs = epochs
        self.eta = eta
        self.W = np.random.rand(3, 1) # Initialize the weights

    def train(self, X, T, animate=False, batch=False):
        """Train the classifier

        Args:
            X (np.ndarray): The training data
            y (np.ndarray): The targets vector
            animate (bool): Flag to determine whether to plot the decision
                            boundary on each epoch.
            batch (bool): Flag to determine whether to use batch training,
                          as opposed to sequential training

        Returns:
            None
        """
        for e in range(self.epochs):
            if batch:
                dW = - self.eta * X.T @ (X@self.W - T) # Delta rule
                self.W += dW # Update the weight vector

                if abs(dW.sum()) < 10**-4:
                    print((f'The delta ruled converged after {e} epochs (i.e. '
                        f'abs(dW.sum() < 10**-4).'))
                    break
            else:
                dw = 0
                for i, (x, t) in enumerate(zip(X, T)):
                    dW = - self.eta * x.T * (x@self.W - t)
                    self.W += dW.reshape(3,1) # Update the weight vector

                    if abs(dW.sum()) < 10**-7:
                        print((f'The delta ruled converged after {e} epochs (i.e. '
                               f'abs(dW.sum() < 10**-4) and {i} data points.'))
                        break
                else: # Makes sure that the outer loop breaks if the inner breaks
                    continue
                break

        if animate:
            decision_boundary_animation(classA, classB, np.linspace(-2, 2, 100),
                self.W, title="Delta Learning Decision Boundary")


class Perceptron:
    """The perceptron learning classifier"""
    def __init__(self, epochs=200, eta=0.001):
        """Class constructor

        Args:
            epochs (int): Number of training epochs
            eta (float): The learning rate
        """
        self.epochs = epochs
        self.eta = eta
        self.W = np.random.rand(3, 1) # Initialize the weights


    def predict(self, x, threshold=0):
        """Perceptron prediction function

        Args:
            x (np.ndarray): Data point to be predicted
            threshold (float): Threshold for perceptron activation function

        Returns:
            activation (int): Perceptron output (0 or 1)
        """
        return 1 if (x @ self.W >= 0) else -1.


    def train(self, X, T, animate):
        """Perceptron learning training function

        Args:
            X (np.ndarray): Training data
            T (np.ndarray): Targets
            animate (bool): Flag to determine whether to plot the decision
                            boundary on each epoch.
        """
        for e in range(self.epochs):
            has_misclassification = False
            for x, t in zip(X, T):
                # t_pred must be of shape (1,)
                t_pred = self.predict(x, t)
                if t_pred != t:
                    has_misclassification = True
                # Reshape required for correct broadcasting
                self.W += self.eta * (t[0] - t_pred) / 2 * (x).reshape(3,1)
            if not has_misclassification:
                print(f'The perceptron converged after {e} epochs.')
                break


        if animate:
            decision_boundary_animation(classA, classB, np.linspace(-2, 2, 100),
                    self.W, title="Perceptron Learning Decision Boundary")


class TwoLayerPerceptron:
    """The two-layer perceptron trained with backprop (the generalized Delta
    Rule)"""
    def __init__(self, h=4, epochs=100, eta=0.001):
        """Class constructor

        Args:
            h (int): Number of hidden nodes
            epochs (int): Number of training epochs
            eta (float): The learning rate
        """
        self.h = h
        self.epochs = epochs
        self.eta = eta


    def initialize_weights(self, X, t):
        """Initialize the weight matrices

        Args:
            X (np.ndarray): Observations of shape (n, d)
            t (np.ndarray): Targets of shape (n, 1)

        Returns:
            None
        """
        # Number of data poitns and the dimensionality of data points
        n, d = X.shape[0], X.shape[1]

        # Initialize the weights
        self.V = np.random.normal(0, 1/np.sqrt(d), (d, self.h))
        self.W = np.random.normal(0, 1/np.sqrt(self.h), (self.h, 1))


    @staticmethod
    def _activation_function(x):
        """Computes the non-linear activation function

        Args:
            x {float or np.ndarray}: Input to be transformed

        Returns:
            (float or np.ndarray): Output
        """
        return 2 / (1 + np.exp(-x)) - 1


    @staticmethod
    def _d_activation_function(x):
        """Computes the derivative of the non-linear activation function

        Args:
            x {float or np.ndarray}: Input transformed by non-linear activation

        Returns:
            (float or np.ndarray): Output

        """
        return np.multiply((1+ x), (1-x)) / 2


    def forward_pass(self, X):
        """ Forward pass of the baackprop algorithm

        Args:
            X (np.ndarray): The input data

        Returns:
            h (np.ndarray): Output of the hidden layer
            o (np.ndarray): Final output
        """
        h = self._activation_function(X @ self.V)
        o = self._activation_function(h @ self.W)

        return h, o


    def predict(self, X, threshold=0):
        """Multilayer perceptron prediction function

        Args:
            X (np.ndarray): Observations to be predicted
            threshold (float): Classification threshold

        Returns:
            X (np.ndarray): Matrix of predictions corresponding to X
        """
        X[X >= threshold] = 1
        X[X < threshold] = -1

        return X


    def compute_accuracy(self, X, t):
        """Calculate training/testing accuracy

        Args:
            X (np.ndarray): Observations of shape (n, d)
            t (np.ndarray): Targets of shape (n, 1)

        Returns:
           (float): The accuracy
        """
        # Make a prediction based on the perceptron's output
        p = self.predict(self.forward_pass(X)[1])

        return 1 - np.mean(p != t)


    def train(self, X, t, print_acc=False):
        """Train the two-layer perceptron

        Args:
            X (np.ndarray): Observations of shape (n, d)
            t (np.ndarray): Targets of shape (n, 1)
            print_acc (bool): Flag to specify whether to print the accuracy
                              after each epoch

        Returns:
            None
        """
        # Initialize weights based on dimensions of observations and targets
        self.initialize_weights(X, t)

        for e in range(self.epochs):
            # Forward pass
            h, o = self.forward_pass(X)

            # Backward pass
            delta_o = np.multiply((o - t), self._d_activation_function(o))
            delta_h = np.multiply(delta_o @ self.W.T,
                    self._d_activation_function(h))

            # Update the weights
            self.V += - self.eta * X.T @ delta_h
            self.W += - self.eta * h.T @ delta_o

            # Compute the accuracy
            acc = self.compute_accuracy(X, t)
            if print_acc:
                print(f'The accuracy after epoch {e}: {acc}')

            if acc == 1.0:
                print((f'Complete convergence after {e} epochs.'))
                break


# Generate toy-data
classA, classB = generate_data(n=100, mA=[1.0, 1.0], sigmaA=0.4,
        mB=[-1.0, -0.5], sigmaB=0.4)

# Transform data to training examples and targets
X, t = create_training_examples_and_targets(classA, classB)

if SHOW_DATA_SCATTER_PLOT:
    create_data_scatter_plot(classA, classB)

if APPLY_DELTA_RULE_BATCH:
    delta_learning = DeltaClassifier()
    delta_learning.train(X, t, animate=True, batch=True)

if APPLY_DELTA_RULE_SEQUENTIAL:
    delta_learning = DeltaClassifier()
    delta_learning.train(X, t, animate=True)

if APPLY_PERCEPTRON_LEARNING_RULE:
    perceptron = Perceptron()
    perceptron.train(X, t, animate=True)

if APPLY_TWO_LAYER_PERCEPTRON_LEARNING_RULE:
    clf = TwoLayerPerceptron()
    clf.train(X, t, print_acc=True)

