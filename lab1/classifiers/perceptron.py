#!/usr/bin/python3

"""Perceptron.py Containing the SingleLayerPerceptorn, TwoLayerPerceptron and
MultiLayerPerceptron classes"""


__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from helper import *


class SingleLayerPerceptron:
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


    def train(self, X, T, classA, classB, animate):
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


    def train(self, X, t, classA, classB, print_acc=False, animate=False):
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
        if animate:
            approx_decision_boundary_animation(classA, classB,
                self, title="Delta Learning Decision Boundary")
