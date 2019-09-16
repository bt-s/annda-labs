#!/usr/bin/python3

"""delta_rule.py Containing the DeltaClassifier class."""

import numpy as np
from helper import *


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

    def train(self, X, T, classA, classB, animate=False, batch=False, bias=True):
        """Train the classifier

        Args:
            X (np.ndarray): The training data
            T (np.ndarray): The targets vector
            classA (np.ndarray): Data points belonging to classA
            classB (np.ndarray): Data points belonging to classB
            animate (bool): Flag to determine whether to plot the decision
                            boundary on each epoch.
            batch (bool): Flag to determine whether to use batch training,
                          as opposed to sequential training
            bias (bool): Flag to determine whether to use the bias weight

        Returns:
            None
        """
        if not bias:
            self.W = self.W[:-1]
            X = X[:, :-1]

        for e in range(self.epochs):
            if batch:
                dW = - self.eta * X.T @ (X@self.W - T) # Delta rule
                self.W += dW # Update the weight vector

                if abs(dW).sum() < 10**-4:
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
                self.W, title="Delta Learning Decision Boundary", bias=bias)

