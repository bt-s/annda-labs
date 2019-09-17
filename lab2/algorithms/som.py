#!/usr/bin/python3

"""som.py Containing the SOM class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
from collections import OrderedDict

class SOM():
    """The Self-Organizing Maps neural network"""
    def __init__(self, h=100, eta=0.2, epochs=20, initial_neighborhood=50):
        """Class constructor

        Args:
            h (int): The number of hidden nodes
            eta (float): The learning rate
            epochs (int): The number of training epochs if using the delta rule
            initial_neighborhood (int): The amount of neighbors in the 1D case
        """
        self.h = h
        self.W = None
        self.eta = eta
        self.epochs = epochs
        self.neighborhood = initial_neighborhood


    def initialize_weights(self, X):
        """Initialize the weights matrix

        Args:
            X (np.ndarray): Input data

        Returns:
            None (sets self.W)
        """
        self.W = np.random.uniform(0, 1, (self.h, X.shape[1]))


    def compute_differences(self, x):
        """Computes the differences between an input and the weight matrix

        Args:
            x (np.ndarray): Single input data point

        Returns:
            differences (np.ndarraY): Differences between an imput and self.W
        """
        return np.einsum('ij, ij -> i', x - self.W, x - self.W)


    def train(self, X):
        """Train the SOM

        Args:
            X (dict): containing the input data
                - key: props; np.ndarray of shape (32, 84)
                - key: names; list of length 32
        """
        for e in range(self.epochs):
            self.neighborhood -= 2.5
            print(self.neighborhood)
            animals = {}
            for i, name in enumerate(X["names"]):
                differences = self.compute_differences(X["props"][i])
                animals[name] = {"ix": np.argmin(differences),
                                "min": np.min(differences)}

                left_nbors = animals[name]["ix"] - round(self.neighborhood/2)
                right_nbors = animals[name]["ix"] + round(self.neighborhood/2)

                lbound = max(0, left_nbors)
                ubound = min(self.W.shape[0], right_nbors)

                neighborhood = range(lbound, ubound)

                self.W[neighborhood] += self.eta * (X["props"][i] -  self.W[neighborhood])

        pos = OrderedDict()
        for i, name in enumerate(X["names"]):
            pos[np.argmin(self.compute_differences(X["props"][i]))] = name

        print(pos)
        print(sorted(pos))
