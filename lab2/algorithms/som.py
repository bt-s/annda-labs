#!/usr/bin/python3

"""som.py Containing the SOM class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

class SOM():
    """The Self-Organizing Maps neural network"""
    def __init__(self, h=100, eta=0.2, epochs=20, initial_neighborhood=50,
            problem=1):
        """Class constructor

        Args:
            h (int): The number of hidden nodes
            eta (float): The learning rate
            epochs (int): The number of training epochs if using the delta rule
            initial_neighborhood (int): The amount of neighbors in the 1D case
            problem (int): Indicate the problem on the assignment (1, 2 or 3)
        """
        self.h = h
        self.W = None
        self.eta = eta
        self.epochs = epochs
        self.neighborhood = initial_neighborhood
        self.problem = problem


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


    def train(self, X, names=None):
        """Train the SOM

        Args:
            X (dict): containing the input data
                - key: props; np.ndarray of shape (32, 84)
                - key: names; list of length 32
            names (list): List of names
        """
        if self.problem == 1:
            for e in range(self.epochs):
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

                    self.W[neighborhood] += self.eta * \
                            (X["props"][i] - self.W[neighborhood])

                self.neighborhood -= 2.5

                pos = OrderedDict()
                for i, name in enumerate(X["names"]):
                    pos[name] = np.argmin(self.compute_differences(X["props"][i]))

                print("The animals are ordered as follows:")
                print(sorted(pos.items(), key=lambda kv: kv[1]))

        if self.problem == 2:
            for e in range(self.epochs):
                cities = {}
                for i, name in zip(range(len(X)), names):
                    differences = self.compute_differences(X[i])
                    cities[name] = {"ix": np.argmin(differences),
                                    "min": np.min(differences)}

                    c = cities[name]["ix"]
                    left_nbors = int(c - self.neighborhood/2) % self.h
                    right_nbors = int(c + self.neighborhood/2) % self.h

                    if (right_nbors - left_nbors) != self.neighborhood:
                        neighborhood = list(range(left_nbors, self.h)) + \
                                list(range(0, right_nbors+1))
                    else:
                        neighborhood = list(range(left_nbors, right_nbors+1))

                    self.W[neighborhood] += self.eta * \
                            (X[i] - self.W[neighborhood])

                self.neighborhood -= 2

            pos = OrderedDict()
            for i, name in zip(range(len(X)), names):
                differences = self.compute_differences(X[i])
                pos[name] = np.argmin(self.compute_differences(X[i]))

            print("The best route is:")
            print(sorted(pos.items(), key=lambda kv: kv[1]))
