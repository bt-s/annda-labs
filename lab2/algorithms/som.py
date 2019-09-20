#!/usr/bin/python3

"""som.py Containing the SOM class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
from collections import OrderedDict
from helper import *
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


    def train(self, X, names=None, districts=None, parties=None, sexes=None):
        """Train the SOM

        Args:
            X (dict): containing the input data
                - key: props; np.ndarray of shape (32, 84)
                - key: names; list of length 32
            names (list): List of names
            districts (list): List of districts
            parties (list): List of parties
            sexes (list): List of sexes
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

        if self.problem == 3:
            for e in range(self.epochs):
                mps = {}
                for i, name in zip(range(len(X)), names):
                    differences = self.compute_differences(X[i])
                    mps[name] = {"ix": np.argmin(differences),
                                 "min": np.min(differences)}

                    # Turn index into coordinate
                    coord = convert_int_to_grid_point(mps[name]["ix"])

                    # Store all Manhatten distances in a dictionary
                    distances = {}
                    for i in range(100):
                        distances[i] = compute_manhattan_dist(coord,
                                convert_int_to_grid_point(i))

                    # Sort the dictionary based on the values
                    sorted_distances = sorted(distances.items(), key=lambda kv: kv[1])

                    # Only keep the indexes sorted on distance
                    sorted_ixs = [x[0] for x in sorted_distances]
                    neighborhood = sorted_ixs[1:self.neighborhood+1]

                    # Update the weights
                    self.W[neighborhood] += self.eta * (X[i] - self.W[neighborhood])

                self.neighborhood -= 4

            # Put 1D position, coordinate, district, party and sex in a dictionary with
            # names as keys
            pos = OrderedDict()
            for i, name, district, party, sex in zip(
                    range(len(X)), names, districts, parties, sexes):
                position = np.argmin(self.compute_differences(X[i]))
                pos[name] = [position, convert_int_to_grid_point(position), district,
                        party, sex]

            # Topologically sort the above dictionary
            sorted_positions = sorted(pos.items(), key=lambda kv: kv[1])

            # Create sorted lists
            names = [x[0] for x in sorted_positions]
            coords = [x[1][1] for x in sorted_positions]
            xcoords = [x[1][1][0] for x in sorted_positions]
            ycoords = [x[1][1][1] for x in sorted_positions]
            districts = [x[1][2] for x in sorted_positions]
            parties = [x[1][3] for x in sorted_positions]
            sexes = [x[1][4] for x in sorted_positions]

            # Create a dict that shows the gender distribution per coordinate on
            # the 2D canvas
            sexes_counts = OrderedDict()
            for coord in set(coords):
                sexes_counts[coord] = {"woman": 0, "man": 0}

                for x, y in zip(coords, sexes):
                    if x == coord:
                        if y == 0:
                            sexes_counts[coord]["woman"] += 1
                        else:
                            sexes_counts[coord]["man"] += 1
                sexes_counts[coord] = {k: v for k, v in \
                        sexes_counts[coord].items() if v is not 0}


            # Create a dict that shows the party distribution per coordinate on
            # the 2D canvas
            parties_counts = OrderedDict()
            for coord in set(coords):
                parties_counts[coord] = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, \
                        "5": 0, "6": 0, "7": 0}

                for x, y in zip(coords, parties):
                    if x == coord:
                        parties_counts[coord][str(y)] += 1

                parties_counts[coord] = {k: v for k, v in \
                        parties_counts[coord].items() if v is not 0}

            # Create a dict that shows the districts distribution per coordinate on
            # the 2D canvas
            districts_counts = OrderedDict()
            for coord in set(coords):
                districts_counts[coord] = {"0": 0, "1": 0, "2": 0, "3": 0, \
                        "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0, \
                        "10": 0, "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, \
                        "16": 0, "17": 0, "18": 0, "19": 0, "20": 0, "21": 0, \
                        "22": 0, "23": 0, "24": 0, "25": 0, "26": 0, "27": 0, \
                        "28": 0, "29": 0}

                for x, y in zip(coords, parties):
                    if x == coord:
                        districts_counts[coord][str(y)] += 1

                districts_counts[coord] = {k: v for k, v \
                        in districts_counts[coord].items() if v is not 0}


            mps_plot(coords, districts_counts, "Districts",
                    "plots/districts.pdf", save_plot=True)
            mps_plot(coords, parties_counts, fname="plots/parties.pdf", save_plot=True,
                    title="Parties: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'")
            mps_plot(coords, sexes_counts, "Sex", fname="plots/sex.pdf",
                    save_plot=True)

