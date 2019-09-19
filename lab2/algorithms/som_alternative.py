#!/usr/bin/python3

"""som.py Containing the SOM class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
from collections import OrderedDict

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



props = get_animal_data("data_lab2/animals.dat")
assert props.shape == (32, 84)

names = get_animal_names('data_lab2/animalnames.txt')
assert len(names) == 32

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
        self.size = initial_neighborhood


    def initialize_weights(self, X):
        """Initialize the weights matrix
        Args:
            X (np.ndarray): Input data
        Returns:
            None (sets self.W)
        """
        self.W = np.random.uniform(0, 1, (self.h, X.shape[1]))


    def compute_differences(self, x, w):
        """Computes the differences between an input and the weight matrix
        Args:
            x (np.ndarray): Single input data point
        Returns:
            differences (np.ndarraY): Differences between an imput and self.W
        """

        return np.dot((x - w).T,(x - w))


    def train(self, X):
        """Train the SOM

        """
        for e in range(self.epochs):

            for i in range(props.shape[0]):
                animal_row = props[i,:]


                #creating a dict which will contain animal_row indices as keys and similarities as values
                differences = {}
                for i,w_row in enumerate(self.W):
                    similarity = self.compute_differences(animal_row,w_row)
                    differences[i] = similarity
                    winner_index = min(differences, key = differences.get)

                #specify neighborhood around winner
                low_index = np.where(winner_index-self.size>0, winner_index-self.size, 0)
                high_index = np.where(winner_index+self.size<100, winner_index+self.size, 99 )
                neighborhood = self.W[low_index:high_index+1,:]
                #update weights
                for weight in neighborhood:
                        weight += self.eta * (animal_row-weight)

            self.size = int(self.size - 2.5 * e)
            if self.size>=0:
                continue
            else:
                break

        #dict contains animal_row indices as keys and weight_winner_indices as values
        dict={}
        for j in range(props.shape[0]):
            animal_row = props[j, :]
            differences = {}
            for i, w_row in enumerate(self.W):
                similarity = self.compute_differences(animal_row, w_row)
                differences[i] = similarity
                winner_index = min(differences, key=differences.get)
                dict[j]=winner_index
        print(dict)

        #sorted dict based on values
        for val in sorted(dict, key=dict.get, reverse=True):
            print(names[val], dict[val]) #printing out names[val] istead of animal_row_indices for visual inspection


















som = SOM(epochs=20)
som.initialize_weights(props)
som.train(props)
