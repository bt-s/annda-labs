#!/usr/bin/python3

"""som.py Containing the SOM class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

def get_cities(fname):

    with open(fname, 'r') as f:
        cities = []
        k = 0
        with open('data_lab2/cities.dat', 'r') as f:
            for line in f:
                k = k + 1
                if (k > 4):
                    for word in line.split():
                        cities.append(float(word[:-1]))

        cities = np.array(cities).reshape((10, 2))

        return cities


cities = get_cities("data_lab2/cities.dat")
assert cities.shape == (10, 2)

class SOM():
    """The Self-Organizing Maps neural network"""
    def __init__(self, h=10, eta=0.2, epochs=20, initial_neighborhood=0):
        """Class constructor
        Args:
            h (int): The number of hidden nodes
            eta (float): The learning rate
            epochs (int): The number of training epochs if using the delta rule
            initial_neighborhood (int): 2d case: it has to be either 0,1 or 2
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
        self.W = np.random.uniform(0, 1, (self.h, X.shape[1])) #shape 10,2


    def compute_differences(self, x, w):
        """Computes the differences between an input and the weight matrix
        Args:
            x (np.ndarray): Single input data point
        Returns:
            differences (np.ndarraY): Differences between an imput and self.W
        """

        return np.dot((x - w).T,(x - w))

    #not necessary?
    def manhattan_distance(self,x,w):
        distX = np.abs(x[0] - w[0])
        distY = np.abs(x[1] - w[1])
        manhattan = distX + distY

    def train(self, X):
        """
        Train the SOM

        """
        for e in range(self.epochs):

            for i in range(cities.shape[0]):
                city_row = cities[i,:]


                #creating a dict which will contain animal_row indices as keys and similarities as values
                differences = {}
                for i,w_row in enumerate(self.W):
                    similarity = self.compute_differences(city_row, w_row)
                    differences[i] = similarity
                winner_index = min(differences, key = differences.get) #which row of weight matrix is the winner
                winner_weight = self.W[winner_index,:] #winner weight

                # specify neighborhood around winner
                low_index = np.where(winner_index - self.size > 0, winner_index - self.size, winner_index)
                high_index = np.where(winner_index + self.size < 100, winner_index + self.size, winner_index)


                if self.size == 2:

                    if low_index==0:
                        neighborhood_indices = [0,1,2,8,9]
                        for indice in neighborhood_indices:
                            neighborhood = np.empty((3,2))
                            neighborhood = np.vstack(( self.W[indice, :], neighborhood))


                        for weight in neighborhood:
                            weight += self.eta * (city_row - weight)

                    elif low_index==1:
                        neighborhood_indices = [0,1,2,3,9]
                        for indice in neighborhood_indices:
                            neighborhood = np.empty((3, 2))
                            neighborhood = np.vstack((self.W[indice, :], neighborhood))

                        for weight in neighborhood:
                            weight += self.eta * (city_row - weight)


                    elif high_index==8:
                        neighborhood_indices = [6,7,8,9,0]
                        for indice in neighborhood_indices:
                            neighborhood = np.empty((3, 2))
                            neighborhood = np.vstack((self.W[indice, :], neighborhood))

                        for weight in neighborhood:
                            weight += self.eta * (city_row - weight)

                    elif high_index==9:
                        neighborhood_indices = [7,8,9,0,1]
                        for indice in neighborhood_indices:
                            neighborhood = np.empty((3, 2))
                            neighborhood = np.vstack((self.W[indice, :], neighborhood))
                        for weight in neighborhood:
                            weight += self.eta * (city_row - weight)

                    else:
                        neighborhood = self.W[low_index:high_index + 1, :]
                        # update weights
                        for weight in neighborhood:
                            weight += self.eta * (city_row - weight)

                elif self.size==1:

                    if low_index == 0:
                        neighborhood_indices = [0, 1, 9]
                        for indice in neighborhood_indices:
                            neighborhood = np.empty((3, 2))
                            neighborhood = np.vstack((self.W[indice, :], neighborhood))

                        for weight in neighborhood:
                            weight += self.eta * (city_row - weight)

                    elif high_index == 9:
                        neighborhood_indices = [0, 8, 9]
                        for indice in neighborhood_indices:
                            neighborhood = np.empty((3, 2))
                            neighborhood = np.vstack((self.W[indice, :], neighborhood))
                        for weight in neighborhood:
                            weight += self.eta * (city_row - weight)

                else:
                    neighborhood = self.W[low_index:high_index + 1, :]
                    # update weights
                    for weight in neighborhood:
                        weight += self.eta * (city_row - weight)




        dict = {}   # contains city row indices as keys and weight_winner_indices as values

        for j in range(cities.shape[0]):
            city_row = cities[j, :]
            differences = {}
            for i, w_row in enumerate(self.W):
                similarity = self.compute_differences(city_row, w_row)
                differences[i] = similarity
                winner_index = min(differences, key=differences.get)
                dict[j] = winner_index
        print(dict)


        plt.xlabel('x coordinates'), plt.ylabel('y coordinates')
        plt.title('Salesman Tour, neighbor')
        for x_coord, y_coord, name in zip(cities[:, 0], cities[:, 1], range(len(cities))):
            plt.scatter(x_coord, y_coord, label=f'{name}')
        plt.plot(self.W[:,0], self.W[:,1], color='black')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('', bbox_inches='tight')
        plt.show()






        # sorted dict based on values
        sorted_dict = {}
        for val in sorted(dict, key=dict.get, reverse=True):
            sorted_dict[val] = dict[val]
        print(sorted_dict)











som2=SOM()
som2.initialize_weights(cities)
winner = som2.train(cities)


