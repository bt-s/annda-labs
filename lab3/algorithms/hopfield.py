#!/usr/bin/python3

"""hopfield.py Containing the HopfieldNet class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np


class HopfieldNet:
    """The Hopfield neural network"""
    def __init__(self, max_it=100, zero_diag=False, asyn=False):
        """Class constructor

        Args:
            max_it (int): Maximum number of update steps
            zero_diag (bool): Whether to set the diagonal of the weight
                                    matrix to all zeros
            asyn (bool): Determines whether to update asynchronously
        """
        self.max_it = max_it
        self.zero_diag = zero_diag
        self.asyn = asyn
        self.W = None


    @staticmethod
    def arrays_equal(a1, a2, element_wise=False):
        """Check for array equality

        Args:
            a1 (np.ndarray): Array 1
            a2 (np.ndarray): Array 2
            element_wise (bool): Whether we want to check element-wise or
                                 for the complete array

        Returns:
            (bool OR np.ndarray of bool): Boolean(s) representing array equality)
        """
        if element_wise: return a1 == a2
        else: return (a1 == a2).all()


    @staticmethod
    def sign(X):
        """Computes the sign function

        Args:
            X (np.ndarray): Input to the sign function

        Returns:
            (np.ndarray): Array of integers
        """
        if type(X) == np.int64:
            return 1 if X >= 0 else -1

        else:
            return np.asarray([1 if x >= 0 else -1 for x in X], dtype=int)


    def train(self, X):
        """Train the Hopfield network

        Args:
            X (np.ndarray): The input data

        Sets the weight matrix W
        """
        self.W = np.dot(X.T, X)
        if self.zero_diag: np.fill_diagonal(self.W, 0)


    def update_rule(self, X):
        """Apply the update rule a single time

        Args:
            X (np.ndarray): The input data

        Returns:
            X (np.ndarray): The updated X
        """
        for i, x in enumerate(X):
            if self.asyn:
                for _ in range(100): # 100 is just randomly chosen
                    idx = np.random.randint(0, len(x))
                    X[i, idx] = self.sign(x@self.W[idx])

            else:
                X[i, :] = self.sign(x@self.W)

        return X


    def recall(self, X):
        """Apply the update rule until convergence to a stable point

        Args:
            X (np.ndarray): The input data to be inferred

        Returns:
            X_new (np.ndarray): The stable point representation of the input
                                in the network
        """
        for i in range(self.max_it):
            X_new = self.update_rule(X)
            if self.asyn:
                pass
                # TODO: Insert convergence criterion for async updating  here
            else:
                if self.arrays_equal(X, X_new):
                    print(f'It took {i} iterations to converge to fixed points.')
                    break

            X = X_new

        else:
            print(f'The network did not converge after {self.max_it} iterations.')

        return X_new

