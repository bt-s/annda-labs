#!/usr/bin/python3

"""main.py Containing the code for lab 3

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np

np.random.seed(42)


A3_1 = True

class HopfieldNet:
    """The Hopfield neural network"""
    def __init__(self, max_it=100):
        """Class constructor

        Args:
            max_it (int): Maximum number of update steps
        """
        self.max_it = max_it
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
        return np.asarray([1 if x >= 0 else -1 for x in X], dtype=int)


    def train(self, X):
        """Train the Hopfield network

        Args:
            X (np.ndarray): The input data

        Sets the weight matrix W
        """
        self.W = np.zeros((X.shape[1], X.shape[1]))
        for x in X:
            self.W += np.outer(x, x.T)


    def update_rule(self, X):
        """Apply the update rule a single time

        Args:
            X (np.ndarray): The input data

        Returns:
            X_star (np.ndarray): The update X
        """
        X_star = np.zeros(X.shape).astype(int)

        for i, x in enumerate(X):
            X_star[i, :] = self.sign(x@self.W)

        return X_star


    def update_till_stable_point(self, X):
        """Apply the update rule until convergence to a stable point

        Args:
            X (np.ndarray): The input data to be inferred

        Returns:
            X_new (np.ndarray): The stable point representation of the input
                                in the network
        """
        for i in range(self.max_it):
            X_new = self.update_rule(X)

            if self.arrays_equal(X, X_new):
                print(f'It took {i} iterations to converge to fixed points.')
                break

            X = X_new

        else:
            print(f'The network did not converge after {self.max_it} iterations.')

        return X_new


if A3_1:
    # Patterns to be learned
    x1 = np.asarray([-1, -1, 1, -1, 1, -1, -1, 1]).reshape((1, 8))
    x2 = np.asarray([-1, -1, -1, -1, -1, 1, -1, -1]).reshape((1, 8))
    x3 = np.asarray([-1, 1, 1, -1, -1, 1, -1, 1]).reshape((1, 8))
    X = np.vstack([x1, x2, x3])

    # Distorded patterns
    x1d = np.asarray([1, -1, 1, -1, 1, -1, -1, 1]).reshape((1, 8))
    x2d = np.asarray([1, 1, -1, -1, -1, 1, -1, -1]).reshape((1, 8))
    x3d = np.asarray([1, 1, 1, -1, 1, 1, -1, 1]).reshape((1, 8))
    Xd = np.vstack([x1d, x2d, x3d])

    # Initialize the Hopfield network
    nn = HopfieldNet()

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    # Sanity check that our update rule works
    # -  update_rule(X) should return X
    print(nn.arrays_equal(nn.update_rule(X), X))

    # Update Xd up to stable point convergence
    Xd_star = nn.update_till_stable_point(Xd)

    # Check whether Xd_star has converged to X
    print(nn.arrays_equal(Xd_star, X, element_wise=True))


