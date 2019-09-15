#!/usr/bin/python3

"""rbfnn.py Containing the RBFNN class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np


class RBFNN:
    """The Radial Basis Function neural network"""
    def __init__(self, n=30, eta=0.001, epochs=5000, solver="least_squares"):
        """Class constructor

        Args:
            n (int): Number of hidden nodes
                     Note that N is the number of data points!
            eta (float): The learning rate
            epochs (int): The number of training epochs if using the delta rule
            solver (str): Specifies what solver to use; either least_squares
                          or the delta rule.
        """
        self.n = n
        self.eta = eta
        self.solver = solver
        self.Phi = None

        if solver=="delta_rule":
            self.epochs = epochs
            # Initialize the weight matrix
            self.w = np.random.normal(0, 1, self.n).reshape(self.n, 1)
        else:
            self.w = None


    @staticmethod
    def _gaussian_activation(x, h):
        """Computes the Gaussian transfer function

        Args:
            x {float or np.ndarray}: Input to be transformed
            h (np.ndarray): Array of two floats:
                mu_h (float): The mean of the hidden input unit
                sigma_h (float): The variance of the hidden input unit

        Returns:
            (float or np.ndarray): Output
        """
        return np.exp(-1 * (x - h[0])**2 / (2*h[1]))


    def initialize_hidden_nodes(self, x_range, variance):
        """Initialzie the hidden units

        Args:
            x_range (tuple): Range for the sampling of x values
            variance (float): The variance of the hidden units

        Sets self.H:
            self.H (np.ndarray): Array of means and variance of the hidden nodes
        """
        # Pre-allocate the array containing the hidden nodes means and variances
        self.H = np.zeros((self.n, 2))
        self.H[:, 1] = variance
        for i in range(self.n):
            # Sample a mean from a uniform distribution
            self.H[i][0] = np.random.uniform(x_range[0], x_range[1])


    def compute_phi(self, X):
        """Computes the RBF matrix Phi

        Args:
            X (np.ndarray): Input data

        Returns:
            Phi (np.ndarray): RBFs on the input (N, n)
        """
        Phi = np.zeros((len(X), self.n))
        for i, x in enumerate(X):
            for j, h in enumerate(self.H):
                Phi[i][j] = self._gaussian_activation(x, h)

        return Phi


    def train(self, X, f, variance):
        """Train the RBF network

        Args:
            X (np.ndarray): Input data (N, 1)
            f (np.ndarray): Vector of target functions of (N, 1)
            variance (float): The variance of the hidden units

        Sets self.PHi and self.w, where self.w (np.ndarray): Weight vector (n, 1)
        """
        # self.H (np.ndarray): mean and variance of hidden units (n, 2)
        self.initialize_hidden_nodes((min(X), max(X)), variance)
        # Phi (np.ndarray): RBFs on the input (N, n)
        Phi = self.compute_phi(X)

        # Solve the weights with least squares batch learning
        if self.solver=="least_squares":
            f = f.reshape(f.shape[0], 1)

            self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(
                Phi.T, Phi)), Phi.T), f)

        # Solve the weights sequentially using the delta rule
        elif self.solver=="delta_rule":
            f_pred = self.predict(X)
            error_old = self.compute_total_error(f, f_pred)
            for e in range(self.epochs):
                for x, phi, f_x in zip(X, Phi, f):
                    f_x_hat = self.compute_f_x_hat(phi)

                    dw = self.eta * (f_x - f_x_hat) * phi.reshape(phi.shape[0], 1)
                    self.w += dw

                # Perhaps we could create a validation set and compute the total
                # error on this set to devise some kind of stopping criterion
                f_pred = self.predict(X)
                error = self.compute_total_error(f, f_pred)

                if error_old - error < 10**-2:
                    print(f'The delta rule converged after {e} epochs')
                    print(f'The total MSE on the training set is: {error}')
                    break

                error_old = error


    def compute_f_x_hat(self, phi_k):
        """Calculates the output f_hat_x for a hidden unit x_k

        Args:
            phi_k (np.ndarray): The RBF matrix corresponding to hidden unit x_k

        Returns:
            (float): Approximation of f at x_k
        """
        return np.dot(phi_k, self.w.reshape(self.w.shape[0]))


    def predict(self, X):
        """Calculates the outputs f_hat for all points X

        Args:
            X (np.ndarray): Input data

        Returns:
            (np.ndarray): Approximation of f for all points X
        """
        Phi = self.compute_phi(X)

        return np.matmul(Phi, self.w)


    def compute_total_error(self, f_hat, f):
        """Compute the total approx. error summed over an input data sequence

        Args:
            f_hat (np.ndarray): Function approximation at input data
            f (np.ndarray): Real function at input data

        Returns:
            (float): The total approximation error
        """
        return ((abs(f) - abs(f_hat.reshape(f_hat.shape[0], 1)))**2).sum()

