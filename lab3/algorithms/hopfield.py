#!/usr/bin/python3

"""hopfield.py Containing the HopfieldNet class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np


class HopfieldNet:
    """The Hopfield neural network"""
    def __init__(self, max_it=100, zero_diag=False, asyn=False, all_units=False,
            energy_convergence=False, compute_energy_per_iteration=False,
            normal_dist_W=False, symmetric_W=False, bias=0):
        """Class constructor

        Args:
            max_it (int): Maximum number of update steps
            zero_diag (bool): Whether to set the diagonal of the weight
                                    matrix to all zeros
            asyn (bool): Determines whether to update asynchronously
            all_units (bool): Determines whether all units should be updated
                              asynchronously. If not, N units are randomly
                              sampled with replacement and updated.
            energy_convergence (bool): Determines whether the state's energy is
                                       used for as a convergence criterion
            compute_energy_per_iteration (bool) Determines whether to compute
                                                the energy after each iteration
            normal_dist_W (bool) Determines whether to initialize W to normally
                                 distributed random numbers
            symmetric_W (bool) Determines whether W should be symmetric
            bias (float): Bias of the network
        """
        self.max_it = max_it
        self.zero_diag = zero_diag
        self.asyn = asyn
        self.all_units = all_units
        self.energy_convergence = energy_convergence
        self.compute_energy_per_iteration = compute_energy_per_iteration
        self.normal_dist_W = normal_dist_W
        self.symmetric_W = symmetric_W
        self.bias = bias
        self.W, self.P, self.N = None, None, None


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
    def check_symmetric(M, rtol=1e-05, atol=1e-08):
        """Checks if matrix M is symmetric

        Args:
            M (np.ndarray): Input matrix
            rtol (int): Relative tolerance
            atol (int): Absolute tolerance

        Returns:
            (bool): Whether M is symmetric
        """
        return np.allclose(M, M.T, rtol=rtol, atol=atol)


    @staticmethod
    def sign(X):
        """Computes the sign function

        Args:
            X (np.ndarray): Input to the sign function

        Returns:
            (np.ndarray): Array of integers
        """
        if type(X) == np.int64 or type(X) == np.float64:
            return 1 if X >= 0 else -1

        else:
            return np.asarray([1 if x >= 0 else -1 for x in X], dtype=int)


    def energy(self, state):
        """Computes the energy for a certain state

        Args:
            state (np.ndarray): A network state

        Returns:
            E (float): The energy of the state
        """
        if len(state.shape) > 1:
            E = 0
            for s in state:
                E -= self.W@s @ s
        else:
            E = - self.W@state @ state

        return E


    def train(self, X):
        """Train the Hopfield network

        Args:
            X (np.ndarray): The input data

        Sets the weight matrix W
        """
        self.P, self.N = X.shape[0], X.shape[1]

        if not self.normal_dist_W:
            self.W = 1/self.N * X.T@X
        else:
            self.W = np.random.normal(0, 1,
                    len((X.T@X).flatten())).reshape((X.T@X).shape)
            if self.symmetric_W:
                self.W = 0.5 * (self.W + self.W.T)
                assert self.check_symmetric(self.W)

        if self.zero_diag: np.fill_diagonal(self.W, 0)


    def update_rule(self, X):
        """Apply the update rule a single time

        Args:
            X (np.ndarray): The input data

        Returns:
            X (np.ndarray): The updated X
        """
        X_new = np.copy(X)
        for i, x in enumerate(X):
            if self.asyn:
                if self.all_units:
                    indices = np.arange(self.N)
                    np.random.shuffle(indices)
                    # Update each unit exactly once, but in random error
                    for idx in indices:
                        X_new[i, idx] = self.sign(x@self.W[idx] + self.bias)
                else:
                    # Sample units with replacement and do N updates
                    for _ in range(self.N):
                        idx = np.random.randint(0, self.N)
                        X_new[i, idx] = self.sign(x@self.W[idx] + self.bias)

            else:
                X_new[i, :] = self.sign(x@self.W)

        return X_new


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

            if self.compute_energy_per_iteration:
                print(f'Iteration {i}, E: {self.energy(X_new)}')

            if self.energy_convergence:
                if not self.energy(X_new) < self.energy(X):
                    print(f'It took {i} iterations to converge to fixed points.')
                    break
            else:
                if self.arrays_equal(X, X_new):
                    print(f'It took {i} iterations to converge to fixed points.')
                    break

            X = X_new

        else:
            print(f'The network did not converge after {self.max_it} iterations.')

        return X_new
