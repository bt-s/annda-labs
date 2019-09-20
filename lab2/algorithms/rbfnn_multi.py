#!/usr/bin/python3

"""rbfnn.py Containing the RBFNN class"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np


class RBFNN_MULTI:
    """The Radial Basis Function neural network"""
    def __init__(self, n=30, eta=0.001, epochs=5000, solver="least_squares",
            cl=False, leaky_learning=False, neighbour_count=10, cl_strat=2, cl_iterations=5000):
        """Class constructor

        Args:
            n (int): Number of hidden nodes
                     Note that N is the number of data points!
            eta (float): The learning rate
            epochs (int): The number of training epochs if using the delta rule
            solver (str): Specifies what solver to use; either least_squares
                          or the delta rule.
            cl (bool): Specifies whether to employ competitive learning
            cl_strat (int): Specifies which competitive learning strategy to use
                            has to be one of 1, 2
            cl_iterations (int): The number of iterations in the competitive
                                 learning loop
        """
        self.n = n
        self.eta = eta
        self.solver = solver
        self.Phi = None
        self.cl = cl
        self.leaky_learning = leaky_learning
        self.neighbour_count = neighbour_count
        self.cl_strat = cl_strat
        self.cl_iterations = cl_iterations

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
        return sum(np.exp(-1 * (x - h[:-1])**2 / (2*h[-1])))


    def initialize_hidden_nodes(self, X, variance):
        """Initialzie the hidden units

        Args:
            x_range (tuple): Range for the sampling of x values
            variance (float): The variance of the hidden units

        Sets self.H:
            self.H (np.ndarray): Array of means and variance of the hidden nodes
        """

        # Get the maximum values of each column i.e. along axis 0
        maxInColumns = np.amax(X, axis=0)
        minInColumns = np.amin(X, axis=0)
        # Pre-allocate the array containing the hidden nodes means and variances
        self.H = np.zeros((self.n, X.shape[1]+1))

        self.H[:, -1] = variance
        for i in range(self.n):
            # Sample a mean from a uniform distribution
            for xdim in range(X.shape[1]):
                self.H[i][xdim] = np.random.uniform(minInColumns[xdim], maxInColumns[xdim])


    def competitive_learning(self, X):
        """ Update the means of the hidden units by competitive learning

        Args:
            X (np.ndarray): Input data

        Updates self.H based on CL
        """



        for i in range(self.cl_iterations):
            x_ind = np.random.choice(range(X.shape[0]))
            x = X[x_ind]
            if self.leaky_learning:

                self.H = self.sort_neighbours(x) # Sort the Hidden nodes based on proximity to x
                closest_dist = abs(abs(x) - abs(self.H[0, 0]))
                for i in range(min(self.n, self.neighbour_count)):
                    if self.cl_strat == 1:
                        self.H[i][:-1] += (closest_dist/abs(abs(x) - abs(self.H[i, :-1]))) * self.eta * x
                    elif self.cl_strat == 2:
                        self.H[i][:-1] += (closest_dist/abs(abs(x) - abs(self.H[i, :-1]))) * self.eta * (abs(x) - abs(self.H[i][:-1]))
                    else:
                        raise ValueError('cl_strat has to be one of 1, 2')
                    # print(f"Updated RBF unit {i} mean to, {self.H[i][0]} for x = {x}")
                    # print(f"Falloff for this node is {(closest_dist / abs(abs(x) - abs(self.H[i, 0])))}")
            else:
                distances = abs(abs(x) - abs(self.H[:, :-1]))
                distance = [sum(part**2) for part in distances]
                winner = np.argmin(distance)

                if self.cl_strat == 1:
                    self.H[winner][:-1] += self.eta * x
                elif self.cl_strat == 2:
                    self.H[winner][:-1] += self.eta * (abs(x) - abs(self.H[winner][:-1]))
                else:
                    raise ValueError('cl_strat has to be one of 1, 2')

    def sort_neighbours(self, x):

        proximities = abs(abs(x) - abs(self.H[:, :-1]))
        proximity = [sum(part ** 2) for part in proximities]
        zipped_pairs = zip(proximity, self.H)
        sorted_with_proximity = sorted(zipped_pairs)
        neighbours_sorted = [x for _, x in sorted_with_proximity]

        return np.array(neighbours_sorted)


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
        self.initialize_hidden_nodes(X, variance)

        if self.cl:
            self.competitive_learning(X)

        # Phi (np.ndarray): RBFs on the input (N, n)
        Phi = self.compute_phi(X)

        # Solve the weights with least squares batch learning
        if self.solver=="least_squares":
            f = f.reshape(f.shape[0], f.shape[1])

            self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(Phi.T, Phi)), Phi.T), f)

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
        return abs(((abs(f) - abs(f_hat.reshape(f_hat.shape[0], f_hat.shape[1])))).mean())




