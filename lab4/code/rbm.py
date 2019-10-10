#!/usr/bin/python3

"""rbm.py - Containing the RestrictedBoltzmannMachine class.

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology

Note: Part of this code was provided by the course coordinators.
"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


from util import *
import numpy as np
import time


class RestrictedBoltzmannMachine():
    """The Restricted Boltzmann Machine"""
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False,
            image_size=[28, 28], is_top=False, n_labels=10, batch_size=20,
            learning_rate=0.01):
        """Class constructor

        Args:
            ndim_visible (int): Number of units in visible layer
            ndim_hidden  (int): Number of units in hidden layer
            is_bottom   (bool): True only if this RBM is at the bottom of the
                                stack in a deep belief net. Used to interpret
                                visible layer as image data with dimensions
                                "image_size"
            image_size  (list): Image dimension for visible layer
            is_top      (bool): True only if this RBM is at the top of stack in
                                deep belief net. Used to interpret visible layer
                                as concatenated with "n_label" unit of label
                                data at the end
            n_label      (int): Number of label categories
            batch_size   (int): Size of mini-batch
            learning_rate (float)
        """

        self.ndim_visible, self.ndim_hidden = ndim_visible, ndim_hidden
        self.is_bottom, self.is_top = is_bottom, is_top
        self.batch_size = batch_size
        self.n_labels = n_labels

        # Initialize W ~ N(0, 0.01) (R^v, R^h)
        self.weight_vh = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_visible,self.ndim_hidden))

        # Initialize bias_v ~ N(0, 01) (R^v)
        self.bias_v = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_visible))

        # Initialize bias_h ~ N(0, 01) (R^h)
        self.bias_h = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_hidden))

        self.momentum = 0.7
        self.learning_rate = learning_rate
        self.image_size = image_size

        # In DBNs we sometimes want to return the hidden layer
        self.H = None

        # Receptive-fields, only applicable when visible layer is input data
        self.rf = {
            # Size of the grid
            "grid" : [5, 5],
            # Pick some random hidden units
            "ids" : np.random.randint(0, self.ndim_hidden, 25)
        }


    @staticmethod
    def _sigmoid(support):
        """ Sigmoid activation function that finds probabilities to turn ON each unit

        Args:
            support (np.ndarray): (size of mini-batch, size of layer)

        Returns:
            (np.ndarray): on_probabilities (size of mini-batch, size of layer)
        """
        support[support < -700] = -700
        return 1. / (1. + np.exp(-support))


    @staticmethod
    def _softmax(support):
        """Softmax activation function that finds probabilities of each category

        Args:
            support (np.ndarray): (size of mini-batch, number of categories)

        Returns:
            (np.ndarray): on_probabilities (size of mini-batch, number of categories)
        """
        expsup = np.exp(support - np.max(support, axis=1)[:,None])

        return expsup / expsup.sum(axis=1)[:, None]


    @staticmethod
    def _sample_binary(on_probabilities):
        """Sample activations ON=1 (OFF=0) from probabilities sigmoid probabilities

        Args:
            on_probabilities: (np.ndarray): activation probabilities
                              shape=(size of mini-batch, size of layer)

        Returns:
            (np.ndarray): activations (size of mini-batch, size of layer)
        """
        return 1. * (on_probabilities >= np.random.random_sample(
            size=on_probabilities.shape))


    @staticmethod
    def _sample_categorical(probabilities):
        """Sample one-hot activations from categorical probabilities

        Args:
            probabilities (np.ndarray): activation probabilities shape of
                                        (size of mini-batch, number of categories)

        Returns:
            activations (np.ndarray): one hot encoded argmax probability
                                      (size of mini-batch, number of categories)
        """
        cumsum = np.cumsum(probabilities, axis=1)
        rand = np.random.random_sample(size=probabilities.shape[0])[:, None]
        activations = np.zeros(probabilities.shape)
        activations[range(probabilities.shape[0]),
                np.argmax((cumsum >= rand), axis=1)] = 1

        return activations


    def cd1(self, X, n_iterations=10000):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
            X (np.ndarray): training data for this rbm,
                            shape is (num_images, ndim_visible)
            n_iterations (int): number of iterations of learning (each iteration
                                learns a mini-batch)

        Note: if using 60,000 MNIST data points with a mini-batch size of 20,
              n_iterations=30000 means that we have 10 training epochs. Why?
              60,000 / 20 = 3,000 iterations is one epoch.

        Returns ():
        """
        epoch = 0
        n_samples = X.shape[0]
        n_it_per_epoch = n_samples / self.batch_size
        n_epochs = n_iterations / n_it_per_epoch

        print(f'\n> Learning CD1 for {n_epochs} epochs...')

        # We want to regenerate the complete visual and hidden layers
        V, self.H = np.zeros(X.shape), np.zeros((n_samples, self.ndim_hidden))

        # Start timing
        start = time.time()
        for it in range(n_iterations):
            mb_start = (self.batch_size * it) % n_samples
            mb_end = mb_start + self.batch_size

            # Create a data batch
            X_batch = X[mb_start:mb_end]

            # Activate and sample hidden units based on mini-batch data
            ph_prob, ph_state = self.get_h_given_v(X_batch)

            # Activate and sample visible units based on hidden state
            v_prob, v_state = self.get_v_given_h(ph_state)

            # Combine to reconstruct the full data matrix
            V[mb_start:mb_end, :] = v_prob

            # Activate and sample hidden units again, based on generated visible
            # state
            nh_prob, _ = self.get_h_given_v(v_state)

            # Reconstruct the complete hidden layer only during the last epoch
            if epoch == n_epochs-1:
               self.H[mb_start:mb_end, :] = nh_prob

            # Updating parameters
            self.update_params(X_batch, ph_prob, v_state, nh_prob)

            # Monitor the updates
            if it % n_it_per_epoch == 0:
                end = time.time()
                if self.is_top:
                    print((f'Epoch {epoch}/{int(n_epochs - 1)}: recon. err = '
                        f'{round(np.linalg.norm(X[:, :-10] - V[:, :-10]), 2)}'))
                else:
                    print((f'Epoch {epoch}/{int(n_epochs - 1)}: recon. err = '
                        f'{round(np.linalg.norm(X - V), 2)}'))

                # Visualize once in a while when visible layer is input images
                if self.is_bottom:
                    viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape(
                        (self.image_size[0], self.image_size[1], -1)), it=it,
                        grid=self.rf["grid"])

                if it != 0:
                    print(f'This epoch took {round(end - start, 2)} seconds to run.\n')

                # Restart the time
                start = time.time()

                # Reshuffle the data so that we don't always have the same
                # mini-batches
                np.random.shuffle(X)

                epoch += 1


    def update_params(self, v0, h0, v1, h1):
        """Update the weight and bias parameters

        Args:
            v0: activities or probabilities of visible layer input
            h0: activities or probabilities of hidden layer input
            v1: activities or probabilities of visible layer output
            h1: activities or probabilities of hidden layer output

        Note: All args have shape (size of mini-batch, size of respective layer)
        Note: You could also add weight decay and momentum for weight updates.
        """
        self.weight_vh += self.learning_rate * (np.dot(v0.T,
            h0) - np.dot(v1.T, h1))
        self.bias_v += self.learning_rate * np.mean(
                v0 - v1, axis=0)
        self.bias_h += self.learning_rate * np.mean(
            h0 - h1, axis=0)


    def untwine_weights(self):
        """Decouple weight matrix"""
        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(self.weight_vh.T)
        self.weight_vh = None


    def get_h_given_v(self, X_batch, directed=False):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)


        Args:
            X_batch (np.ndarray): (size of mini-batch, size of visible layer)
            directed (bool): Whether to use weight_v_to_h or weight_vh

        Returns:
            Returns:
            h_prob (np.ndarray): p(h=1|v) (size mini-batch, size hidden layer)
            h_state (np.ndarray): State of the hidden layer of shape
                                  (size mini-batch, size hidden layer)
        """
        if not directed:
            h_prob = self._sigmoid(self.bias_h + np.dot(X_batch, self.weight_vh))
        else:
            h_prob = self._sigmoid(self.bias_h + np.dot(X_batch, self.weight_v_to_h))

        h_state = self._sample_binary(h_prob)

        return h_prob, h_state


    def get_v_given_h(self, H_batch, directed=False):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Args:
           H_batch: shape is (size of mini-batch, size of hidden layer)
            directed (bool): Whether to use weight_h_to_v or weight_vh

        Returns:
           v_prob (np.ndarray): p(v=1|h) (size mini-batch, size hidden layer)
           v_state (np.ndarray): State of the visual layer of shape
                                 (size mini-batch, size hidden layer)
        """
        if self.is_top:
            # Separate H_batch into the data support and labels support
            support = self.bias_v + np.dot(H_batch, self.weight_vh.T)
            support_data = support[:, :-self.n_labels]
            support_labels = support[:, -self.n_labels:]

            # Activate and sample for the data
            v_prob_data = self._sigmoid(support_data)
            v_state_data = self._sample_binary(v_prob_data)

            # Activate and sample for the labels
            v_prob_labels = self._softmax(support_labels)
            v_state_labels = self._sample_categorical(v_prob_labels)

            # Concatenate into a normal visible layer
            v_prob = np.hstack((v_prob_data, v_prob_labels))
            v_state = np.hstack((v_state_data, v_state_labels))

        else:
            if not directed:
                v_prob = self._sigmoid(self.bias_v + np.dot(H_batch,
                    self.weight_vh.T))
            else:
                v_prob = self._sigmoid(self.bias_v + np.dot(H_batch,
                    self.weight_h_to_v))

            v_state = self._sample_binary(v_prob)

        return v_prob, v_state

