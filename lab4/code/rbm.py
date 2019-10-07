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
            image_size  (list): Image dimension for visible layer
            n_label      (int): Number of label categories
            batch_size   (int): Size of mini-batch
            learning_rate (float)
        """

        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden
        self.batch_size = batch_size

        # Initialize W ~ N(0, 0.01) (R^v, R^h)
        self.weight_vh = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_visible,self.ndim_hidden))

        # Initialize bias_v ~ N(0, 01) (R^v)
        self.bias_v = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_visible))

        # Initialize bias_h ~ N(0, 01) (R^h)
        self.bias_h = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_hidden))

        self.learning_rate = learning_rate
        self.image_size = image_size

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
        return 1. / (1. + np.exp(-support))


    @staticmethod
    def _softmax(support):
        """Softmax activation function that finds probabilities of each category

        Args:
            support (np.ndarray): (size of mini-batch, number of categories)

        Returns:
            (np.ndarray): on_probabilities (size of mini-batch, number of categories)
        """
        return np.exp(support - np.sum(support, axis=1)[:, None])


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
        """Sample one-hot activations from a categorical probabilities

        Args:
            probabilities (np.ndarray): activation probabilities shape of
                                        (size of mini-batch, number of categories)

        Returns:
            activations (np.ndarray): one hot encoded argmax probability
                                      (size of mini-batch, number of categories)
        """
        cumsum = np.cumsum(probabilities,axis=1)
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
        print ("\nLearning CD1...")

        n_samples = X.shape[0]

        n_it_per_mb = n_samples / self.batch_size

        # Construct a full regeneration of the input
        V = np.zeros(X.shape)
        epoch = 0

        # Start timing
        start = time.time()
        for it in range(n_iterations):
            mb_start = (self.batch_size * it) % n_samples
            mb_end = mb_start + self.batch_size

            # Create a data batch
            X_batch = X[mb_start:mb_end]

            # Activate and sample hidden units based on mini-batch data
            ph_prob = self._sigmoid(self.bias_h + np.dot(X_batch,
                self.weight_vh))
            ph_state = self._sample_binary(ph_prob)

            # Activate and sample visible units based on hidden state
            v_prob = self._sigmoid(self.bias_v + np.dot(ph_state,
                self.weight_vh.T))
            v_state = self._sample_binary(v_prob)

            V[mb_start:mb_end, :] = v_prob

            # Activate and sample hidden units again, based on generated visible
            # state
            nh_prob = self._sigmoid(self.bias_h + np.dot(v_state,
                self.weight_vh))

            # Updating parameters
            self.weight_vh += self.learning_rate * (np.dot(X_batch.T,
                ph_prob) - np.dot(v_state.T, nh_prob))
            self.bias_v += self.learning_rate * np.mean(
                    X_batch - v_state, axis=0)
            self.bias_h += self.learning_rate * np.mean(
                ph_prob - nh_prob, axis=0)

            if it % n_it_per_mb == 0:
                end = time.time()
                print((f'At epoch={epoch}, the reconstruction error is: '
                       f'{round(np.linalg.norm(X - V), 2)}'))

                # Visualize once in a while when visible layer is input images
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


            # Print progress
            #if it % self.print_period == 0:
                #print("iteration=%7d recon_loss=%4.4f" % (it,
                    #np.linalg.norm(X - V)))

    ## RBM as a belief layer: the functions below do not have to be changed
    # until running a deep belief net
    def untwine_weights(self):
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None


    def get_h_given_v_dir(self,X_batch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           X_batch: shape is (size of mini-batch, size of visible layer)

        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """
        assert self.weight_v_to_h is not None

        n_samples = X_batch.shape[0]

        return np.zeros((n_samples,self.ndim_hidden)), np.zeros((n_samples,self.ndim_hidden))


    def get_v_given_h_dir(self,H_batch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args:
           H_batch: shape is (size of mini-batch, size of hidden layer)

        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = H_batch.shape[0]

        if self.is_top:
            """Here visible layer has both data and labels. Compute total input
            for each unit (identical for both cases), and split into two parts,
            something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get
            probabilities and a sampling method to get activities. The
            probabilities as well as activities can then be concatenated back
            into a normal visible layer.
            """
            pass

        else:
            pass

        return np.zeros((n_samples,self.ndim_visible)), \
                np.zeros((n_samples,self.ndim_visible))

