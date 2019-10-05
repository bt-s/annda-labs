#!/usr/bin/python3

"""rbm.py - Containing the RestrictedBoltzmannMachine class.

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology

Note: Part of this code was provided by the course coordinators.
"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


from util import *
import numpy as np


class RestrictedBoltzmannMachine():
    """The Restricted Boltzmann Machine"""
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False,
            image_size=[28,28], is_top=False, n_labels=10, batch_size=20):
        """Class constructor

        Args:
            ndim_visible (int): Number of units in visible layer
            ndim_hidden  (int): Number of units in hidden layer
            is_bottom   (bool): True only if this rbm is at the bottom of the
                                stack in a deep belief net. Used to interpret
                                visible layer as image data with dimensions
                                "image_size"
            image_size  (list): Image dimension for visible layer
            is_top      (bool): True only if this rbm is at the top of stack in
                                deep belief net. Used to interpret visible layer
                                as concatenated with "n_label" unit of label
                                data at the end
            n_label      (int): Number of label categories
            batch_size   (int): Size of mini-batch
        """

        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden
        self.is_bottom = is_bottom
        self.is_top = is_top

        if is_bottom: self.image_size = image_size
        if is_top: self.n_labels = 10

        self.batch_size = batch_size
        self.delta_bias_v = 0
        self.delta_weight_vh = 0
        self.delta_bias_h = 0

        # Initialize W ~ N(0, 0.01) (R^v, R^h)
        self.weight_vh = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_visible,self.ndim_hidden))

        # Initialize bias_v ~ N(0, 01) (R^v)
        self.bias_v = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_visible))

        # Initialize bias_h ~ N(0, 01) (R^h)
        self.bias_h = np.random.normal(loc=0.0, scale=0.01,
                size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0
        self.delta_weight_h_to_v = 0
        self.weight_v_to_h = None
        self.weight_h_to_v = None
        self.learning_rate = 0.01
        self.momentum = 0.7
        self.print_period = 5000

        # Receptive-fields, only applicable when visible layer is input data
        self.rf = {
            # iteration period to visualize
            "period" : 5000,
            # size of the grid
            "grid" : [5,5],
            # pick some random hidden units
            "ids" : np.random.randint(0, self.ndim_hidden, 25)
        }


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

        for it in range(n_iterations):
            start = (self.batch_size * it) % n_samples
            end = start + self.batch_size
            X_batch = X[start:end]

            # Not too sure about the line below; but what else should h0 be?
            # h0 should be of equal shape as h_activation...
            if it == 0:
                _, H_batch = self.get_h_given_v(X_batch)

            # Positive phase
            p_h_given_v, h_activation  = self.get_h_given_v(X_batch)

            # Negative phase
            p_v_given_h, v_activation  = self.get_v_given_h(h_activation)

            # Updating parameters
            self.update_params(X_batch, H_batch, v_activation, h_activation)

            # Visualize once in a while when visible layer is input images

            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape(
                    (self.image_size[0], self.image_size[1], -1)), it=it,
                    grid=self.rf["grid"])

            # Print progress
            if it % self.print_period == 0:
                print("iteration=%7d recon_loss=%4.4f" % (it,
                    np.linalg.norm(X - X)))

        return


    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters

        You could also add weight decay and momentum for weight updates.

        Args:
            v_0: activities or probabilities of visible layer (data to the rbm)
            h_0: activities or probabilities of hidden layer
            v_k: activities or probabilities of visible layer
            h_k: activities or probabilities of hidden layer
            all args have shape (size of mini-batch, size of respective layer)
        """
        self.delta_bias_v += self.learning_rate * np.mean(v_0 - v_k, axis=0)
        self.delta_weight_vh += self.learning_rate * \
                (np.dot(v_0.T, h_0) - np.dot(v_k.T, h_k))

        self.delta_bias_h += self.learning_rate * np.mean(h_0 - h_k, axis=0)

        # Sanity checks: the gradients of W and the biases should be of the same
        # shape as W and the biases
        assert self.bias_v.shape == self.delta_bias_v.shape
        assert self.weight_vh.shape == self.delta_weight_vh.shape
        assert self.bias_h.shape == self.delta_bias_h.shape

        self.bias_v    += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h    += self.delta_bias_h


    def get_h_given_v(self, X_batch):
        """Compute probabilities p(h=1|v) and activations h ~ p(h=1|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
            X_batch: shape is (size of mini-batch, size of visible layer)

        Returns:
            p_h_given_v (np.ndarray): p(h=1|v) of shape
                                      (size mini-batch, size hidden layer)
            h_activation (np.ndarray): Activations of the hidden layer of shape
                                       (size mini-batch, size hidden layer)
        """
        p_h_given_v = sigmoid(self.bias_h + np.dot(X_batch, self.weight_vh))
        h_activation = sample_binary(p_h_given_v)

        return p_h_given_v, h_activation


    def get_v_given_h(self, H_batch):
        """Compute probabilities p(v=1|h) and activations v ~ p(v=1|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           H_batch: shape is (size of mini-batch, size of hidden layer)

        Returns:
           p_v_given_h (np.ndarray): p(v=1|h) of shape
                                     (size mini-batch, size hidden layer)
           v_activation (np.ndarray): Activations of the visual layer of shape
                                      (size mini-batch, size hidden layer)
        """
        p_v_given_h = sigmoid(self.bias_v + np.dot(H_batch, self.weight_vh.T))

        if self.is_top:
            """Here visible layer has both data and labels. Compute total input
            for each unit (identical for both cases), and split into two parts,
            something like support[:, :-self.n_labels] and support[:, -self.n_labels:].
            Then, for both parts, use the appropriate activation function to get
            probabilities and a sampling method to get activities. The
            probabilities as well as activities can then be concatenated back
            into a normal visible layer."""
            pass

        else:
            v_activation = sample_binary(p_v_given_h)

        return p_v_given_h, v_activation


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


    def update_generate_params(self, inps, trgs, preds):
        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        self.delta_weight_h_to_v += 0
        self.delta_bias_v        += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v        += self.delta_bias_v


    def update_recognize_params(self,inps,trgs,preds):
        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        self.delta_weight_v_to_h += 0
        self.delta_bias_h        += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h        += self.delta_bias_h

