#!/usr/bin/python3

"""3_2.py Containing the code for lab 3 exercise 2

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import itertools
from algorithms.hopfield import HopfieldNet
from helper import *


A3_2_A = False
A3_2_B = False
A3_2_C = False
A3_2_D = False

# Load the pictures
images = get_pict_data('pict.dat')
X = images[:3, :]

# Initialize the Hopfield network
nn = HopfieldNet(zero_diag=True)

# Train the Hopfield network on the patterns to be learned
nn.train(X)

if A3_2_A:
    plot_images(images)

if A3_2_B:
    # Sanity check that our update rule works and that the three patterns are stable
    # -  update_rule(X) should return X
    print(nn.arrays_equal(nn.update_rule(X), X, element_wise=False))

if A3_2_C:
    # Update p10 and p11 to stable point convergence
    p10_star = nn.recall(images[9].reshape((1, images[9].shape[0])))
    p11_star = nn.recall(images[10].reshape((1, images[9].shape[0])))

    # Check whether p10 has converged to p1
    print(nn.arrays_equal(p10_star, images[0]))
    plot_images([p10_star])

    # Check whether p11 has converged to p2 or p3
    print(nn.arrays_equal(p11_star, images[1]))
    print(nn.arrays_equal(p11_star, images[2]))
    plot_images([p11_star])


if A3_2_D:
    # Reinitialize the Hopfield network (with async updating)
    nn = HopfieldNet(zero_diag=True, asyn=True)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    # Update p10 and p11 to stable point convergence
    p10_star = nn.recall(images[9].reshape((1, images[9].shape[0])))
    p11_star = nn.recall(images[10].reshape((1, images[9].shape[0])))

    # Check whether p10 has converged to p1
    print(nn.arrays_equal(p10_star, images[0]))
    plot_images([p10_star])

    # Check whether p11 has converged to p2 or p3
    print(nn.arrays_equal(p11_star, images[1]))
    print(nn.arrays_equal(p11_star, images[2]))
    plot_images([p11_star])
