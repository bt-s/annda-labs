#!/usr/bin/python3

"""main.py Containing the code for lab 3

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from algorithms.hopfield import HopfieldNet


A3_1 = True


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
    nn = HopfieldNet(zero_diag=False)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    # Sanity check that our update rule works
    # -  update_rule(X) should return X
    print(nn.arrays_equal(nn.update_rule(X), X))

    # Update Xd up to stable point convergence
    Xd_star = nn.recall(Xd)

    # Check whether Xd_star has converged to X
    print(nn.arrays_equal(Xd_star, X, element_wise=True))


