#!/usr/bin/python3

"""main.py Containing the code for lab 3

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import itertools
from algorithms.hopfield import HopfieldNet


A3_1 = True
A3_1_a = False
A3_1_b = True
A3_1_c = True


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

    # More dissimilar inputs
    x1md = [1, 1, -1, 1, -1, -1, -1, 1] # 4 out of 8 dissimilar
    x2md = [1, 1, 1, 1, 1, 1, -1, -1]   # 5 out of 8 dissimilar
    x3md = [1, -1, -1, 1, 1, 1, -1, 1]  # 5 out of 8 dissimilar
    Xmd = np.vstack((x1md, x2md, x3md))

    # Initialize the Hopfield network
    nn = HopfieldNet(zero_diag=False)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    # Sanity check that our update rule works
    # -  update_rule(X) should return X
    print(nn.arrays_equal(nn.update_rule(X), X))

    if A3_1_a:
        # Update Xd up to stable point convergence
        Xd_star = nn.recall(Xd)

        # Check whether Xd_star has converged to X
        print(nn.arrays_equal(Xd_star, X, element_wise=True))

    if A3_1_b:
        # Create an array of all possible 8-bit arrays (256, 8)
        xx = np.array([list(i) for i in itertools.product([-1, 1], repeat=8)])

        # Update xx up to stable point convergence (256, 8)
        xx_star = nn.recall(xx)
        xx_star = [tuple(row) for row in xx_star]

        # Keep all unique rows of xx_star
        attractors = np.unique(xx_star, axis=0)
        print(f'There are {attractors.shape[0]} attractors:')
        print(attractors)

    if A3_1_c:
        # Update Xd up to stable point convergence
        Xmd_star = nn.recall(Xmd)

        # Check whether Xd_star has converged to X
        print(nn.arrays_equal(Xmd_star, X, element_wise=True))

        # Check whether they converge to an attractor
        for x in Xmd_star:
            if tuple(x) in attractors:
                print("Nice")


