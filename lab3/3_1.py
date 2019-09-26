#!/usr/bin/python3

"""3_1.py Containing the code for lab 3 exercise 1

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import itertools
from algorithms.hopfield import HopfieldNet
from data import *


A3_1_A = False
A3_1_B = False
A3_1_C = False

X = load_X()
Xd = load_Xd()
Xmd = load_Xmd()

# Initialize the Hopfield network
nn = HopfieldNet(zero_diag=False)

# Train the Hopfield network on the patterns to be learned
nn.train(X)

# Sanity check that our update rule works
# -  update_rule(X) should return X
assert nn.arrays_equal(nn.update_rule(X), X)

if A3_1_A:
    # Update Xd up to stable point convergence
    Xd_star = nn.recall(Xd)

    # Check whether Xd_star has converged to X
    print(nn.arrays_equal(Xd_star, X, element_wise=True))

if A3_1_B:
    # Create an array of all possible 8-bit arrays (256, 8)
    xx = np.array([list(i) for i in itertools.product([-1, 1], repeat=8)])

    # Update xx up to stable point convergence (256, 8)
    xx_star = nn.recall(xx)
    xx_star = [tuple(row) for row in xx_star]

    # Keep all unique rows of xx_star
    attractors = np.unique(xx_star, axis=0)
    print(f'There are {attractors.shape[0]} attractors:')
    print(attractors)

if A3_1_C:
    # Update Xd up to stable point convergence
    Xmd_star = nn.recall(Xmd)

    # Check whether Xd_star has converged to X
    print(nn.arrays_equal(Xmd_star, X, element_wise=True))

    # Check whether they converge to an attractor
    for x in Xmd_star:
        if tuple(x) in attractors:
            print("I converge to an attractor.")

