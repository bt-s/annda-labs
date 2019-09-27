#!/usr/bin/python3

"""3_6.py Containing the code for lab 3 exercise 6

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from algorithms.hopfield import HopfieldNet
from helper import *
from data import *

biases = np.arange(0, 10, 0.5)
sparsity = 0.1

for bias in biases:
    print(f'Bias: {bias}')
    # Initialize the Hopfield network with async updating
    nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
            energy_convergence=True, bias=bias, binary=True)

    num_of_patterns = np.arange(1, 25, 1)
    units = 100

    for patterns in num_of_patterns:
        # Create binary patterns with a fixed sparsity
        X = create_random_patterns(patterns, units, sparsity=sparsity,
                binary=True)
        nn.train(X)

        # Try to restore original patterns
        X_star = nn.update_rule(X)

        # Print success of retrieval for various numbers of patterns
        s = [nn.arrays_equal(xs, x) for xs, x in zip(X_star, X)]
        print((f'For {patterns} patterns of {units} units, '
            f'{round(np.sum(s)/len(X)*100, 2)}% can be correctly retrieved.'))

    print()

