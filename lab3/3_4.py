#!/usr/bin/python3

"""3_4.py Containing the code for lab 3 exercise 4

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import itertools
from algorithms.hopfield import HopfieldNet
from helper import *
from data import *

A3_4_A = True

# Load the pictures
images = get_pict_data('pict.dat')
X = images[:3, :]

if A3_4_A:
    # Initialize the Hopfield network with async updating
    nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
            energy_convergence=True)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    noise_levels = np.arange(0.01, 1, 0.01)

    for i, x in enumerate(X):
        x_old = np.copy(x)
        for nl in noise_levels:
            x_prime = add_noise(x, nl)
            x_star = nn.recall(x_prime)
            print((f'Can p{i+1} be restored with {int(nl*100)}% noise? - '
                   f'{nn.arrays_equal(x_star, x_old)}'))
            print()
