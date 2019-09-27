#!/usr/bin/python3

"""3_3.py Containing the code for lab 3 exercise 3

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import itertools
from algorithms.hopfield import HopfieldNet
from helper import *
from data import *

A3_3_A = False
A3_3_B = False
A3_3_C = False
A3_3_D = False
A3_3_E = False

X = load_X()
Xd = load_Xd()

if A3_3_A:
    print("A3_3_A")
    # Initialize the Hopfield network with async updating
    nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
            energy_convergence=True)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    # Create an array of all possible 8-bit arrays (256, 8)
    xx = np.array([list(i) for i in itertools.product([-1, 1], repeat=8)])

    # Update xx up to stable point convergence (256, 8)
    xx_star = nn.recall(xx)
    xx_star = [tuple(row) for row in xx_star]

    # Keep all unique rows of xx_star
    attractors = np.unique(xx_star, axis=0)

    attractors_star = nn.recall(attractors)
    energies = []
    for i, a in enumerate(attractors_star):
        energy = nn.energy(a)
        energies.append(energy)
        print((f'The energy after convergence for attractor {i} is: '
            f'{energy}'))
    print(f'These are the various energy levels of the attractors: {set(energies)}')

if A3_3_B:
    print("A3_3_B")
    # Initialize the Hopfield network with async updating
    nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
            energy_convergence=True)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    Xd_star = nn.recall(Xd)
    for i, xd_star in enumerate(Xd_star):
        print((f'The energy after convergence for distorted pattern {i} is: '
            f'{nn.energy(xd_star)}'))

if A3_3_C:
    print("A3_3_C")
    # Initialize the Hopfield network with async updating
    nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
            energy_convergence=True, compute_energy_per_iteration=True)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    for i, xd in enumerate(Xd):
        print(f'Distorted pattern {i}:')
        xd_star = nn.recall(xd.reshape((1, 8)))
        print()

if A3_3_D:
    print("A3_3_D")
    nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
            energy_convergence=True, compute_energy_per_iteration=True,
            normal_dist_W=True)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    for i, xd in enumerate(Xd):
        print(f'Distorted pattern {i}:')
        xd_star = nn.recall(xd.reshape((1, 8)))
        print()

if A3_3_E:
    print("A3_3_E")
    nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
            energy_convergence=True, compute_energy_per_iteration=True,
            normal_dist_W=True, symmetric_W=True)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    for i, xd in enumerate(Xd):
        print(f'Distorted pattern {i}:')
        xd_star = nn.recall(xd.reshape((1, 8)))
        print()

