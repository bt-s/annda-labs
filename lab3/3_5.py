#!/usr/bin/python3

"""3_5.py Containing the code for lab 3 exercise 5

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from algorithms.hopfield import HopfieldNet
from helper import *
from data import *

NUMBER_OF_TRAINING_IMAGES = 3

A3_5_A = False
A3_5_B = False
A3_5_C = False
A3_5_D = False
A3_5_E = False

# Initialize the Hopfield network with async updating
nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
        energy_convergence=True)

if A3_5_A:
    print("A3_5_A")
    # Load the pictures
    images = get_pict_data('pict.dat')

    # Show that storing capabilities abruptly cease to exist
    for n_images in range(1, 8):
        X = images[:n_images, :]
        nn.train(X)
        # Try to restore original patterns
        X_star = nn.update_rule(X)
        s = [nn.arrays_equal(xs, x) for xs, x in zip(X_star, X)]
        print((f'For {n_images} training images of shape {X[0].shape}, '
                f'{round(np.sum(s)/len(X)*100, 2)}% can be correctly retrieved.'))

    X = images[:NUMBER_OF_TRAINING_IMAGES, :]

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    # Update p10 and p11 to stable point convergence
    p10_star = nn.recall(images[9].reshape((1, images[9].shape[0])))
    p11_star = nn.recall(images[10].reshape((1, images[9].shape[0])))

    # Check whether p10 has converged to p1
    print(f'Did p10 converge to p1? - {nn.arrays_equal(p10_star, images[0])}')
    plot_images([p10_star])

    # Check whether p11 has converged to p2 or p3
    print(f'Did p11 converge to p2? - {nn.arrays_equal(p11_star, images[1])}')
    print(f'Did p11 converge to p3? - {nn.arrays_equal(p11_star, images[2])}')
    plot_images([p11_star])

if A3_5_B:
    print("A3_5_B")

    # Show that the storing capabilities for random patterns are different
    # TODO: try for example sparsity=0.3
    for n_patterns in range(1, 200, 1):
        X = create_random_patterns(n_patterns, sparsity=0.5)
        nn.train(X)
        # Try to restore original patterns
        X_star = nn.update_rule(X)
        s = [nn.arrays_equal(xs, x) for xs, x in zip(X_star, X)]
        print((f'For {n_patterns} training images of shape {X[0].shape}, '
                f'{round(np.sum(s)/len(X)*100, 2)}% can be correctly retrieved.'))

    # Generate random training patterns
    X = create_random_patterns(32)

    # Train the Hopfield network on the patterns to be learned
    nn.train(X)

    # Set the noise level
    noise_level = 0.4

    X_prime = np.copy(X)
    for i, x in enumerate(X):
        X_prime[i] = add_noise(x, noise_level)
        x_star = nn.recall(X_prime[i].reshape(1, 1024))
        print((f'Can x_prime[{i}] be restored with {int(noise_level*100)}% noise? - '
                f'{nn.arrays_equal(x_star, x)}'))

if A3_5_C:
    print(("A3_5_C: What happens with the number of stable patterns as more are ")
          ("learned?"))
    num_of_patterns = np.arange(1, 40, 1)
    units = 100

    for patterns in num_of_patterns:
        # Generate random training patterns
        X = create_random_patterns(patterns, units, sparsity=0.5)

        # Train the Hopfield network on 100 random patterns
        nn.train(X)

        # Try to restore original patterns
        X_star = nn.update_rule(X)

        s = [nn.arrays_equal(xs, x) for xs, x in zip(X_star, X)]
        print((f'For {patterns} patterns of {units} units, '
               f'{round(np.sum(s)/len(X)*100, 2)}% can be correctly retrieved.'))

if A3_5_D:
    print(("A3_5_D: What happens if convergence to the pattern from a noisy ")
          ("version is used?"))
    if A3_5_E:
        print("A3_5_E: Added bias")
        # Initialize the Hopfield network with async updating and a bias
        bias = 0.05
        nn = HopfieldNet(max_it=100, zero_diag=True, asyn=True, all_units=True,
                energy_convergence=True, bias=bias)

    num_of_patterns = np.arange(1, 35, 1)
    units = 100

    for patterns in num_of_patterns:
        # Generate random training patterns
        X = create_random_patterns(patterns, units, sparsity=0.5)

        # Train the Hopfield network on 100 random patterns
        nn.train(X)

        # Corrupt the patterns with a little noise
        noise_level = 0.01
        X_prime = np.asarray([add_noise(x, noise_level) for \
                x in X]).reshape((patterns, units))

        # Try to restore original patterns
        X_star = nn.recall(X_prime)

        s = [nn.arrays_equal(xs, x) for xs, x in zip(X_star, X)]
        print((f'For {patterns} patterns of {units} units, '
               f'{round(np.sum(s)/len(X)*100, 2)}% can be correctly retrieved.\n'))

