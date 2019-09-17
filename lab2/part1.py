#!/usr/bin/python3

"""part1.py Containing the code for the first part of lab 2

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from helper import *
from algorithms.rbfnn import RBFNN

np.random.seed(42)

NOISY_DATA = False
SIN2X = True
SQUARE2X = True

hidden_nodes_values=[0, 2, 4, 6, 63, 80]

if SIN2X:
    errors=[]
    for i in hidden_nodes_values:
        if NOISY_DATA:
            X_train, y_train = generate_data(None, step_size=0.1, sin2x=True, noise=True,
                    noise_level=[0,0.1])
            X_test, y_test = generate_data(None, step_size=0.1, data_range=(0.05, 2 * np.pi),
                    sin2x=True, noise=True, noise_level=[0,0.1])

        else:
            X_train, y_train = generate_data(None, step_size=0.1, sin2x=True)
            X_test, y_test = generate_data(None, step_size=0.1,
                    data_range=(0.05, 2*np.pi), sin2x=True)

        # Initialize the RFB neural network regressor
        regressor = RBFNN(n=i, solver="delta_rule")


        # Train the regressor
        regressor.train(X_train, y_train, variance=0.3)

        # Predict on the test set
        y_pred = regressor.predict(X_test)

        # Plot the real and the predicted curves
        plot_1d_funcs([X_train, X_test], [y_train, y_pred], names=["y_train", "y_pred"],
                title=f"sin(2x) for {i} hidden nodes", fname=f"sin(2x) for {i} hidden nodes")

        errors.append(regressor.compute_total_error(y_pred, y_test))

    plot_error_vs_rbfunits(errors,hidden_nodes_values,title="", fname="",save_plot=False)


if SQUARE2X:
    X_train, y_train = generate_data(None, step_size=0.1,  square2x=True)
    X_test, y_test = generate_data(None, step_size=0.1,
            data_range=(0.05, 2*np.pi), square2x=True)

    # Initialize the RFB neural network regressor
    regressor = RBFNN(n=10, solver="least_squares")

    # Train the regressor
    regressor.train(X_train, y_train, variance=0.3)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Plot the real and the predicted curves
    plot_1d_funcs([X_train, X_test], [y_train, y_pred],
            names=["y_train", "y_pred"], title="square(2x)")
