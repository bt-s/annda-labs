#!/usr/bin/python3

"""part1.py Containing the code for the first part of lab 2

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from helper import *
from algorithms.rbfnn import RBFNN

np.random.seed(42)

SIN2X = False
SQUARE2X = True

if SIN2X:
    X_train, y_train = generate_data(None, step_size=0.1, sin2x=True)
    X_test, y_test = generate_data(None, step_size=0.1,
            data_range=(0.05, 2*np.pi), sin2x=True)

    # Initialize the RFB neural network regressor
    regressor = RBFNN(n=30, solver="delta_rule")

    # Train the regressor
    regressor.train(X_train, y_train, variance=0.3)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Plot the real and the predicted curves
    plot_1d_funcs([X_train, X_test], [y_train, y_pred],
            names=["y_train", "y_pred"], title="sin(2x)")

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
