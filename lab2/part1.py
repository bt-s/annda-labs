#!/usr/bin/python3

"""part1.py Containing the code for the first part of lab 2

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from helper import *
from algorithms.rbfnn import RBFNN
from algorithms.rbfnn_multi import RBFNN_MULTI

np.random.seed(42)
NOISY_DATA = False
SIN2X = True
SQUARE2X = False
BALLISTIC = False
PLOT_REAL_VS_PRED_CURVES = True
FIND_RESIDUAL_ERRORS = True
USE_CL = False
LEAKY_CL = False

hidden_nodes_values = [20]

if SIN2X:
    errors=[]
    for i in hidden_nodes_values:
        if NOISY_DATA:
            X_train, y_train = generate_data(None, step_size=0.1, sin2x=True,
                    noise=True, noise_level=[0,0.1])
            X_test, y_test = generate_data(None, step_size=0.1,
                    data_range=(0.05, 2 * np.pi), sin2x=True, noise=False,
                    noise_level=[0,0.1])

        else:
            X_train, y_train = generate_data(None, step_size=0.1, sin2x=True)
            X_test, y_test = generate_data(None, step_size=0.1,
                    data_range=(0.05, 2*np.pi), sin2x=True)

        # Initialize the RFB neural network regressor
        regressor = RBFNN(n=i, solver="delta_rule", cl=USE_CL, leaky_learning=LEAKY_CL)

        # Train the regressor
        regressor.train(X_train, y_train, variance=0.25)

        # Predict on the test set
        y_pred = regressor.predict(X_test)

        # Plot the real and the predicted curves
        if PLOT_REAL_VS_PRED_CURVES:
            plot_1d_funcs([X_train, X_test], [y_train, y_pred],
                    names=["y_train", "y_pred"],
                    title=f"sin(2x) for {i} hidden nodes",
                    fname=f"sin(2x) for {i} hidden nodes")

        error = regressor.compute_total_error(y_pred, y_test)
        errors.append(error)
        print(f"TestError = {errors}")

        if FIND_RESIDUAL_ERRORS:
            if error < 0.001:
                print(f'Res error smaller than 0.001 with {i} units.')
            elif error < 0.01:
                print(f'Res error smaller than 0.01 with {i} units.')
            elif error < 0.1:
                print(f'Res error smaller than 0.1 with {i} units.')

    #plot_error_vs_rbfunits(errors, hidden_nodes_values, title="", fname="",
            #save_plot=False)


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

if BALLISTIC:

    # Fetch and format data from test and training datasets
    train_data = get_ballistic_data("data/ballist.dat")
    X_train = [[i[0], i[1]] for i in train_data]
    X_train = np.array(X_train)
    y_train = [[i[2], i[3]] for i in train_data]
    y_train = np.array(y_train)
    test_data = get_ballistic_data("data/balltest.dat")
    X_test = [[i[0], i[1]] for i in test_data]
    X_test = np.array(X_test)
    y_test = [[i[2], i[3]] for i in test_data]
    y_test = np.array(y_test)

    errors = []
    for i in hidden_nodes_values:
        # Initialize the RFB neural network regressor
        regressor = RBFNN_MULTI(n=i, solver="least_squares", cl=USE_CL, leaky_learning=LEAKY_CL)

        # Train the regressor
        regressor.train(X_train, y_train, variance=0.25)

        # Predict on the test set
        y_pred = regressor.predict(X_test)

        plot_in_inputspace(X_train, regressor.H, title="Plotting data and RBF units")
        # Plot the real and the predicted curves
        if PLOT_REAL_VS_PRED_CURVES:
            plot_1d_funcs([X_train, X_test], [y_train, y_pred],
                          names=["y_train", "y_pred"],
                          title=f"sin(2x) for {i} hidden nodes",
                          fname=f"sin(2x) for {i} hidden nodes")

        error = regressor.compute_total_error(y_pred, y_test)
        errors.append(error)
        print(errors)

    if FIND_RESIDUAL_ERRORS:
        if error < 0.001:
            print(f'Res error smaller than 0.001 with {i} units.')
        elif error < 0.01:
            print(f'Res error smaller than 0.01 with {i} units.')
        elif error < 0.1:
            print(f'Res error smaller than 0.1 with {i} units.')
