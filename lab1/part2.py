#!/usr/bin/python3

"""part2.py"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import matplotlib.pyplot as plt
import collections
from itertools import product
import time

from helper import mackey_glass, create_mg_data, plot_mg_time_series, \
    plot_weights, mean_squared_error

from sklearn.neural_network import MLPRegressor


np.random.seed(42)

PLOT_MG_TIME_SERIES = False
RUN_EXPERIMENTS_4_3_1 = False
RUN_EXPERIMENTS_4_3_2 = False
RUN_GRID_SEARCH = False
ADD_GAUSSAIN_NOISE = False

# Generate the Mackey-Glass time-series
x = mackey_glass()

if PLOT_MG_TIME_SERIES:
    plot_mg_time_series([x], [''], title="Mackey-Glass time-series",
            fname="plots/time_series.pdf", save_plot=True)

if RUN_EXPERIMENTS_4_3_1:
    # Use a sequence of 1200 values to generate 800 training examples,
    # 200 validation examples, and 200 testing examples
    seq = x[300:1500]
    X_train, T_train = create_mg_data(seq[:800])
    X_val, T_val = create_mg_data(seq[800:-200])
    X_test, T_test = create_mg_data(seq[-200:])

    if RUN_GRID_SEARCH:
        # Perform a manual grid search on the hold-out validation set
        parameters = {
            'learning_rate_init' : [0.0001, 0.001, 0.01, 0.1],
            'learning_rate' : ['constant', 'adaptive'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
            'hidden_layer_sizes' : [(2), (3), (4), (5), (6), (7), (8)]
        }

        min_mse = np.inf
        for lri in parameters["learning_rate_init"]:
            for lr in parameters["learning_rate"]:
                for alpha in parameters["alpha"]:
                    for hls in parameters["hidden_layer_sizes"]:
                        clf = MLPRegressor(solver='lbfgs', learning_rate_init=lri,
                                learning_rate=lr, alpha=alpha, hidden_layer_sizes=hls,
                                early_stopping=True, random_state=42)
                        clf.fit(X_train, T_train)
                        T_pred = clf.predict(X_val)
                        if mean_squared_error(T_val, T_pred) < min_mse:
                            min_mse = mean_squared_error(T_val, T_pred)
                            best_estimator = clf.get_params()

        print(f'The best paramters returned by the grid search are:\n {best_estimator}')
        print(f'The minimal MSE on the validation set is: {min_mse}')

        # Fit the best estimator
        clf = MLPRegressor(**best_estimator)
        clf.fit(X_train, T_train)

        # Predict on the test set
        T_pred = clf.predict(X_test)

        # Plot T_pred against T_test
        plot_mg_time_series([T_test, T_pred], names=["T_test", "T_pred"],
                title="MG time-series: T_test and optimal T_pred")
    else:
        # Running the above code, the optimal hyper-parameters are:
        # - 'alpha': 0.01
        # - 'hidden_layer_sizes': 6
        # - 'learning_rate': 'constant'
        # - 'learning_rate_init': 0.0001
        # So let's now only fit the best classifier
        clf = MLPRegressor(solver='lbfgs', alpha=0.01, hidden_layer_sizes=6,
            learning_rate='constant', learning_rate_init=0.0001,
            early_stopping=True, random_state=42)

        # Perform a grid search fit on the training data
        clf.fit(X_train, T_train)

        # Predict on the test set
        T_pred = clf.predict(X_test)

        # Plot T_pred against T_test
        plot_mg_time_series([T_test, T_pred], names=["T_test", "T_pred"],
                title="MG time-series: T_test and optimal T_pred",
                fname="plots/2lp_t_test_t_pred.pdf", save_plot=True)

        alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
        sizes = [1, 2, 3, 4, 5, 6, 7, 8]

        # Find the effects of different alphas on the MSE
        predictions, weights = [T_test], np.zeros((7, 30))
        for ix, alpha in enumerate(alphas):
            clf = MLPRegressor(solver='lbfgs', hidden_layer_sizes=6, alpha=alpha,
                    learning_rate_init=0.0001,  early_stopping=True, random_state=42)
            clf.fit(X_train, T_train)

            # Get the prpedictions on the test set, MSE and weights
            T_pred = clf.predict(X_test)
            ws = clf.coefs_[0]
            predictions.append(T_pred)
            weights[ix] = np.ravel(ws)

        # Plot the weights based on the different alphas
        plot_weights(weights, alphas, title="Weights for different alphas",
                fname="plots/weights_different_alphas.pdf", save_plot=True)

        # Plot time-series predictions for the different alphas
        plot_mg_time_series(predictions, names=["T_test", "T_pred:alpha=0.00001",
            "T_pred:alpha=0.0001", "T_pred:alpha=0.001", "T_pred:alpha=0.01",
            "T_pred:alpha=0.1", "T_pred:alpha=1", "T_pred:alpha=10"],
            title="MG time-series: T_test and T_pred for several values of alpha.",
            fname="plots/2lp_ts_different_alphas", save_plot=True)

        # Find the effects of different size of hidden layer on MSE
        predictions = [T_test]
        for size in sizes:
            clf = MLPRegressor(solver='lbfgs', hidden_layer_sizes=size, alpha=0,
                    learning_rate_init=0.0001,  early_stopping=True)
            clf.fit(X_train, T_train)

            T_pred = clf.predict(X_test)

            predictions.append(T_pred)


        # Plot time-series predictions for the different numbers of hidden nodes
        plot_mg_time_series(predictions, names=["T_test", "T_pred:h_nodes=1",
           "T_pred:h_nodes=2", "T_pred:h_nodes=3", "T_pred:h_nodes=4",
           "T_pred:h_nodes=5", "T_pred:h_nodes=6", "T_pred:h_nodes=7",
           "T_pred:h_nodes=8"], fname="plots/2lp_ts_different_nodes",
           title="MG time-series: T_test and T_pred for different hidden nodes.",
           save_plot=True)


if RUN_EXPERIMENTS_4_3_2:
    if ADD_GAUSSAIN_NOISE:
        noise_levels = [0, 0.03, 0.09, 0.18]
    else:
        noise_levels = [0]

    for noise in noise_levels:
        #print(f'Noise {noise}')
        # Use a sequence of 1200 values to generate 800 training examples,
        # 200 validation examples, and 200 testing examples
        seq = x[300:1500] + np.random.normal(0, noise)
        X_train, T_train = create_mg_data(seq[:800])
        X_val, T_val = create_mg_data(seq[800:-200])
        X_test, T_test = create_mg_data(seq[-200:])

        if RUN_GRID_SEARCH:
            # We can base the parameter grid on the results of section 4.3.1
            parameters = {
                'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                'hidden_layer_sizes': list(product([4, 5, 6, 7, 8], repeat=2))
            }
            min_mse = np.inf
            mse_train = np.inf
            for alpha in parameters["alpha"]:
                for hls in parameters["hidden_layer_sizes"]:
                    clf = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                            learning_rate='constant', alpha=alpha, hidden_layer_sizes=hls,
                            early_stopping=True, random_state=42)
                    clf.fit(X_train, T_train)
                    T_pred = clf.predict(X_val)
                    if mean_squared_error(T_val, T_pred) < min_mse:
                        min_mse = mean_squared_error(T_val, T_pred)
                        mse_train = mean_squared_error(T_train, clf.predict(X_train))
                        mse_test = mean_squared_error(T_test, clf.predict(X_test))
                        best_estimator = clf.get_params()

            print(f'The best paramters returned by the grid search are:\n {best_estimator}')
            print(f'The minimal MSE on the validation set is: {min_mse}')
            print(f'The MSE on the training set is: {mse_train}')
            print(f'The MSE on the test set is: {mse_test}')

            # Fit the best estimator
            clf = MLPRegressor(**best_estimator)
            clf.fit(X_train, T_train)

            # Predict on the test set
            T_pred = clf.predict(X_test)

            # Fit the optimal two-layer perceptron again
            opt_tw_clf = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                    hidden_layer_sizes=6, early_stopping=True, random_state=42)
            opt_tw_clf.fit(X_train, T_train)
            T_pred_tw_opt = opt_tw_clf.predict(X_test)
            mse_train = mean_squared_error(T_train, opt_tw_clf.predict(X_train))
            mse_val = mean_squared_error(T_val, opt_tw_clf.predict(X_val))
            mse_test = mean_squared_error(T_test, T_pred_tw_opt)

            print("Optimal two-layer regressor:")
            print(f'The minimal MSE on the validation set is: {mse_val}')
            print(f'The MSE on the training set is: {mse_train}')
            print(f'The MSE on the test set is: {mse_test}')


            plot_mg_time_series([T_test, T_pred_tw_opt, T_pred],
                    names=["T_test", "T_pred_2l_opt", "T_pred"],
                    title="MG time-series: T_test and optimal T_pred for 2L and 3L perceptrons (noise: std=0.09)",
                    fname="plots/2lp_t_test_t_pred_opt_0_09_noise.pdf", save_plot=True)

        else:
            # Running hte above code, the optimal hyper-parameters are:
            # - 'alpha': 0.00001
            # - 'hidden_layer_sizes': (7, 8)
            # - 'learning_rate': 'constant'
            # - 'learning_rate_init': 0.0001
            # So let's now only fit the best classifier
            start = time.time()
            clf = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                learning_rate='constant', alpha=0.00001, hidden_layer_sizes=(8, 7),
                early_stopping=True)
            clf.fit(X_train, T_train)
            T_pred = clf.predict(X_test)
            end = time.time()
            time_three_layer = end-start
            print(f'It took {time_three_layer} seconds to run this model')
            print(f'The MSE on the test set is: {mean_squared_error(T_test, T_pred)}')

            start = time.time()
            # Fit the optimal two-layer perceptron again
            opt_tw_clf = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                    hidden_layer_sizes=6, early_stopping=True)
            opt_tw_clf.fit(X_train, T_train)
            T_pred_tw_opt = opt_tw_clf.predict(X_test)
            end = time.time()
            time_two_layer = end-start
            print(f'It took {time_two_layer} seconds to run this model')
            print(f'It took the three layer {time_three_layer / time_two_layer} as long')

            # Plot T_pred against T_test and T_pred_tw_opt
            plot_mg_time_series([T_test, T_pred_tw_opt, T_pred],
                    names=["T_test", "T_pred_2l_opt", "T_pred"],
                    title="MG time-series: T_test and optimal T_pred for 2L and 3L perceptrons",
                    fname="plots/2lp_t_test_t_pred.pdf", save_plot=True)

