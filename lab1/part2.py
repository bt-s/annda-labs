#!/usr/bin/python3

"""part2.py"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
import matplotlib.pyplot as plt
import collections
from itertools import product
import time

from helper import mackey_glass, create_mg_data, plot_mg_time_series, \
    plot_weights

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


np.random.seed(42)

PLOT_MG_TIME_SERIES = False
RUN_EXPERIMENTS_4_3_1 = False
RUN_EXPERIMENTS_4_3_2 = True
RUN_GRID_SEARCH = False
ADD_GAUSSAIN_NOISE = False

# Generate the Mackey-Glass time-series
x = mackey_glass()

if PLOT_MG_TIME_SERIES:
    plot_mg_time_series([x], [''])

if RUN_EXPERIMENTS_4_3_1:
    # Use a sequence of 1200 values to generate 1000 training examples
    # and 200 testing examples
    seq = x[300:1500]
    X_train, T_train= create_mg_data(seq[:-200])
    X_test, T_test= create_mg_data(seq[-200:])

    if RUN_GRID_SEARCH:
        # The assignment mentions the use of a single hold-out validation set, however,
        # 3-fold cross-validation seems more appropriate. We have strong arguments to
        # argue for this approach in the report or during the presentation. Therefore,
        # use a grid-search to find the optimal hyper-parameters
        parameters = {
            'learning_rate_init' : [0.0001, 0.001, 0.01, 0.1],
            'learning_rate' : ['constant', 'adaptive'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
            'hidden_layer_sizes' : [(2), (3), (4), (5), (6), (7), (8)]
        }

        # Instantiate the MLP regressor
        mlpr = MLPRegressor(solver='lbfgs', early_stopping=True, random_state=42)

        # Instantiate the grid search
        clf = GridSearchCV(mlpr, parameters, cv=3, scoring='neg_mean_squared_error')

        # Perform a grid search fit on the training data
        clf.fit(X_train, T_train)

        # Predict on the test set
        T_pred = clf.predict(X_test)

        print(f'The best paramters returned by the grid search are:\n {clf.best_params_}')
        print(f'The MSE on the test set is: {mean_squared_error(T_test, T_pred)}')

        # Plot T_pred against T_test
        plot_mg_time_series([T_test, T_pred], names=["T_test", "T_pred"],
                title="MG time-series: T_test and optimal T_pred")
    else:
        # Running hte above code, the optimal hyper-parameters are:
        # - 'alpha': 0.001
        # - 'hidden_layer_sizes': 6
        # - 'learning_rate': 'constant'
        # - 'learning_rate_init': 0.0001
        # So let's now only fit the best classifier
        clf = MLPRegressor(solver='lbfgs', alpha=0.001, hidden_layer_sizes=6,
            learning_rate='constant', learning_rate_init=0.0001,
            early_stopping=True, random_state=42)
        # Perform a grid search fit on the training data
        clf.fit(X_train, T_train)

        # Predict on the test set
        T_pred = clf.predict(X_test)

        # Plot T_pred against T_test
        plot_mg_time_series([T_test, T_pred], names=["T_test", "T_pred"],
                title="MG time-series: T_test and optimal T_pred")

    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

    predictions, mses, weights = [T_test], [], np.zeros((7, 30))
    for ix, alpha in enumerate(alphas):
        clf = MLPRegressor(solver='lbfgs', hidden_layer_sizes=6, alpha=alpha,
        learning_rate_init=0.0001,  early_stopping=True, random_state=42)
        clf.fit(X_train, T_train)

        # Get the prpedictions on the test set, MSE and weights
        T_pred = clf.predict(X_test)
        mse = mean_squared_error(T_test, T_pred)
        ws = clf.coefs_[0]
        predictions.append(T_pred)
        mses.append(mse)
        weights[ix] = np.ravel(ws)


    # Plot the weights based on the different alphas
    plot_weights(weights, alphas)

    plot_mg_time_series(predictions, names=["T_test", "T_pred:alpha=0.00001",
        "T_pred:alpha=0.0001", "T_pred:alpha=0.001", "T_pred:alpha=0.01",
        "T_pred:alpha=0.1", "T_pred:alpha=1", "T_pred:alpha=10"],
        title="MG time-series: T_test and T_pred for several values of alpha.")

if RUN_EXPERIMENTS_4_3_2:
    if ADD_GAUSSAIN_NOISE:
        noise_levels = [0, 0.03, 0.09, 0.18]
    else:
        noise_levels = [0]

    for noise in noise_levels:
        seq = x[300:1500] + np.random.normal(0, noise)
        X_train, T_train= create_mg_data(seq[:-200])
        X_test, T_test= create_mg_data(seq[-200:])

        if RUN_GRID_SEARCH:
            # We can base the parameter grid on the results of section 4.3.1
            parameters = {
                'learning_rate' : ['constant'],
                'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                'hidden_layer_sizes': list(product([4, 5, 6, 7, 8], repeat=2))
            }

            # Instantiate the MLP regressor
            mlpr = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                early_stopping=True, random_state=42)

            # Instantiate the grid search
            clf = GridSearchCV(mlpr, parameters, cv=3, scoring='neg_mean_squared_error')

            # Perform a grid search fit on the training data
            clf.fit(X_train, T_train)

            # Predict on the test set
            T_pred = clf.predict(X_test)

            print(f'The best paramters returned by the grid search are:\n {clf.best_params_}')
            print(f'The MSE on the test set is: {mean_squared_error(T_test, T_pred)}')

            # Fit the optimal two-layer perceptron again
            opt_tw_clf = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                    hidden_layer_sizes=6, early_stopping=True, random_state=42)
            opt_tw_clf.fit(X_train, T_train)
            T_pred_tw_opt = opt_tw_clf.predict(X_test)

            # Plot T_pred against T_test and T_pred_tw_opt
            plot_mg_time_series([T_test, T_pred_tw_opt, T_pred],
                    names=["T_test", "T_pred_2l_opt", "T_pred"],
                    title="MG time-series: T_test and optimal T_pred for 2L and 3L perceptrons")

        else:
            # Running hte above code, the optimal hyper-parameters are:
            # - 'alpha': 0.001
            # - 'hidden_layer_sizes': (4, 7)
            # - 'learning_rate': 'constant'
            # - 'learning_rate_init': 0.0001
            # So let's now only fit the best classifier
            start = time.time()
            clf = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                learning_rate='constant', alpha=0.001, hidden_layer_sizes=(4,7),
                early_stopping=True, random_state=42)
            clf.fit(X_train, T_train)
            T_pred = clf.predict(X_test)
            end = time.time()
            time_three_layer = end-start
            print(f'It took {time_three_layer} seconds to run this model')
            print(f'The MSE on the test set is: {mean_squared_error(T_test, T_pred)}')

            start = time.time()
            # Fit the optimal two-layer perceptron again
            opt_tw_clf = MLPRegressor(solver='lbfgs', learning_rate_init=0.0001,
                    hidden_layer_sizes=6, early_stopping=True, random_state=42)
            opt_tw_clf.fit(X_train, T_train)
            T_pred_tw_opt = opt_tw_clf.predict(X_test)
            end = time.time()
            time_two_layer= end-start
            print(f'It took {time_two_layer} seconds to run this model')
            print(f'It took the three layer {time_three_layer / time_two_layer} as long')

            # Plot T_pred against T_test and T_pred_tw_opt
            plot_mg_time_series([T_test, T_pred_tw_opt, T_pred],
                    names=["T_test", "T_pred_2l_opt", "T_pred"],
                    title="MG time-series: T_test and optimal T_pred for 2L and 3L perceptrons")

