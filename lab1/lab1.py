#!/usr/bin/python3

"""lab1.py Containing the code for lab1

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"


import numpy as np
from itertools import product
from helper import *
from classifiers.delta_rule import DeltaClassifier
from classifiers.perceptron import SingleLayerPerceptron, TwoLayerPerceptron, TwoLayerFunctionApproximation
from classifiers.encoder import Encoder

np.random.seed(420)

# Flags to decide which part of  the program should be run
LINEARLY_SEPARABLE_DATA = True
LINEARLY_UNSEPARABLE_DATA_3_1_3 = False
ENCODER = False
SUBSAMPLE = False
SHOW_DATA_SCATTER_PLOT = False
APPLY_DELTA_RULE_BATCH = True
APPLY_DELTA_RULE_SEQUENTIAL = False
APPLY_PERCEPTRON_LEARNING_RULE = False
APPLY_TWO_LAYER_PERCEPTRON_LEARNING_RULE = False
FUNCTION_APPROXIMATION = False
BIAS = True


# Generate toy-data
if LINEARLY_SEPARABLE_DATA:
    classA, classB = generate_data(n=100, mA=[1.0, 1.0], sigmaA=0.4,
            mB=[-1.0, -0.5], sigmaB=0.4)

elif LINEARLY_UNSEPARABLE_DATA_3_1_3:
    classA, classB = generate_data(100, [1.0, 0.3], 0.2, [0.0, -0.1], 0.3, special_case=True)

elif FUNCTION_APPROXIMATION:
    x, y, z = generate_bell_shape_data()

else:
    classA, classB = generate_data(n=100, mA=[.5, .5], sigmaA=0.5,
            mB=[-.5, -0.5], sigmaB=0.5)

if SUBSAMPLE:
    classA_train, classB_train, classA_validation, classB_validation, = subsample_data(classA, classB, 25, 25)

# Transform data to training examples and targets
if ENCODER:
    X, t = generate_encoder_data()
elif FUNCTION_APPROXIMATION:
    X, t = bell_shape_training_examples(x, y, z)
else:
    X, t = create_training_examples_and_targets(classA, classB)


if SHOW_DATA_SCATTER_PLOT:
    if LINEARLY_SEPARABLE_DATA:
        create_data_scatter_plot(classA, classB, linearly_separable=True)
    else:
        create_data_scatter_plot(classA, classB)

if APPLY_DELTA_RULE_BATCH:
    delta_learning = DeltaClassifier(eta=0.001)
    if BIAS:
        delta_learning.train(X, t, classA, classB, animate=True, batch=True)
    else:
        delta_learning.train(X, t, classA, classB, animate=True, batch=True,
                bias=False)

if APPLY_DELTA_RULE_SEQUENTIAL:
    delta_learning = DeltaClassifier()
    delta_learning.train(X, t, classA, classB, animate=True)

if APPLY_PERCEPTRON_LEARNING_RULE:
    perceptron = SingleLayerPerceptron()
    perceptron.train(X, t, classA, classB, animate=True)

if APPLY_TWO_LAYER_PERCEPTRON_LEARNING_RULE:
    if ENCODER:
        clf = Encoder(h=3, epochs=100000, eta=0.3)
        clf.train(X, t, X, t, X, t, X, t, print_acc=True, animate=False)
        print(clf.predict(clf.forward_pass(X)[1]))
    elif SUBSAMPLE:
        X_validation, t_validation = create_training_examples_and_targets(classA_validation, classB_validation)
        X, t = create_training_examples_and_targets(classA_train, classB_train)
        clf = TwoLayerPerceptron(epochs=5000)
        clf.train(X, t, X_validation, t_validation, classA_train, classB_train,
            classA_validation, classB_validation, print_acc=True, animate=True, subsampling=SUBSAMPLE)
    elif FUNCTION_APPROXIMATION:


        X_train, T_train, X_validation, T_validation = subsample_function_data(X, t, 20)
        """
        hidden = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 50]
        clflist = []
        for i in hidden:
            print("Trying ", i , " hidden nodes")
            clf = TwoLayerFunctionApproximation(h=i, epochs=4000)
            clf.train(X, t, X_train, T_train, X_validation, T_validation,
                      print_acc=False, animate=False, subsampling=False)
            clflist.append(clf)
        minMSE = 100
        bestCLF = None
        for clf in clflist:
            if(clf.mse[-1] < minMSE):
                minMSE = clf.mse[-1]
                bestCLF = clf
        print(clf.h)
        """
        clf = TwoLayerFunctionApproximation(h=25, epochs=4000)
        clf.train(X, t, X_train, T_train, X_validation, T_validation, alpha=0,
            print_acc=True, animate=True, subsampling=False)

    else:
        print("Stop it")

