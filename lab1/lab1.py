#!/usr/bin/python3

"""lab1.py Containing the code for lab1

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton, Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(42)

### Classification with a single-layer perceptron

## 3.1.1 Generation of linearly-separable data
n = 100
mA, sigmaA = [1.0, 1.0], 0.5
mB, sigmaB = [-1.0, -0.5], 0.5

# Let's swap columns and rows, since this simplifies shuffling
classA, classB = np.zeros((n, 2)), np.zeros((n, 2))
classA[:, 0] = np.random.randn(1, n) * sigmaA + mA[0]
classA[:, 1] = np.random.randn(1, n) * sigmaA + mA[1]
classB[:, 0] = np.random.randn(1, n) * sigmaB + mB[0]
classB[:, 1] = np.random.randn(1, n) * sigmaB + mB[1]

# Create a scatter plot
plt.scatter(classA[:, 0], classA[:, 1], color='red')
plt.scatter(classB[:, 0], classB[:, 1], color='green')
plt.xlabel("x1"), plt.ylabel("x2")
plt.title("Linearly separable data")
#plt.show()

## 3.1.2 Classification with a single-layer perceptron and analysis

# Implementation of the Delta Rule:

# Add an bias row to the matrices representing the two different classes
classA = np.hstack((classA, np.ones((n, 1))))
classB = np.hstack((classB, np.ones((n, 1))))

# Store the training data in a shuffled way
X = np.vstack((classA, classB))

# Create a targets vector where classA = -1 and classB = 1
t = np.vstack((np.ones((n, 1))*-1, np.ones((n, 1))))

# Shuffle the data consistently
X, t = shuffle(X, t, random_state=0)

# Initialize the weight matrix
W = np.random.rand(3, 1)

# Set the learning rate
eta = 0.1

# The Delta Rule in matrix form will now be:
# dW = - eta * X.T @ (X@W - t)
# For the dimensions to match

