#!/usr/bin/python3

"""lab1.py Containing the code for lab1

For the DD2437 Artificial Neural Networks and Deep Architectures course at KTH
Royal Institute of Technology"""

__author__ = "Anton, Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

### Classification with a single-layer perceptron

## 3.1.1 Generation of linearly-separable data
n = 100
mA, sigmaA = [1.0, 0.5], 0.5
mB, sigmaB = [-1.0, 0.0], 0.5

classA, classB = np.zeros((2, n)), np.zeros((2, n))
classA[0, :] = np.random.randn(1, n) * sigmaA + mA[0]
classA[1, :] = np.random.randn(1, n) * sigmaA + mA[1]
classB[0, :] = np.random.randn(1, n) * sigmaB + mB[0]
classB[1, :] = np.random.randn(1, n) * sigmaB + mB[1]

# Create a scatter plot
plt.scatter(classA[0, :], classA[1,:], color='red')
plt.scatter(classB[0, :], classB[1,:], color='green')
plt.xlabel("x"), plt.ylabel("y")
plt.title("Linearly separable data")
plt.show()
