#!/usr/bin/python3

"""helper.py Helper functions for data preparation and visualization purposes."""

__author__ = "Anton Anderzén, Stella Katsarou, Bas Straathof"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def generate_data(n, mA, sigmaA, mB, sigmaB, special_case=False):
    """Generates toy data

    Args:
        n (int): Number of data points to be generated per class
        mA (np.ndarray): Means of classA
        sigmaA (float): Variabnce of classA
        mB (np.ndarray): Means of classB
        sigmaB (float): Variance of classB
        special_case (bool): Flag to specify the special case of non-linearly
                             separable data at 3.1.3

    Returns:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB

    Note: a row in the lab description is a column in the code here, and vice
    versa. This simplifies shuffling, and doesn't have any adverse side-effects.
    """
    classA, classB = np.zeros((n, 2)), np.zeros((n, 2))

    if special_case:
        classA[:, 0] = np.hstack((np.random.randn(1, round(0.5*n))
            * sigmaA - mA[0], np.random.randn(1, round(0.5*n))
            * sigmaA + mA[0]))
    else:
        classA[:, 0] = np.random.randn(1, n) * sigmaA + mA[0]

    classA[:, 1] = np.random.randn(1, n) * sigmaA + mA[1]
    classB[:, 0] = np.random.randn(1, n) * sigmaB + mB[0]
    classB[:, 1] = np.random.randn(1, n) * sigmaB + mB[1]

    return classA, classB


def generate_encoder_data():
    """Generates all 8 combinations of one-hot encoded data 
            for a vector of size 8

    Returns:
        X (np.ndarray):
        classB (np.ndarray): Data points belonging to classB
    """
    X = np.zeros((8, 8))
    blank = -1*np.ones(8,)
    diag = 0
    for idx, val in enumerate(blank):
        blank = -1*np.ones(8,)
        if(idx == diag):
            temp = -1*np.ones(8,)
            temp[idx] = 1
            X[idx] = temp
            diag += 1
            continue
    return X, X


def subsample_data(classA, classB, percA, percB):
    """Subsample from classA and classB by percentages

    Args:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB
        percA (int): What percentage should be randomly selected from classA
        percB (int): What percentage should be randomly selected from classB

    Returns:
        classA_train (np.ndarray): Subsampled training set data points belonging to classA
        classB_train (np.ndarray): Subsampled training set data points belonging to classB
        classA_validation (np.ndarray): Subsampled validation set data points belonging to classA
        classB_validation (np.ndarray): Subsampled validation set data points belonging to classB
    """
    sizeA = round(classA.shape[0] / 100 * percA)
    sizeB = round(classB.shape[0] / 100 * percB)
    classA_train = classA[np.random.randint(classA.shape[0], size=sizeA), :]
    classB_train = classB[np.random.randint(classB.shape[0], size=sizeB), :]
    classA_validation = np.array([[x[0], x[1]] for x in classA if x not in classA_train])
    classB_validation = np.array([[x[0], x[1]] for x in classB if x not in classB_train])
    return classA_train, classB_train, classA_validation, classB_validation


def create_training_examples_and_targets(classA, classB):
    """Transforms toy data to trianing examples and targets

    Args:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB

    Returns:
        X (np.ndarray): Training data including bias term
        t (np.ndarray): Target vector

    """
    # Get number of data points per class
    n = classA.shape[0]
    m = classB.shape[0]

    # Add an bias row to the matrices representing the two different classes
    classA = np.hstack((classA, np.ones((n, 1))))
    classB = np.hstack((classB, np.ones((m, 1))))

    # Store the training data in a big matrix
    X = np.vstack((classA, classB))

    # Create a targets vector where classA = -1 and classB = 1
    t = np.vstack((np.ones((n, 1))*-1, np.ones((m, 1))))

    # Shuffle the training data and targets in a consistent manner
    X, t = shuffle(X, t, random_state=0)

    return X, t


def create_data_scatter_plot(classA, classB, linearly_separable=False, fname="",
        save_plot=False):
    """Creates a scatter plot of the input data

    Args:
        classA (np.ndarray): Data points belonging to classA
        classB (np.ndarray): Data points belonging to classB
        linearly_separable (bool): Flag to specify whether data is linearly
                                   separable
        fname (str): File name for saving
        save_plot (bool): Flag to specify whether to save the plot

    Returns:
        None
    """
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")

    if linearly_separable:
        plt.title("Linearly separable data")
    else:
        plt.title("Linearly inseparable data")

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def decision_boundary_animation(classA, classB, x, W, title, bias=True, fname="",
        save_plot=False):
    """Draws the decision boundary

    Args:
        classA (np.ndarray): The data corresponding to class A
        classB (np.ndarray): The data corresponding to class B
        x (np.ndarray): A linspace
        W (np.ndarrat): The weight vector
        title (str): Plot title
        bias (bool): Flag to determine whether to use the bias weight
        fname (str): File name for saving
        save_plot (bool): Flag to specify whether to save the plot

    Returns:
        None
    """
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")

    if bias:
        y = -(W[0]*x + W[2])/W[1] # y will coincide with x2 in the plot
        plt.title(title)
    else:
        y = -(W[0]*x)/W[1]
        plt.title(title + " without bias")

    plt.plot(x, y, '-b', label="line")
    plt.scatter(classA[:, 0], classA[:, 1], color='red')
    plt.scatter(classB[:, 0], classB[:, 1], color='green')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def approx_decision_boundary_animation(classA_train, classB_train, 
    classA_validation, classB_validation, net, title, fname="", save_plot=False):
    """Draws the approximated decision boundary e.g. network output = 0

    Args:
        classA (np.ndarray): The data corresponding to class A
        classB (np.ndarray): The data corresponding to class B
        x (np.ndarray): A linspace
        net (np.ndarrat): The network object.
        title (str): Plot title
        fname (str): File name for saving
        save_plot (bool): Flag to specify whether to save the plot

    Returns:
        None
    """
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-2, 2])
    plt.xlabel("x1"), plt.ylabel("x2")
    res = np.linspace(-2, 2, 1000)
    xlist, ylist = np.meshgrid(res, res)
    grid_data = np.vstack((np.ravel(xlist), np.ravel(ylist)))
    grid_data = np.vstack((grid_data, np.ones((1, len(grid_data[0])))))
    grid_data = np.transpose(grid_data)
    Z = net.predict(net.forward_pass(grid_data)[1])
    Z = np.reshape(Z, (len(xlist), len(xlist[0])))
    plt.contour(res, res, Z, [0], color='black')
    plt.scatter(classA_train[:, 0], classA_train[:, 1], color='red')
    plt.scatter(classB_train[:, 0], classB_train[:, 1], color='green')
    plt.scatter(classA_validation[:, 0], classA_validation[:, 1], color='red', marker='x')
    plt.scatter(classB_validation[:, 0], classB_validation[:, 1], color='green', marker='x')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_accuracy(train_acc, validation_acc, title):
    """Plots the accuracy over epochs

    Args:
        train_acc (list): List with precalculated training accuracy from each epoch in order
        validation_acc (list): List with precalculated validation accuracy from each epoch in order
        title (str): Plot title
    """
    training_acc_plot = plt.plot(range(len(train_acc)), train_acc, '-g', label="Training accuracy")
    validation_acc_plot = plt.plot(range(len(validation_acc)), validation_acc, '-b', label="Validation accuracy")
    plt.legend()
    plt.title(title)
    plt.show()

def mackey_glass(seq_length=2000, beta=0.2, gamma=0.1, tau=25,
        x0=1.5, n=10):
    """Generate a data sequence using the approximated Mackey Glass time-series

    Args:
      seq_length (int): The length of the output sequence
      beta (float): MG beta parameter
      gamma (float): MG gamma parameter
      tau (int): MG tau parameter
      x0 (float): Starting value of the sequence

    Returns:
      x (np.ndarray): Sequence based on Mackey-Glass time-series approximation
    """
    # Pre-allocate the array
    x = np.zeros((seq_length, 1))

    x[0] = x0
    for t in range(0, seq_length-1):
        x[t+1] = x[t] + beta * x[t - tau] / (1 + x[t - tau]**10) - gamma * x[t]

    return x


def create_mg_data(seq):
    """Create training examples and targets based on a MG time-series sequence

    Note:
        To predict x[t+5] we use x[t], x[t-5], x[t-10], x[t-15] and x[t-20]

    Args:
        seg (np.ndarray): Time-series sequence

    Returns:
        X (np.ndarray): Matrix of observations
        T (np.ndarray): Matrix of targets
    """
    # Pre-allocate arrays for X and T. The first 20 and last 5 values of seq
    # cannot be used for training, hence the -25
    X, T = np.zeros((len(seq)-25, 5)), np.zeros((len(seq)-25))

    # Iterate through all conceivable sequences of 5 training values
    for i in range(20, len(seq)-5):
        X[i-20] = [seq[i-20], seq[i-15], seq[i-10], seq[i-5], seq[i]]
        T[i-20] = seq[i+5]

    return X, T


def plot_mg_time_series(seqs, names, title="MG time-series", fname="",
        save_plot=False):
    """Generates a plot of the Mackey-Glass time-series

    Args:
        seqs (list): List of time series (np.ndarray)
        names (list): Names of the sequences (str)
        title (str): The title of the plot
        fname (str): File name for saving
        save_plot (bool): Flag to specify whether to save the plot

    Returns:
        None
    """
    plt.xlabel("x"), plt.ylabel("y")
    plt.title(title)

    for seq, name in zip(seqs, names):
        plt.plot(np.linspace(0, len(seq)-1, len(seq)), seq, label=f'{name}')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_weights(weights, alphas, title, fname="", save_plot=False):
    """Creates a weight histogram for the hidden layer of a two-layer perceptron
       based on the regularization constants alpha

    Args:
        weights (np.ndarray): Array containing all the weights
        alphas (list): List of regularization constants
        title (str): Plot title
        fname (str): File name for saving
        save_plot (bool): Flag to specify whether to save the plot

    Returns:
        None
    """
    for i, (weight, alpha) in enumerate(zip(weights, alphas)):
        plt.bar(x=np.asarray(range(1,len(weight)+1))+.5-1/len(weights)*i,
        width=1/len(weights), height=weight,
        label=f'alpha: {alpha}')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.xlabel('Weights')
    plt.tight_layout()
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


def test_hidden_layer_values(clf):
    """
        Wierd placeholder for trying to make sense of the weight vectors.
        Should be looking at input weights instead of output weights.
    """
    hidden = product([10, -10], repeat=3)
    hidden = list(hidden)
    result = []
    for comb in hidden:
        result.append(np.multiply(np.array([[comb[0]], [comb[1]], [comb[2]]]), clf.W))
    
    out = [0,0,0,0,0,0,0,0]
    for res in result:
        for idx, row in enumerate(res[0]):
            out[idx] += res[0][idx] + res[1][idx] + res[2][idx]
        print(out)
        out = [0,0,0,0,0,0,0,0]


def mean_squared_error(y_true, y_pred):
    """Calculate the mean squared error

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        (float): MSE
    """
    return np.square(np.subtract(y_true, y_pred)).mean()

