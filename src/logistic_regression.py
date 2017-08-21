import pandas as pd
import numpy as np


def my_logistic_regression(df_competition):
    """
    Computes optimal parameter values theta using logistic regression
    and batch gradient descent cost function minimization.

    :param df_competition: Pandas dataframe containing training data
    :returns: vector of optimal parameter values theta
    """
    data = df_competition.values  # convert data frame to numpy array
    m, n = data[:, :-1].shape
    X = np.concatenate((np.ones((m, 1)), data[:, :-1]), axis=1)
    y = data[:, -1]

    # set params
    initial_theta = np.zeros(X.shape[1])  # initialize theta to zeroes
    alpha = 0.01  # set learning rate
    numIterations = 10000  # set number of steps
    lamda = 0.1  # set value of regularization parameter

    # compute theta
    theta = gradientDescent(X, y, initial_theta, alpha, numIterations, lamda)
    return theta


def sigmoid(z):
    """
    Computes the sigmoid function for all values of z.

    :param z: scalar, vector or array of values for which the sigmoid
    function should be computed
    :returns: scalar, vector or array of values of size z containing computed
    sigmoid values
    """
    return (1 / (1 + np.power(np.e, -z)))


def costFunction(theta, X, y, lamda=0.1):
    """
    Computes the cost of using theta as the parameter for logistic regression
    and the gradient of the cost w.r.t. to the parameters.

    :param theta: parameter vector containing corresponding values of theta
    :param X: MxN array that contains feature set (x-values)
    :param y: Mx1 array that contains resulting outcomes (y-values)
    :param lamda: regularization parameter
    :returns: a scalar of the cost "J"
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))  # get predictions
    regTerm = (float(lamda) / 2) * theta**2
    cost = -(1 / m) * (np.log(h).T.dot(y) +
                       np.log(1 - h).T.dot(1 - y))  # calculate cost
    J = cost + (sum(regTerm[1:]) / m)

    return J


def calculateGradient(theta, X, y, lamda=0.1):
    """
    Computes the gradient of the cost with respect to cost J and theta.

    :param theta: parameter vector containing corresponding values of theta
    :param X: MxN array that contains feature set (x-values)
    :param y: Mx1 array that contains resulting outcomes (y-values)
    :param lamda: regularization parameter
    :returns: gradient of the cost with the same size as theta
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    regTerm = float(lamda) * (theta / m)
    return (np.dot(X.transpose(), h - y) / m) + regTerm.T


def gradientDescent(X, y, theta, alpha, numIterations, lamda=0.1):
    """
    Runs gradient descent to optimize a cost function for linear regression
    and returns the optimal parameter values theta.

    :param X: MxN array that contains feature set (x-values)
    :param y: Mx1 array that contains resulting outcomes (y-values)
    :param alpha: scalar value that defines the learning rate for gradient
    descent
    :param theta: parameter vector containing corresponding values of theta
    :param y: Mx1 array that contains resulting outcomes (y-values)
    :param lamda: regularization parameter
    :returns: optimal parameter set "theta"
    """
    J_history = np.zeros(numIterations)

    for i in range(0, numIterations):
        J_history[i] = costFunction(theta, X, y, lamda)  # calculate cost
        gradient = calculateGradient(theta, X, y, lamda)
        theta = theta - (alpha * gradient)  # update theta

    return theta
