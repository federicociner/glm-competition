import pandas as pd
import numpy as np
import random
from scipy.optimize import minimize

def my_logistic_regression(df_competition):
    """
    Computes
    """

    # format data
    data = df_competition.values  # convert data frame to numpy array
    m, n = data[:, :-1].shape
    X = np.concatenate((np.ones((m, 1)), data[:, :-1]), axis=1)
    y = data[:, -1]

    # optimize theta
    initial_theta = np.zeros(X.shape[1])
    res = minimize(costFunction, initial_theta, args=(X, y),
               method=None, jac=calculateGradient, options={'maxiter': 1000})
    return res.x


def sigmoid(z):
    """
    Computes the sigmoid function for all values of z.

    :param z: scalar, vector or array of values for which the sigmoid function should be computed
    :returns: scalar, vector or array of values of size z containing computed sigmoid values
    """
    return (1 / (1 + np.power(np.e, -z)))


def costFunction(theta, X, y):
    """
    Computes the cost of using theta as the parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters

    :param theta: parameter vector containing corresponding values of theta
    :param X: MxN array that contains feature set (x-values)
    :param y: Mx1 array that contains resulting outcomes (y-values)
    :returns: a scalar of the cost "J" and gradient "grad" of the cost with the same size as theta
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))  # get predictions
    J = -(1 / m) * (np.log(h).T.dot(y) +
                    np.log(1 - h).T.dot(1 - y))  # calculate cost

    return J


def calculateGradient(theta, X, y):
    """
    Computes the gradient of the cost with respect to cost J and theta.

    :param theta: parameter vector containing corresponding values of theta
    :param X: MxN array that contains feature set (x-values)
    :param y: Mx1 array that contains resulting outcomes (y-values)
    :returns: gradient of the cost with the same size as theta
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    return (1 / m) * np.dot(X.transpose(), h - y)


def gradientDescent(X, y, theta, alpha, numIterations):
    """
    Runs gradient descent to optimize a cost function for linear regression
    and returns the optimal parameter values theta.

    :param X: MxN array that contains feature set (x-values)
    :param y: Mx1 array that contains resulting outcomes (y-values)
    :param alpha: scalar value that defines the learning rate for gradient descent
    :param theta: parameter vector containing corresponding values of theta
    :returns: optimal parameter set "theta" and an array "J_history" containing the values of J
    for each iteration
    """
    m = len(y)
    J_history = np.zeros(numIterations)

    for i in range(0, numIterations):
        pred = np.dot(X, theta)  # get predictions
        loss = pred - y  # calculate loss
        J_history[i] = costFunction(theta, X, y)  # calculate cost
        gradient = calculateGradient(theta, X, y)
        theta = theta - (alpha * gradient)  # update theta

    return theta, J_history
