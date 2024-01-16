#Calculate the mean squared error between two signals
#y is the target signal
#y_hat is the predicted signal
#input is are two lists of equal length containing the signals

import numpy as np

def MSE(y, y_hat):
    """
    Calculates the Mean Squared Error (MSE) between two signals.

    :param y: List or numpy array of the target signal values.
    :param y_hat: List or numpy array of the predicted signal values.
                 Both y and y_hat should be of equal length.
    :return: The mean squared error as a float.
    """
    return np.mean(np.square(np.subtract(y, y_hat)))

def MSE_grad(y, y_hat):
    #calculate the gradient of the mean squared error
    #y is the target signal
    #y_hat is the predicted signal
    #input is are two lists of equal length containing the signals
    #output is the gradient of the mean squared error
    return np.subtract(y_hat, y)



