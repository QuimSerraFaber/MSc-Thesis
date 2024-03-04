#Calculate the mean squared error between two signals
#y is the target signal
#y_hat is the predicted signal
#input is are two lists of equal length containing the signals

import numpy as np
import torch
from Data_simulation import IRF

def MSE(y, y_hat):
    """
    Calculates the Mean Squared Error (MSE) between two signals.

    :param y: List or numpy array of the target signal values.
    :param y_hat: List or numpy array of the predicted signal values.
                 Both y and y_hat should be of equal length.
    :return: The mean squared error as a float.
    """
    return np.mean(np.square(np.subtract(y, y_hat)))


def compute_parameter_loss(predicted_param, true_param, use_absolute=False):
    # Loss is the difference between the predicted and true parameter
    # Input must be torch tensors
    if use_absolute:
        # Compute the absolute difference
        return (predicted_param - true_param).abs().mean()
    else:
        # Compute the regular mean difference
        return (predicted_param - true_param).mean()
    

def TAC_loss(predicted_param, true_param, use_absolute=False):
    
    true_tac = []
    pred_tac = []

    # Get all timepoints
    timepoints = np.linspace(0.125, 90, 2048)

    # Calculate the impulse response functions:
    true_irf = IRF(true_param, timepoints)
    pred_irf = IRF(predicted_param, timepoints)

    # Calculate the C-Tissue values
    




