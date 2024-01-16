# Establishes functions to predict the signal for a given set of predicted parameters
import numpy as np
from numpy.polynomial.polynomial import Polynomial

def predict_polynomial_signal(parameters, signal_length):
    # Calculates a predicted polynomial signal for a given set of parameters
    # parameters is a list of polynomial parameters
    # signal_length is the length of the signal to be predicted
    # output is a list of the predicted signal with length signal_length
    signal = []
    for i in range(signal_length):
        signal.append(0)
        for j in range(len(parameters)):
            signal[i] += parameters[j] * (i ** j)
    return signal

def predict_polynomial(coefficients, data_points):
    """
    Evaluate a polynomial for a list of data points.

    :param coefficients: A list of coefficients (including the bias) for the polynomial.
    :param data_points: A list of data points to evaluate the polynomial at.
    :return: A list of values corresponding to the polynomial evaluated at each data point.
    """
    # Create a Polynomial object with the given coefficients
    poly = Polynomial(coefficients)

    # Evaluate the polynomial for each data point
    evaluated_points = [poly(x) for x in data_points]

    return evaluated_points



