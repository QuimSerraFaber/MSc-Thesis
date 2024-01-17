# Establishes functions to predict the signal for a given set of predicted parameters
import numpy as np
from numpy.polynomial.polynomial import Polynomial

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



