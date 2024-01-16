# Generates data for a fifth degree polynomial (6 parameters) to test the model

import numpy as np

def generate_polynomial_data(coefficients, x, noise_level=0.1):
    """
    Generates data points for a fifth-degree polynomial.

    :param coefficients: List of coefficients (a0, a1, a2, a3, a4, a5) for the polynomial.
    :param x: List of points for which to calculate the polynomial.
    :param noise_level: Standard deviation of Gaussian noise added to the data.
    :return: y values as numpy arrays.
    """

    y = np.polyval(coefficients[::-1], x)  # Reverses the list to fit np.polyval's order
    y += np.random.normal(0, noise_level, y.shape)  # Adding noise to the data

    return y

# Coefficients for the polynomial (a0, a1, a2, a3, a4)
# coefficients = [1, -2, 3, -1, 0.5]