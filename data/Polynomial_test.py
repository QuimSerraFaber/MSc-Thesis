# Generates data for a fifth degree polynomial (6 parameters) to test the model

import numpy as np

def generate_polynomial_data(coefficients, x, noise_level=0.1):
    """
    Generates data points for a nth-degree polynomial.

    :param coefficients: List of coefficients (a0, a1, a2, a3, a4, ...) for the polynomial.
    :param x: List of points for which to calculate the polynomial.
    :param noise_level: Standard deviation of Gaussian noise added to the data.

    :return: y values as numpy arrays.
    """

    y = np.polyval(coefficients[::-1], x)  # Reverses the list to fit np.polyval's order
    y += np.random.normal(0, noise_level, y.shape)  # Adding noise to the data

    return y


def load_polynomial_data(num_samples=10000, x_range=(-50, 50), num_points=10000, noise_level=0.1, degree=3):
    """
    Generates data for a nth-degree polynomial (n coefficients) to test the model.

    :param num_samples: Number of samples to generate.
    :param x_range: Range of x values to generate.
    :param num_points: Number of points to generate within the range.

    :return: Tuple of input data and true parameters.
    """
    inputs = []
    true_params = []

    for _ in range(num_samples):
        coefficients = np.random.randn(degree + 1)  # Generate random coefficients for a nth-degree polynomial
        x = np.linspace(x_range[0], x_range[1], num_points)  # Generate evenly spaced points within the range
        y = generate_polynomial_data(coefficients, x, noise_level)

        inputs.append(y)
        true_params.append(coefficients)

    return np.array(inputs), np.array(true_params)