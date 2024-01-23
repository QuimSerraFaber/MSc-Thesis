# Loads the originally simulated data csv file
# Uses simulated_tac to generate the new simulated data for the same parameters

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Load the file into a pandas DataFrame
df = pd.read_csv('data/all_data_003.csv')

# Show the first few rows of the DataFrame to confirm it's loaded correctly
print(df.head())


def DataLoader(row, df):
    """
    Extracts rtim_list, pl_list, bl_list, tac_list, and ground truth parameters list from a given row of the dataset.

    Parameters:
    row (int): The index of the row in the dataframe from which to extract the data.
    df (pandas.DataFrame): The dataframe containing the dataset.

    Returns:
    dict: A dictionary containing the extracted lists and ground truth parameters.
    """
    # Initialize lists
    rtim_list = []
    pl_list = []
    bl_list = []
    tac_list = []
    gt_parameters_list = []

    # Extract rtim_, pl_, bl_, and tac_ values from the row
    for column in df.columns:
        if column.startswith('rtim_'):
            rtim_list.append(df.at[row, column])
        elif column.startswith('pl_') and not column.endswith('23'):  # Exclude the 23rd column for plasma
            pl_list.append(df.at[row, column])
        elif column.startswith('bl_') and not column.endswith('23'):  # Exclude the 23rd column for blood
            bl_list.append(df.at[row, column])
        elif column.startswith('tac_'):
            tac_list.append(df.at[row, column])
        elif column in ['k1', 'k2', 'k3', 'vb']:  # Ground truth parameters
            gt_parameters_list.append(df.at[row, column])

    # Return the lists and ground truth parameters in a dictionary
    return {
        'rtim_list': rtim_list,
        'pl_list': pl_list,
        'bl_list': bl_list,
        'tac_list': tac_list,
        'gt_parameters_list': gt_parameters_list
    }

# Usage example:
data_from_first_row = DataLoader(0, df)

# Output the data extracted from the first row to verify the DataLoader function
print(data_from_first_row)


def equidistant_interpolation(rtim_list, pl_list, num_points):
    """
    Performs equidistant interpolation on the given rtim_list and pl_list.

    Parameters:
    rtim_list (list): The original list of non-equidistant timepoints.
    pl_list (list): The list of plasma concentration values corresponding to rtim_list.
    num_points (int): The number of desired equidistant points.

    Returns:
    tuple: Two lists containing the new equidistant timepoints and interpolated plasma values.
    """
    # Generate equidistant timepoints
    min_time = min(rtim_list)
    max_time = max(rtim_list)
    equidistant_rtim = np.linspace(min_time, max_time, num_points)

    # Perform linear interpolation using numpy
    linear_interp_pl = np.interp(equidistant_rtim, rtim_list, pl_list)

    # Perform cubic interpolation using scipy (as an alternative example)
    cubic_interp_func = interp1d(rtim_list, pl_list, kind='cubic', fill_value="extrapolate")
    cubic_interp_pl = cubic_interp_func(equidistant_rtim)

    return equidistant_rtim, linear_interp_pl, cubic_interp_pl

# Example usage:
num_equidistant_points = 1500
new_rtim, linear_pl, cubic_pl = equidistant_interpolation(data_from_first_row['rtim_list'],
                                                          data_from_first_row['pl_list'],
                                                          num_equidistant_points)



import matplotlib.pyplot as plt

def plot_interpolations(rtim_list, pl_list, new_rtim, linear_pl, cubic_pl):
    """
    Plots the original plasma concentration values and the interpolated values.

    Parameters:
    rtim_list (list): Original list of timepoints.
    pl_list (list): Original list of plasma concentration values.
    new_rtim (list): Equidistant timepoints.
    linear_pl (list): Plasma concentration values interpolated linearly.
    cubic_pl (list): Plasma concentration values interpolated using cubic spline.
    """
    plt.figure(figsize=(10, 6))

    # Plot original data
    plt.plot(rtim_list, pl_list, 'o-', label='Original Data', color='blue')

    # Plot linear interpolation
    plt.plot(new_rtim, linear_pl, label='Linear Interpolation', color='red')

    # Plot cubic interpolation
    plt.plot(new_rtim, cubic_pl, label='Cubic Interpolation', color='green')

    plt.xlabel('Time')
    plt.ylabel('Plasma Concentration')
    plt.title('Plasma Concentration vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
plot_interpolations(data_from_first_row['rtim_list'], 
                    data_from_first_row['pl_list'], 
                    new_rtim, linear_pl, cubic_pl)
