# Loads the originally simulated data csv file
# Uses simulated_tac to generate the new simulated data for the same parameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

if __name__ == "__main__":
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

if __name__ == "__main__":
    # Usage example:
    data_from_first_row = DataLoader(0, df)

    # Output the data extracted from the first row to verify the DataLoader function
    print(data_from_first_row)


from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator

def equidistant_interpolation(rtim_list, pl_list, num_points):
    """
    Performs equidistant interpolation on the given rtim_list and pl_list.

    Parameters:
    rtim_list (list): The original list of non-equidistant timepoints.
    pl_list (list): The list of plasma concentration values corresponding to rtim_list.
    num_points (int): The number of desired equidistant points.

    Returns:
    tuple: Lists containing the new equidistant timepoints and interpolated plasma values.
    """
    # Generate equidistant timepoints
    min_time = min(rtim_list)
    max_time = max(rtim_list) + 5  # Add 5 to max_time to ensure the last timepoint is included
    equidistant_rtim = np.linspace(min_time, max_time, num_points)

    # Perform linear interpolation using numpy
    linear_interp_pl = np.interp(equidistant_rtim, rtim_list, pl_list)

    # Perform cubic interpolation using scipy (as an alternative example)
    cubic_interp_func = interp1d(rtim_list, pl_list, kind='cubic', fill_value="extrapolate")
    cubic_interp_pl = cubic_interp_func(equidistant_rtim)

    # Perform monotonic cubic interpolation using PchipInterpolator
    pchip_interp_func = PchipInterpolator(rtim_list, pl_list)
    pchip_interp_pl = pchip_interp_func(equidistant_rtim)

    return equidistant_rtim, linear_interp_pl, cubic_interp_pl, pchip_interp_pl

if __name__ == "__main__":
    # Example usage:
    num_equidistant_points = 1024
    new_rtim, linear_pl, cubic_pl, pchip_pl = equidistant_interpolation(data_from_first_row['rtim_list'],
                                                            data_from_first_row['pl_list'],
                                                            num_equidistant_points)


def plot_interpolations(rtim_list, pl_list, new_rtim, linear_pl, cubic_pl, pchip_pl, Type):
    """
    Plots the original plasma concentration values and the interpolated values.

    Parameters:
    rtim_list (list): Original list of timepoints.
    pl_list (list): Original list of plasma concentration values.
    new_rtim (list): Equidistant timepoints.
    linear_pl (list): Plasma concentration values interpolated linearly.
    cubic_pl (list): Plasma concentration values interpolated using cubic spline.
    pchip_pl (list): Plasma concentration values interpolated using PCHIP.
    Type (str): The type of data being plotted.
    """
    plt.figure(figsize=(10, 6))

    # Plot original data
    plt.plot(rtim_list, pl_list, 'o-', label='Original Data', color='blue')

    # Plot linear interpolation
    plt.plot(new_rtim, linear_pl, label='Linear Interpolation', color='red')

    # Plot cubic interpolation
    plt.plot(new_rtim, cubic_pl, label='Cubic Interpolation', color='green')

    # Plot pchip interpolation
    plt.plot(new_rtim, pchip_pl, label='PCHIP Interpolation', color='orange')

    plt.xlabel('Time')
    plt.ylabel(Type + ' Concentration')
    plt.title(Type + ' Concentration vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    plot_interpolations(data_from_first_row['rtim_list'], 
                        data_from_first_row['pl_list'], 
                        new_rtim, linear_pl, cubic_pl, pchip_pl, 'Plasma')


def IRF(gt_parameters_list, equidistant_rtim):
    """
    Calculates the impulse response function (IRF) for the given ground truth parameters and equidistant timepoints.

    Parameters:
    gt_parameters_list (list): List of ground truth parameters.
    equidistant_rtim (list): Equidistant timepoints.

    Returns:
    list: The IRF values.
    """
    # Extract ground truth parameters
    k1 = gt_parameters_list[0]
    k2 = gt_parameters_list[1]
    k3 = gt_parameters_list[2]
    k4 = 0 # For the current data, k4 is always 0

    # Calculate alphas:
    alpha1 = (k2 + k3 + k4) - np.sqrt((k2 + k3 + k4)**2 - 4*k2*k4)
    alpha1 /= 2

    alpha2 = (k2 + k3 + k4) + np.sqrt((k2 + k3 + k4)**2 - 4*k2*k4)
    alpha2 /= 2

    # Calculate IRF
    IRF = []
    for t in equidistant_rtim:
        value = ( (k3 + k4 - alpha1) * np.exp(-alpha1 * t) + (alpha2 - k3 - k4) * np.exp(-alpha2 * t) ) / (alpha2 - alpha1)
        value *= k1
        IRF.append(value)
    
    return IRF

if __name__ == "__main__":
    # Example usage
    IRF_values = IRF(data_from_first_row['gt_parameters_list'], new_rtim)
    # print(IRF_values)


