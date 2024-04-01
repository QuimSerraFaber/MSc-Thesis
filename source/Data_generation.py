import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Set the random seed for reproducibility
np.random.seed(42)

if __name__ == "__main__":
    # Load the file into a pandas DataFrame
    df = pd.read_csv('data/all_data_003.csv')

    # Show the first few rows of the DataFrame to confirm it's loaded correctly
    print("Dataframe head: \n", df.head())


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
    # Load data from the first row of the dataset
    data_from_first_row = DataLoader(0, df)

    # Output the data extracted from the first row to verify the DataLoader function
    print("Loaded data: \n", data_from_first_row)


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
    # Interpolate plasma concentration values:
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
    plt.ylabel(Type)
    plt.title(Type + ' vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Plot the interpolated plasma concentration values:
    plot_interpolations(data_from_first_row['rtim_list'], 
                        data_from_first_row['pl_list'], 
                        new_rtim, linear_pl, cubic_pl, pchip_pl, 'Plasma Concentration')


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
    # Calculate the IRF values:
    IRF_values = IRF(data_from_first_row['gt_parameters_list'], new_rtim)
    # print(IRF_values)


import scipy.signal

def c_tissue(IRF_values, pchip_pl, dt):
    """
    Calculates the simulated C_Tissue values for the given IRF and plasma concentration values.

    Parameters:
    IRF_values (list): The IRF values.
    pchip_pl (list): Plasma concentration values interpolated using PCHIP.
    dt (float): The time step increment.

    Returns:
    list: The simulated C_Tissue values.
    """
    num_points = len(IRF_values) 

    # Compute the convolution of the two lists
    simulated_c_tissue_values = scipy.signal.convolve(IRF_values, pchip_pl, mode='full')[:num_points]

    # Normalize the convolution result
    normalized_c_tissue = simulated_c_tissue_values * dt # Multiply by dt

    return normalized_c_tissue

if __name__ == "__main__":
    # Calculate the simulated C_Tissue values:
    dt = new_rtim[1] - new_rtim[0]
    simulated_c_tissue_values = c_tissue(IRF_values, pchip_pl, dt)

    # Divide simulated tac values by constant
    # WARNING: This is a temoporary fix to the scaling issue
    # simulated_c_tissue_values = [x / 1.24 for x in simulated_c_tissue_values]

if __name__ == "__main__":
    # Interpolate TAC values:
    new_rtim, linear_tac, cubic_tac, pchip_tac = equidistant_interpolation(data_from_first_row['rtim_list'],
                                                          data_from_first_row['tac_list'],
                                                          num_equidistant_points)
    # Plot the interpolated TAC values:
    plot_interpolations(data_from_first_row['rtim_list'], 
                        data_from_first_row['tac_list'], 
                        new_rtim, linear_tac, cubic_tac, pchip_tac, 'TAC')

if __name__ == "__main__":
    # Interpolate blood concentration values:
    new_rtim, linear_bl, cubic_bl, pchip_bl = equidistant_interpolation(data_from_first_row['rtim_list'],
                                                          data_from_first_row['bl_list'],
                                                          num_equidistant_points)

    # Plot the interpolated blood concentration values:
    plot_interpolations(data_from_first_row['rtim_list'], 
                        data_from_first_row['bl_list'], 
                        new_rtim, linear_bl, cubic_bl, pchip_bl, 'Blood Concentration')
    
    
def simulated_tac(c_tissue, gt_parameters_list, bl_list):
    """
    Calculates the simulated TAC values for the given C_Tissue and blood concentration values.

    Parameters:
    c_tissue (list): The C_Tissue values.
    gt_parameters_list (list): The ground truth parameters.
    bl_list (list): Blood concentration values.

    Returns:
    list: The simulated TAC values.
    """
    simulated_tac_values = []
    vb = gt_parameters_list[3]

    for i in range(len(c_tissue)):
        value = c_tissue[i] * (1-vb) + vb * bl_list[i]
        simulated_tac_values.append(value)

    return simulated_tac_values

if __name__ == "__main__":
    # Calculate the simulated TAC values:
    simulated_tac_values = simulated_tac(simulated_c_tissue_values, data_from_first_row['gt_parameters_list'], pchip_bl)

    # Plot the simulated C_Tissue values against the original TAC values:
    plt.figure(figsize=(10, 6))
    plt.plot(new_rtim, simulated_tac_values, label='Simulated TAC', color='red')
    plt.plot(new_rtim, simulated_c_tissue_values, label='Simulated C_Tissue', color='green')
    plt.xlabel('Time')
    plt.title('C-Tissue vs Simulated TAC')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Plot the simulated TAC values against the original TAC values:
    plt.figure(figsize=(10, 6))
    plt.plot(new_rtim, simulated_tac_values, label='Simulated TAC', color='red')
    plt.plot(new_rtim, pchip_tac, label='Original TAC', color='blue')
    plt.xlabel('Time')
    plt.ylabel('TAC')
    plt.title('Simulated TAC vs Original TAC')
    plt.legend()
    plt.grid(True)
    plt.show()


def adding_noise_simple(simulated_tac_values, new_rtim, original_time, COVi=None):
    """
    Adds normal noise to the simulated TAC values.

    Parameters:
    simulated_tac_values (list): The simulated TAC values.
    new_rtim (list): The new resampled time points.
    original_time (list): The original time points.
    COVi (float): The COVi value.

    Returns:
    list: The noisy TAC values.
    list: The added noise.
    float: The approximate COVi value.
    """
    # Convert inputs to numpy arrays for efficient computation
    simulated_tac_values = np.array(simulated_tac_values)
    new_rtim = np.array(new_rtim)
    original_time = np.array(original_time)

    # Get the indices of the closest points in original time to the new resampled time points
    closest_indices = [np.argmin(np.abs(new_rtim - t)) for t in original_time]
    
    # Select the corresponding TAC values based on the closest indices
    tac_values_at_closest_indices = simulated_tac_values[closest_indices]

    # Get the last three TAC values
    last_three_tac = tac_values_at_closest_indices[-3:]

    # Calculate the standard deviation using the last three TAC values
    std_dev = np.std(last_three_tac)

    # Calculate the mean of the last three TAC values
    mean = np.mean(last_three_tac)

    # Calculate approximate COVi:
    approx_COVi = std_dev / mean
    
    # Calculate the noise
    if COVi is not None: # If COVi is provided, use it to calculate the noise
        noise = np.random.normal(0, COVi * mean, len(simulated_tac_values))
    else: # If COVi is not provided, use the standard deviation to calculate the noise
        noise = np.random.normal(0, std_dev, len(simulated_tac_values))
        
    # Add the noise to the simulated TAC values
    noisy_tac = simulated_tac_values + noise

    return noisy_tac, noise, approx_COVi

if __name__ == "__main__":
    noisy_tac, noise, COVi = adding_noise_simple(simulated_tac_values, new_rtim, data_from_first_row['rtim_list'])

    # Plot the noisy TAC values
    plt.figure(figsize=(10, 6))
    plt.plot(new_rtim, noisy_tac, label='Simple noisy TAC', color='red')
    plt.plot(new_rtim, simulated_tac_values, label='Simulated TAC', color='blue')
    plt.xlabel('Time')
    plt.ylabel('TAC')
    plt.title('Noisy TAC')
    plt.text(x=max(new_rtim) * 0.85, y=max(noisy_tac) * 0.2, s=f'COVi: {COVi:.4f}', fontsize=12, color='black')
    plt.legend()
    plt.show()


def adding_noise_advanced(simulated_tac_values, new_rtim, type='Normal'):
    """
    Adds noise to the simulated TAC values in a more advanced way. The variance of the noise is calculated for each tac window.

    Parameters:
    simulated_tac_values (list): The simulated TAC values.
    new_rtim (list): The new time points.
    type (str): The type of noise to add. Either 'Poisson' or 'Normal'.

    Returns:
    list: The simulated TAC values with added noise.
    list: The added noise.
    """
    # List of original time points
    times = [0.125, 0.291667, 0.375, 0.458333, 0.583333, 0.75, 0.916667, 1.5, 2.5, 3.5, 4.5, 6.25, 8.75, 12.5, 17.5, 25, 35, 45, 55, 65, 75, 85]
    lengths = [0.166667, 0.083333, 0.083333, 0.125, 0.166667, 0.166667, 0.583333, 1, 1, 1, 1.75, 1.5, 3.75, 5, 7.5, 10, 10, 10, 10, 10, 10, 10]

    # Find new_rtim values that are closest to the original time points
    closest_indices = [np.argmin(np.abs(new_rtim - t)) for t in times]

    # Calculate the mean and standard deviation of the TAC values in each window
    tac_mean_values = []
    tac_std_dev_values = []
    for i in range(len(times)):
        if i == len(times) - 1:
            # Last interval
            interval_tac = simulated_tac_values[closest_indices[i]:]
        else:
            interval_tac = simulated_tac_values[closest_indices[i]:closest_indices[i+1]]
        tac_mean_values.append(np.mean(interval_tac))
        tac_std_dev_values.append(np.std(interval_tac))
    
    # Add noise to each window
    noisy_tac = []
    noise = []
    for i in range(len(times)):
        # Determine start index
        start_index = closest_indices[i]
        
        # Determine end index: If it's the last element, slice to the end of the array. Otherwise, use the next closest index.
        if i == len(times) - 1:
            end_index = None 
        else:
            end_index = closest_indices[i + 1]
        
        # Slicing tac_window and corresponding new_rtim values
        tac_window = simulated_tac_values[start_index:end_index]
        rtim_window = new_rtim[start_index:end_index]
        
        # Calculate the decay correction factor
        decay = np.log(2) / 109.8 # Fluorine-18 half-life is 109.8 minutes
        F_decay = 1 * np.exp(-decay * rtim_window)
        dcfi = np.trapz(F_decay, rtim_window) / lengths[i]

        # Calculate Ti
        Ti = tac_mean_values[i] * lengths[i] / dcfi

        # Calculate the local variance
        local_variance = (dcfi**2 / (lengths[i]**2)) * Ti
        
        # Generate the noise
        if type == 'Poisson':
            #local_noise = np.random.poisson(0.02 * tac_mean_values[i], size=len(tac_window))
            local_noise = np.random.poisson(np.sqrt(abs(local_variance)), size=len(tac_window))
        elif type == 'Normal':
            #local_noise = np.random.normal(0, 0.02 * tac_mean_values[i], size= len(tac_window))
            local_noise = np.random.normal(0, np.sqrt(abs(local_variance)), size=len(tac_window))
        
        noise += local_noise.tolist()

    # Add the noise to the simulated TAC values
    simulated_tac_values = np.array(simulated_tac_values)
    noise = np.array(noise)
    noisy_tac = simulated_tac_values + noise
        
    return noisy_tac, noise

if __name__ == "__main__":
    noisy_tac, noise = adding_noise_advanced(simulated_tac_values, new_rtim, 'Normal')

    # Plot the noisy TAC values
    plt.figure(figsize=(10, 6))
    plt.plot(new_rtim, noisy_tac, label='Advanced noisy TAC', color='red')
    plt.plot(new_rtim, simulated_tac_values, label='Simulated TAC', color='blue')
    plt.xlabel('Time')
    plt.ylabel('TAC')
    plt.title('Noisy TAC')
    plt.text(x=max(new_rtim) * 0.85, y=max(noisy_tac) * 0.2, s=f'COVi: {COVi:.4f}', fontsize=12, color='black')
    plt.legend()
    plt.show()


def generate_tac(data_row, num_points, type='Simple', COVi=None):
    """
    Generates the TAC signal for the given data row.

    Parameters:
    data_row (pd.Series): The data row containing the required information.
    num_points (int): The number of points to generate.
    type (str): The type of noise to add. Either 'Simple' or 'Advanced'.
    COVi (float): The COVi value.

    Returns:
    list: The new time points.
    list: The simulated TAC values.
    list: The generated TAC signal.
    """
    # Interpolate the required signals using PCHIP
    new_rtim, _, _, pchip_pl = equidistant_interpolation(data_row['rtim_list'], data_row['pl_list'], num_points)
    _, _, _, pchip_bl = equidistant_interpolation(data_row['rtim_list'], data_row['tac_list'], num_points)

    # Calculate the IRF values
    IRF_values = IRF(data_row['gt_parameters_list'], new_rtim)

    # Calculate the C_Tissue values
    dt = new_rtim[1] - new_rtim[0]
    simulated_c_tissue_values = c_tissue(IRF_values, pchip_pl, dt)

    # Calculate the simulated TAC values
    simulated_tac_values = simulated_tac(simulated_c_tissue_values, data_row['gt_parameters_list'], pchip_bl)

    # Add noise to the simulated TAC values
    if type == 'Simple': # Use the simple noise addition method
        noisy_tac, _, _ = adding_noise_simple(simulated_tac_values, new_rtim, data_row['rtim_list'], COVi)
    elif type == 'Advanced': # Use the advanced noise addition method
        noisy_tac, _ = adding_noise_advanced(simulated_tac_values, new_rtim, 'Normal')

    return new_rtim, simulated_tac_values, noisy_tac

if __name__ == "__main__":
    # Generate the TAC signal for a given row using the simple noise addition method
    data_row = DataLoader(466540, df)
    new_rtim, simulated_tac_values, noisy_tac = generate_tac(data_row, num_equidistant_points)

    # Plot the simulated and noisy TAC signals
    plt.figure(figsize=(10, 6))
    plt.plot(new_rtim, simulated_tac_values, label='Simulated TAC', color='blue', linewidth=2)
    plt.plot(new_rtim, noisy_tac, label='Noisy TAC', color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('TAC')
    plt.title('Simulated vs Simple noisy TAC')
    plt.legend()
    plt.show()

    # Generate the TAC signal for a given row using the advanced noise addition method
    data_row = DataLoader(466540, df)
    new_rtim, simulated_tac_values, noisy_tac = generate_tac(data_row, num_equidistant_points, type='Advanced')

    # Plot the simulated and noisy TAC signals
    plt.figure(figsize=(10, 6))
    plt.plot(new_rtim, simulated_tac_values, label='Simulated TAC', color='blue', linewidth=2)
    plt.plot(new_rtim, noisy_tac, label='Noisy TAC', color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('TAC')
    plt.title('Simulated vs Advanced noisy TAC')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Ask the user if they want to generate new data
    generate = input("Generate data? (y/n): ").lower() == 'y'

    if generate:
        # Save the generated TAC signal to a npz file
        noisy_tacs = []
        gt_parameters = []
        num_equidistant_points = 2048
        type = 'Advanced'
        COVi = 0.05

        for i in range(0, df.index[-1], 1): 
            data_row = DataLoader(i, df)
            _, _, noisy_tac = generate_tac(data_row, num_equidistant_points, type, COVi)

            # Append the noisy TAC and ground truth parameters to the lists
            noisy_tacs.append(noisy_tac)
            data_row['gt_parameters_list'].extend([0])  # Append an extra zero for k4
            gt_parameters.append(data_row['gt_parameters_list'])

            # Print the progress
            if i % 10000 == 0 and i != 0:
                print(i)
                print(data_row['gt_parameters_list'])
            
        # Convert the lists to numpy arrays
        noisy_tacs = np.array(noisy_tacs)
        gt_parameters = np.array(gt_parameters)

        # Save the arrays to a .npz file
        np.savez('data/Generated_Data/simulation_advanced.npz', noisy_tacs=noisy_tacs, gt_parameters=gt_parameters)
        print("Data saved to data/Generated_Data/simulation_advanced.npz")
    
    else:
        print("Data generation cancelled.")