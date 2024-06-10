import numpy as np
from Data_generation import equidistant_interpolation
from Train import ki_macro
from Plots import *
from scipy.optimize import curve_fit
import scipy.signal
from Data_generation import equidistant_interpolation


def fit_signals(inputs, t_values, parameter_bounds):
    num_signals = inputs.shape[0]
    fitted_params = np.zeros((num_signals, 4))
    for i in range(num_signals):
        try:
            popt, _ = curve_fit(model_function, t_values, inputs[i, :], p0=initial_guesses, bounds=parameter_bounds, maxfev=10000)
            fitted_params[i, :] = popt
        except Exception as e:
            print(f"Fit failed for signal {i} with error {e}")
            fitted_params[i, :] = np.nan  # Use NaN to indicate failed fits
        if i % 100 == 0:
            print(f"Finished fitting signal {i}")
    return fitted_params


def model_function(t, k1, k2, k3, vb):
    k4 = 0

    # Calculate alphas:
    alpha1 = (k2 + k3 + k4) - np.sqrt((k2 + k3 + k4)**2 - 4*k2*k4)
    alpha1 /= 2

    alpha2 = (k2 + k3 + k4) + np.sqrt((k2 + k3 + k4)**2 - 4*k2*k4)
    alpha2 /= 2

    # Calculate IRF:
    IRF = ( (k3 + k4 - alpha1) * np.exp(-alpha1 * t) + (alpha2 - k3 - k4) * np.exp(-alpha2 * t) ) / (alpha2 - alpha1)
    IRF *= k1

    # Convolve IRF with the vascular input function:
    c_tissue = scipy.signal.convolve(IRF, pchip_pl, mode='full')[:len(t)] * dt

    # Calculate TAC value:
    tac = c_tissue * (1 - vb) + pchip_bl * vb

    return tac


def add_ki_column(params):
    ki_values = ki_macro(params[:, 0], params[:, 1], params[:, 2])
    return np.concatenate([params, ki_values[:, np.newaxis]], axis=1)


def save_results(true_params, fitted_params):
    results = {
        "true_params": true_params,
        "predicted_params": fitted_params
    }
    results_list = [results]
    return results_list


# Main execution
if __name__ == "__main__":
    config = { 
    'data': np.load("data/Generated_Data/simulation_simple_0.0.npz"),
    'model_class': "Tradional fitter",
    'loss_function': "Nonlinear Least Squares Loss",
    'n_models': "",
    }
    # Fixed data values
    rtim_list = [0.125, 0.291667, 0.375, 0.458333, 0.583333, 0.75, 0.916667, 1.5, 2.5, 3.5, 4.5, 6.25, 8.75, 12.5, 
                 17.5, 25, 35, 45, 55, 65, 75, 85]
    pl_list = [0.05901, 0.0550587, 110.023, 83.0705, 55.6943, 44.4686, 36.9873, 27.5891, 13.5464, 6.33916, 3.52664, 
               2.49758, 1.44494, 1.04103, 0.71615, 0.52742, 0.43791, 0.35239, 0.32866, 0.27326, 0.26068, 0.24129]
    bl_list = [0.08737, 0.081723, 164.763, 125.181, 84.5654, 68.4428, 58.1711, 44.4919, 24.3138, 14.0272, 9.47834, 
               7.88443, 5.59859, 4.81384, 3.79608, 3.04436, 2.67881, 2.23747, 2.13455, 1.80559, 1.74821, 1.63998]
    
    # Interpolate plasma and blood concentration values
    num_equidistant_points = 2048
    new_rtim, _, _, pchip_pl = equidistant_interpolation(rtim_list, pl_list, num_equidistant_points)
    _, _, _, pchip_bl = equidistant_interpolation(rtim_list, bl_list, num_equidistant_points)
    parameter_bounds = ([0.13, 0.014, 0.025, 0.05], [0.41, 0.164, 0.725, 0.4])
    #parameter_bounds = ([0.01, 0.01, 0.01, 0.01], [1.0, 0.5, 2.0, 0.7]) # Recommended by Maqsood
    initial_guesses = [0.27, 0.089, 0.375, 0.225] # Average of each parameter value's range
    dt = new_rtim[1] - new_rtim[0]
    
    # Fit the signals
    inputs, true_params = config['data']["noisy_tacs"][:], config['data']["gt_parameters"][:, :-1]
    fitted_params = fit_signals(inputs, new_rtim, parameter_bounds)
    
    # Add Ki column to the fitted and true parameters
    fitted_params = add_ki_column(fitted_params)
    true_params = add_ki_column(true_params)
    
    # Save results
    results = save_results(true_params, fitted_params)
    print("Results processing complete.")

    # Plotting
    plot_mean_variance(results, config)
    distribution_mean_std(results)
    scatter_representation(results)

    # Ask the user if they want to save the results
    save = input("Save data? (y/n): ").lower() == 'y'

    if save:
        # Save the results to a .npz file
        np.savez('data/Fitted_Data/simulation_simple_0.0.npz', results=results)
        print("Data saved to data/Fitted_Data/simulation_simple_0.0.npz")
    
    else:
        print("Data saving cancelled.")