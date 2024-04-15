import numpy as np
from Data_generation import equidistant_interpolation
from Train import ki_macro
from Plots import *
from scipy.optimize import curve_fit
from Data_generation import equidistant_interpolation


def fit_signals(inputs, t_values, parameter_bounds):
    num_signals = inputs.shape[0]
    fitted_params = np.zeros((num_signals, 4))
    for i in range(num_signals):
        try:
            popt, _ = curve_fit(model_function, t_values, inputs[i, :], bounds=parameter_bounds)
            fitted_params[i, :] = popt
        except Exception as e:
            print(f"Fit failed for signal {i} with error {e}")
            fitted_params[i, :] = np.nan  # Use NaN to indicate failed fits
        if i % 1000 == 0:
            print(f"Finished fitting signal {i}")
    return fitted_params

def model_function(t, k1, k2, k3, vb):
    k4 = 0

    # Calculate alphas:
    alpha1 = (k2 + k3 + k4) - np.sqrt((k2 + k3 + k4)**2 - 4*k2*k4)
    alpha1 /= 2

    alpha2 = (k2 + k3 + k4) + np.sqrt((k2 + k3 + k4)**2 - 4*k2*k4)
    alpha2 /= 2

    IRF = ( (k3 + k4 - alpha1) * np.exp(-alpha1 * t) + (alpha2 - k3 - k4) * np.exp(-alpha2 * t) ) / (alpha2 - alpha1)
    IRF *= k1
    return IRF

def add_ki_column(params):
    ki_values = ki_macro(params[:, 0], params[:, 1], params[:, 2])
    return np.concatenate([params, ki_values[:, np.newaxis]], axis=1)

def save_results(true_params, fitted_params):
    results = {
        "true_params": true_params,
        "predicted_params": fitted_params
    }
    results_list = [results]  # Not clear why this needs to be a list, consider directly returning `results`
    return results_list

# Main execution
if __name__ == "__main__":
    config = { 
    'data': np.load("data/Generated_Data/simulation_simple_0.01.npz"),
    'model_class': "Tradional fitter",
    'loss_function': "Nonlinear Least Squares Loss",
    'n_models': "",
}
    t_values = np.linspace(0, 90, 2048)
    parameter_bounds = ([0.13, 0.014, 0.025, 0.05], [0.41, 0.164, 0.725, 0.4])
    
    inputs, true_params = config['data']["noisy_tacs"], config['data']["gt_parameters"][:, :-1]
    fitted_params = fit_signals(inputs, t_values, parameter_bounds)
    
    fitted_params = add_ki_column(fitted_params)
    true_params = add_ki_column(true_params)
    
    results = save_results(true_params, fitted_params)
    print("Results processing complete.")

    # Plotting
    #plot_mean_variance(results, config)
    distribution_mean_std(results)
    scatter_representation(results)