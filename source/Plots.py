import matplotlib.pyplot as plt
import numpy as np

def plot_mean_variance(results_list, config):
    """
    Plot the mean and variance of the results from multiple models.
    The function calculates the mean and standard deviation of the percentage differences and differences for each parameter.
    It then plots the average percentage differences and differences for each parameter.

    Parameters:
    results_list (list): A list of dictionaries containing the true and predicted parameters for each model.
    config (dict): A dictionary containing the configuration parameters for the model training.
    """
    # Extract the configuration parameters
    model_class = config['model_class']
    loss = config['loss_function']
    n_models = config['n_models']

    # Initialize lists to store the mean and standard deviation of the percentage differences and differences
    mean_percentage_diffs = []
    std_percentage_diffs = []
    mean_diffs = []
    std_diffs = []

    # Iterate over results list
    for i, results in enumerate(results_list):
        true_params = results["true_params"]
        predicted_params = results["predicted_params"]

        # Compute the percentile differences for each parameter
        diff = true_params - predicted_params
        epsilon = 1e-8  # Small constant to avoid division by zero
        percentage_diff = (diff / (true_params + epsilon)) * 100

        # Calculate the mean and standard deviation of the percentage differences
        mean_percentage_diff = np.mean(percentage_diff, axis=0)
        std_percentage_diff = np.std(percentage_diff, axis=0)

        # Calculate the mean and standard deviation of the differences
        mean_diff = np.mean(diff, axis=0)
        std_diff = np.std(diff, axis=0)

        # Append the results to the lists
        mean_percentage_diffs.append(mean_percentage_diff)
        std_percentage_diffs.append(std_percentage_diff)
        mean_diffs.append(mean_diff)
        std_diffs.append(std_diff)

    # Convert lists to 2D numpy arrays
    mean_percentage_diffs_array = np.array(mean_percentage_diffs)
    std_percentage_diffs_array = np.array(std_percentage_diffs)
    mean_diffs_array = np.array(mean_diffs)
    std_diffs_array = np.array(std_diffs)
    
    # Calculate the average of each column
    mean_percentage_diffs_avg = np.mean(mean_percentage_diffs_array, axis=0)
    std_percentage_diffs_avg = np.mean(std_percentage_diffs_array, axis=0)
    mean_diffs_avg = np.mean(mean_diffs_array, axis=0)
    std_diffs_avg = np.mean(std_diffs_array, axis=0)

    # Print the average values for each column
    print("Average of mean percentage differences:", mean_percentage_diffs_avg)
    print("Average of std percentage differences:", std_percentage_diffs_avg)
    print("Average of mean differences:", mean_diffs_avg)
    print("Average of std differences:", std_diffs_avg)

    # Custom parameter labels
    parameters = ['k1', 'k2', 'k3', 'vb', 'ki']

    # Plotting the percentile differences
    plt.figure(figsize=(10, 6))
    # Error bars for the standard deviation (symmetrical)
    errorbar_container = plt.errorbar(parameters, mean_percentage_diffs_avg, 
                yerr=[std_percentage_diffs_avg, std_percentage_diffs_avg],  # Symmetrical error bars
                fmt='s', capsize=5, capthick=2, ecolor='red', markersize=5, 
                linestyle='None', label='Average Std')

    # Annotating each point with its mean value +- std
    for i, (mean, std) in enumerate(zip(mean_percentage_diffs_avg, std_percentage_diffs_avg)):
        plt.annotate(f'{mean:.2f} ± {std:.2f}', (parameters[i], mean), textcoords="offset points", xytext=(10,0), ha='left')


    # Extending the graph to the right by adjusting x-axis limits
    plt.xlim(-0.25, len(parameters)-0.1)  # Set dynamic limits based on the number of parameters

    # Customizing the plot
    plt.title(f'Percentual difference in predictions: {model_class.__name__}, {loss} & {n_models} models')
    plt.xlabel('Parameter')
    plt.ylabel('Average Percentual Difference [%]')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.axhline(0, color='black', linestyle='--')  # Dashed line at 0

    # Handling legend for both Average Difference and Average Std
    plt.scatter(parameters, mean_percentage_diffs_avg, color='blue', label='Average Difference')
    plt.legend(handles=[plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=5, label='Average Percentual Difference'),
                        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=5, label='Average Std of Percentual Difference')],
            loc='best')

    # Show plot
    plt.show()

    # Plotting the absolute differences
    plt.figure(figsize=(10, 6))
    # Error bars for the standard deviation
    errorbar_container = plt.errorbar(parameters, mean_diffs_avg,
                    yerr=[std_diffs_avg, std_diffs_avg],  # Symmetrical error bars
                    fmt='s', capsize=5, capthick=2, ecolor='red', markersize=5,
                    linestyle='None', label='Average Std')

    # Annotating each point with its value
    for i, (mean, std) in enumerate(zip(mean_diffs_avg, std_diffs_avg)):
        plt.annotate(f'{mean:.2f} ± {std:.2f}', (parameters[i], mean), textcoords="offset points", xytext=(10,0), ha='left')

    # Extending the graph to the right by adjusting x-axis limits
    plt.xlim(-0.25, len(parameters)-0.1)  # Set dynamic limits based on the number of parameters

    # Customizing the plot
    plt.title(f'Difference in predictions: {model_class.__name__}, {loss} & {n_models} models')
    plt.xlabel('Parameter')
    plt.ylabel('Average Absolute Difference')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.axhline(0, color='black', linestyle='--')  # Dashed line at 0

    # Handling legend for both Average Difference and Average Std
    plt.scatter(parameters, mean_diffs_avg, color='blue', label='Average Difference')
    plt.legend(handles=[plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=5, label='Average Difference'),
                        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=5, label='Average Std of Difference')],
            loc='best')

    # Show plot
    plt.show()


def distribution_mean_std(results_list):
    """
    Plot the mean and standard deviation of the results from multiple models.
    The function calculates the mean and standard deviation of the percentage differences and differences for each parameter.
    It then plots the average percentage differences and differences for each parameter.

    Parameters:
    results_list (list): A list of dictionaries containing the true and predicted parameters for each model.
    """
    # Initialize lists to store the mean and standard deviation of the percentage differences and differences
    mean_percentage_diffs = []
    std_percentage_diffs = []
    mean_diffs = []
    std_diffs = []

    # Iterate over results list
    for i, results in enumerate(results_list):
        true_params = results["true_params"]
        predicted_params = results["predicted_params"]

        # Compute the percentile differences for each parameter
        diff = true_params - predicted_params
        epsilon = 1e-8  # Small constant to avoid division by zero
        percentage_diff = (diff / (true_params + epsilon)) * 100

        # Calculate the mean and standard deviation of the percentage differences
        mean_percentage_diff = np.mean(percentage_diff, axis=0)
        std_percentage_diff = np.std(percentage_diff, axis=0)

        # Calculate the mean and standard deviation of the differences
        mean_diff = np.mean(diff, axis=0)
        std_diff = np.std(diff, axis=0)

        # Append the results to the lists
        mean_percentage_diffs.append(mean_percentage_diff)
        std_percentage_diffs.append(std_percentage_diff)
        mean_diffs.append(mean_diff)
        std_diffs.append(std_diff)

    # Convert lists to 2D numpy arrays
    mean_percentage_diffs_array = np.array(mean_percentage_diffs)
    std_percentage_diffs_array = np.array(std_percentage_diffs)
    mean_diffs_array = np.array(mean_diffs)
    std_diffs_array = np.array(std_diffs)

    # Parameters labels
    parameters = ['k1', 'k2', 'k3', 'vb', 'ki']
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different colors for each parameter line

    # Creating side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Calculate individual max values for adjusting y-axis range
    mean_max_abs_value = np.max(np.abs(mean_percentage_diffs_array)) * 1.1
    std_max_value = np.max(std_percentage_diffs_array) * 1.1  # std values are only positive

    # Plotting mean percentual differences
    for idx, param in enumerate(parameters):
        axs[0].plot(range(1, len(mean_percentage_diffs_array)+1), mean_percentage_diffs_array[:, idx], label=param, color=colors[idx])
    axs[0].set_title('Mean Percentual Difference per Model')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('Mean Percentual Difference [%]')
    axs[0].axhline(y=0, color='black', linestyle='--')
    axs[0].set_xticks(range(1, len(mean_percentage_diffs_array) + 1))
    axs[0].set_ylim([-mean_max_abs_value, mean_max_abs_value])
    axs[0].set_yticks(np.linspace(-mean_max_abs_value, mean_max_abs_value, num=5))
    axs[0].legend()

    # Plotting std percentual differences
    for idx, param in enumerate(parameters):
        axs[1].plot(range(1, len(std_percentage_diffs_array)+1), std_percentage_diffs_array[:, idx], label=param, color=colors[idx])
    axs[1].set_title('Std Percentual Difference per Model')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Std Percentual Difference [%]')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].set_xticks(range(1, len(std_percentage_diffs_array) + 1))
    axs[1].set_ylim([0, std_max_value])  # Adjust y-axis for std values
    axs[1].set_yticks(np.linspace(0, std_max_value, num=5))
    axs[1].legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

    # Do the same for the differences
    # Creating side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    
    # Calculate individual max values for adjusting y-axis range
    mean_max_abs_value = np.max(np.abs(mean_diffs_array)) * 1.1
    std_max_value = np.max(std_diffs_array) * 1.1  # std values are only positive

    # Plotting mean differences
    for idx, param in enumerate(parameters):
        axs[0].plot(range(1, len(mean_diffs_array)+1), mean_diffs_array[:, idx], label=param, color=colors[idx])
    axs[0].set_title('Mean Difference per Model')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('Mean Difference')
    axs[0].axhline(y=0, color='black', linestyle='--')
    axs[0].set_xticks(range(1, len(mean_diffs_array) + 1))
    axs[0].set_ylim([-mean_max_abs_value, mean_max_abs_value])
    axs[0].set_yticks(np.linspace(-mean_max_abs_value, mean_max_abs_value, num=5))
    axs[0].legend()

    # Plotting std differences
    for idx, param in enumerate(parameters):
        axs[1].plot(range(1, len(std_diffs_array)+1), std_diffs_array[:, idx], label=param, color=colors[idx])
    axs[1].set_title('Std Difference per Model')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Std Difference')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].set_xticks(range(1, len(std_diffs_array) + 1))
    axs[1].set_ylim([0, std_max_value])  # Adjust y-axis for std values
    axs[1].set_yticks(np.linspace(0, std_max_value, num=5))
    axs[1].legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()


def scatter_representation(results_list):
    # Initialize arrays to store the true and predicted parameters
    true_params_array = np.array([results["true_params"] for results in results_list])
    predicted_params_array = np.array([results["predicted_params"] for results in results_list])

    # Reshaping arrays to aggregate all models together for each parameter
    true_params_array = true_params_array.reshape(-1, 5)  # Assuming the second dimension is 5 for the number of parameters
    predicted_params_array = predicted_params_array.reshape(-1, 5)

    # Parameters labels
    parameters = ['k1', 'k2', 'k3', 'vb', 'ki']

    # Create subplots for each parameter
    fig, axs = plt.subplots(1, 5, figsize=(25, 5), sharex=False, sharey=False)
    for i, ax in enumerate(axs):
        # Density plot for each parameter with the custom colormap
        hb = ax.hexbin(true_params_array[:, i], predicted_params_array[:, i], gridsize=50, cmap='viridis', mincnt=1)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")  # Diagonal line
        
        ax.set_title(parameters[i])
        ax.set_xlabel('True Value')
        if i == 0:
            ax.set_ylabel('Predicted Value')
        
        # Individual tick adjustment
        true_min, true_max = np.min(true_params_array[:, i]), np.max(true_params_array[:, i])
        pred_min, pred_max = np.min(predicted_params_array[:, i]), np.max(predicted_params_array[:, i])
        ax.set_xticks(np.linspace(true_min, true_max, num=5))
        ax.set_yticks(np.linspace(pred_min, pred_max, num=5))
    
    # Add a colorbar to show density scale, placed outside the subplots
    cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust these values as needed to fit your layout
    cb = fig.colorbar(hb, cax=cb_ax)
    cb.set_label('count in bin')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    plt.show()