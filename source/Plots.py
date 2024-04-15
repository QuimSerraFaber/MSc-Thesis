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
    Plot the mean and standard deviation of the results from multiple models using points.
    The function calculates the mean and standard deviation of the percentage differences and differences for each parameter.
    It then plots the average percentage differences and differences for each parameter as separate points with unique colors and shapes.

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

    # Parameters labels, colors, and markers
    parameters = ['k1', 'k2', 'k3', 'vb', 'ki']
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different colors for each parameter
    markers = ['o', '^', 's', 'P', '*']  # Different shapes for each parameter

    # Creating side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Calculate individual max values for adjusting y-axis range
    mean_max_abs_value = np.max(np.abs(mean_percentage_diffs_array)) * 1.1
    std_max_value = np.max(std_percentage_diffs_array) * 1.1  # std values are only positive

    # Add semi-transparent grid
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Plotting mean percentual differences with points
    for idx, param in enumerate(parameters):
        axs[0].scatter(range(1, len(mean_percentage_diffs_array)+1), mean_percentage_diffs_array[:, idx], label=param, color=colors[idx], marker=markers[idx])
    axs[0].set_title('Mean Percentual Difference per Model')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('Mean Percentual Difference [%]')
    axs[0].axhline(y=0, color='black', linestyle='--')
    axs[0].set_xticks(range(1, len(mean_percentage_diffs_array) + 1))
    axs[0].set_ylim([-mean_max_abs_value, mean_max_abs_value])
    axs[0].set_yticks(np.linspace(-mean_max_abs_value, mean_max_abs_value, num=5))
    axs[0].legend()

    # Plotting std percentual differences with points
    for idx, param in enumerate(parameters):
        axs[1].scatter(range(1, len(std_percentage_diffs_array)+1), std_percentage_diffs_array[:, idx], label=param, color=colors[idx], marker=markers[idx])
    axs[1].set_title('Std Percentual Difference per Model')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Std Percentual Difference [%]')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].set_xticks(range(1, len(std_percentage_diffs_array) + 1))
    axs[1].set_ylim([0, std_max_value])  # Adjust
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

    # Add semi-transparent grid
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Plotting mean differences with points
    for idx, param in enumerate(parameters):
        axs[0].scatter(range(1, len(mean_diffs_array)+1), mean_diffs_array[:, idx], label=param, color=colors[idx], marker=markers[idx])
    axs[0].set_title('Mean Difference per Model')
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('Mean Difference')
    axs[0].axhline(y=0, color='black', linestyle='--')
    axs[0].set_xticks(range(1, len(mean_diffs_array) + 1))
    axs[0].set_ylim([-mean_max_abs_value, mean_max_abs_value])
    axs[0].set_yticks(np.linspace(-mean_max_abs_value, mean_max_abs_value, num=5))
    axs[0].legend()

    # Plotting std differences with points
    for idx, param in enumerate(parameters):
        axs[1].scatter(range(1, len(std_diffs_array)+1), std_diffs_array[:, idx], label=param, color=colors[idx], marker=markers[idx])
    axs[1].set_title('Std Difference per Model')
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Std Difference')
    axs[1].axhline(y=0, color='black', linestyle='--')
    axs[1].set_xticks(range(1, len(std_diffs_array) + 1))
    axs[1].set_ylim([0, std_max_value])  # Adjust
    axs[1].set_yticks(np.linspace(0, std_max_value, num=5))
    axs[1].legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()


def scatter_representation(results_list):
    """
    Create a scatter plot to visualize the true and predicted parameters of the best performing model.

    Parameters:
    results_list (list): A list of dictionaries containing the true and predicted parameters for each model.
    """
    # Initialize list to store the sum of mean percentage differences for each model
    sum_mean_percentage_diffs = []

    # Iterate over results list to calculate mean percentage differences
    for results in results_list:
        true_params = np.array(results["true_params"])
        predicted_params = np.array(results["predicted_params"])
        
        # Compute the percentage differences for each parameter
        epsilon = 1e-8  # Small constant to avoid division by zero
        percentage_diff = (true_params - predicted_params) / (true_params + epsilon) * 100
        
        # Calculate the mean percentage difference across all parameters for this model and sum them
        mean_percentage_diff = np.mean(np.abs(percentage_diff), axis=1)
        sum_mean_percentage_diffs.append(np.sum(mean_percentage_diff))
    
    # Find the index of the model with the lowest sum of mean percentage differences
    best_model_index = np.argmin(sum_mean_percentage_diffs)

    # Proceed to plot the true and predicted parameters for the best model
    best_model_results = results_list[best_model_index]
    true_params = np.array(best_model_results["true_params"])
    predicted_params = np.array(best_model_results["predicted_params"])

    # Parameters labels
    parameters = ['k1', 'k2', 'k3', 'vb', 'ki']

    # Create subplots for each parameter in a 2x3 grid, adjusting the figsize if needed
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=False, sharey=False)

    # Flatten axs to make it easier to iterate over
    axs = axs.flatten()

    # Loop through the first 5 plots to create hexbin plots
    for i in range(5):
        ax = axs[i]
        # Density plot for each parameter with the custom colormap
        hb = ax.hexbin(true_params[:, i], predicted_params[:, i], gridsize=50, cmap='viridis_r', mincnt=1)

        ax.set_title(parameters[i])
        ax.set_xlabel('True Value')
        if i % 3 == 0:  # Only set ylabel for the first plot of each row
            ax.set_ylabel('Predicted Value')

        # Setting the same limits for x and y axes with margin
        combined_min = min(np.min(true_params[:, i]), np.min(predicted_params[:, i]))
        combined_max = max(np.max(true_params[:, i]), np.max(predicted_params[:, i]))
        range_val = combined_max - combined_min
        margin = range_val * 0.05
        final_min, final_max = combined_min - margin, combined_max + margin
        ax.set_xlim(combined_min - margin, combined_max + margin)
        ax.set_ylim(combined_min - margin, combined_max + margin)

        # Draw the diagonal line after setting the final axis limits
        ax.plot([final_min, final_max], [final_min, final_max], ls="--", c=".3")

        ticks = np.linspace(combined_min - margin, combined_max + margin, num=5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    # Remove the last (unused) subplot to maintain the 3-2 layout
    fig.delaxes(axs[-1])

    # Adjusting the layout to make room for colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Add titles for the entire figure and the best model number
    fig.suptitle(f'Scatter Representation of True vs Predicted Parameters: model {best_model_index + 1}', fontsize=16)
    fig.subplots_adjust(top=0.9)

    # Create an axis for the colorbar. Adjust the position to not overlap plots.
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(hb, cax=cb_ax)
    cb.set_label('count in bin')

    plt.show()